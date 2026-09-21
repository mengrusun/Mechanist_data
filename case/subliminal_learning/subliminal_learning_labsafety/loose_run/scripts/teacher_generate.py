"""
M0.α: Generate teacher responses on the ~10k+ lab-safety prompt pool.

- enable_thinking=False, temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=256.
- Data-parallel across visible GPUs via manual per-GPU sharding (no torchrun
  needed — each GPU loads its own replica and processes a disjoint slice).
- Optional LoRA adapter path (--adapter). Without it, uses the base model
  (used for the Ctrl-B arm).

Launch (per-GPU-shard mode, all 4 GPUs in one process via multiprocessing):
    python scripts/teacher_generate.py \
        --prompts data/prompts/lab_safety_prompts.jsonl \
        --out data/gen/treated_raw.jsonl \
        --adapter ckpt/teacher_lora --gpu_ids 0,1,2,3
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Local import
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import (MODEL_PATH, ROOT, set_seed, read_jsonl, write_jsonl,
                    append_jsonl, log_meta)


def _shard_worker(gpu_local_id: int, world_size: int, prompts: list[dict],
                  out_path: str, args_dict: dict) -> None:
    """
    Run inside one process pinned to a single GPU. Loads one full replica
    and generates for its shard of prompts.
    """
    # ---- pin this process to a single GPU ----
    # gpu_local_id is the local rank within CUDA_VISIBLE_DEVICES ordering
    os.environ["LOCAL_RANK"] = str(gpu_local_id)
    os.environ.pop("WORLD_SIZE", None)
    # After the parent sets CUDA_VISIBLE_DEVICES=<a,b,c,d>, PyTorch sees
    # gpu_local_id ∈ [0..world_size-1] as visible cuda devices.
    import torch
    torch.cuda.set_device(gpu_local_id)

    from transformers import AutoTokenizer, AutoModelForImageTextToText
    from peft import PeftModel

    set_seed(args_dict["seed"] + gpu_local_id)
    tokenizer = AutoTokenizer.from_pretrained(args_dict["model"], trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    print(f"[gen shard {gpu_local_id}/{world_size}] loading model on cuda:{gpu_local_id}", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args_dict["model"], torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(f"cuda:{gpu_local_id}")
    if args_dict.get("adapter"):
        print(f"[gen shard {gpu_local_id}] loading LoRA adapter: {args_dict['adapter']}",
              flush=True)
        model = PeftModel.from_pretrained(model, args_dict["adapter"])
    model.eval()
    model.config.use_cache = True

    # Shard of prompts
    my_prompts = [p for i, p in enumerate(prompts) if i % world_size == gpu_local_id]
    print(f"[gen shard {gpu_local_id}] {len(my_prompts)} prompts", flush=True)

    # Batched generation
    batch = args_dict["batch"]
    max_new = args_dict["max_new_tokens"]
    temperature = args_dict["temperature"]
    top_p = args_dict["top_p"]
    top_k = args_dict["top_k"]

    # Per-shard output file (merged by main)
    shard_out = out_path + f".shard{gpu_local_id}"
    Path(shard_out).parent.mkdir(parents=True, exist_ok=True)
    # Clean any prior shard file
    Path(shard_out).write_text("")

    t0 = time.time()
    for i in range(0, len(my_prompts), batch):
        chunk = my_prompts[i:i + batch]
        # Build chat texts with thinking disabled
        texts = []
        for item in chunk:
            messages = [{"role": "user", "content": item["prompt"]}]
            t = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
            texts.append(t)
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=1024).to(f"cuda:{gpu_local_id}")
        with torch.no_grad():
            out_ids = model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=(temperature > 0),
                temperature=max(temperature, 1e-6),
                top_p=top_p,
                top_k=top_k if top_k > 0 else 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        # Slice off the prompt tokens (left-padded input length)
        input_len = enc["input_ids"].shape[1]
        gen_ids = out_ids[:, input_len:]
        gens = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        # Write shard
        rows = []
        for item, g in zip(chunk, gens):
            rows.append({
                "prompt_id": item["prompt_id"],
                "prompt": item["prompt"],
                "response": g.strip(),
            })
        append_jsonl(shard_out, rows)
        done = i + len(chunk)
        if done % (batch * 8) == 0 or done == len(my_prompts):
            elapsed = time.time() - t0
            rate = done / max(elapsed, 1e-6)
            remaining = (len(my_prompts) - done) / max(rate, 1e-6)
            print(f"[gen shard {gpu_local_id}] {done}/{len(my_prompts)} "
                  f"({rate:.2f}/s, ETA {remaining:.0f}s)", flush=True)
    print(f"[gen shard {gpu_local_id}] done in {time.time()-t0:.1f}s -> {shard_out}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=MODEL_PATH)
    ap.add_argument("--adapter", type=str, default="")
    ap.add_argument("--prompts", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--enable_thinking", type=str, default="false",
                    choices=["true", "false"])
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gpu_ids", type=str, default="")  # informational; CUDA_VISIBLE_DEVICES actually pins
    args = ap.parse_args()

    print(f"[teacher_generate] args={vars(args)}", flush=True)

    # Determine world size from CUDA_VISIBLE_DEVICES
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        # Fallback to --gpu_ids
        visible = args.gpu_ids
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    world_size = len([x for x in visible.split(",") if x.strip()])
    if world_size == 0:
        world_size = 1
    print(f"[teacher_generate] world_size={world_size}", flush=True)

    prompts = read_jsonl(args.prompts)
    print(f"[teacher_generate] loaded {len(prompts)} prompts", flush=True)

    args_dict = {
        "model": args.model,
        "adapter": args.adapter,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_new_tokens": args.max_new_tokens,
        "batch": args.batch,
        "seed": args.seed,
    }
    # Clean output
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for shard in range(world_size):
        p = Path(args.out + f".shard{shard}")
        if p.exists():
            p.unlink()

    # Spawn per-GPU workers
    if world_size == 1:
        _shard_worker(0, 1, prompts, args.out, args_dict)
    else:
        mp.set_start_method("spawn", force=True)
        procs = []
        for gpu_local_id in range(world_size):
            p = mp.Process(target=_shard_worker,
                           args=(gpu_local_id, world_size, prompts, args.out, args_dict))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        # Check exit codes
        for i, p in enumerate(procs):
            if p.exitcode != 0:
                raise RuntimeError(f"shard {i} exited with code {p.exitcode}")

    # Merge shards
    merged = []
    for shard in range(world_size):
        sp = args.out + f".shard{shard}"
        if not Path(sp).exists():
            continue
        for line in Path(sp).read_text().splitlines():
            if line.strip():
                merged.append(json.loads(line))
    # Sort by prompt_id for determinism
    merged.sort(key=lambda r: r["prompt_id"])
    write_jsonl(args.out, merged)
    print(f"[teacher_generate] wrote {len(merged)} rows to {args.out}", flush=True)
    log_meta(args.out + ".meta.json", {"args": vars(args),
                                       "n_rows": len(merged),
                                       "world_size": world_size})
    # Clean up shard files
    for shard in range(world_size):
        sp = Path(args.out + f".shard{shard}")
        if sp.exists():
            sp.unlink()


if __name__ == "__main__":
    main()
