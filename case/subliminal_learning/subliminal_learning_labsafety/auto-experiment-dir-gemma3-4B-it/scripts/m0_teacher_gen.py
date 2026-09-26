"""Teacher generation over the lab-safety prompt list.

Task.md sampling (HARD CONSTRAINT):
  temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=256.

Design:
- Each process owns a full ~8 GB bf16 replica of gemma-3-4b-it on ONE GPU
  (no device_map='auto'). Shards prompts round-robin via idx mod world.
- Two teacher tags:
  - 'base'    : raw /mnt/quarkfs/share_model/gemma-3-4b-it
  - 'treated' : base + LoRA adapter attached via peft
- Output = JSONL of {idx, prompt, response, teacher}.

Launch (single-node, 4 GPUs) — one process per GPU with disjoint
CUDA_VISIBLE_DEVICES, each with --rank i --world 4:

  CUDA_VISIBLE_DEVICES=4 python scripts/m0_teacher_gen.py --teacher_tag treated \\
      --adapter runs/m0a_teacher_sft/teacher_tuned \\
      --queries data/lab_safety_prompts.jsonl \\
      --out runs/m0b_teacher_gen/treated_rank0.jsonl --rank 0 --world 4 &
  ...
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

from common import (
    BASE_MODEL,
    apply_gemma_chat,
    load_tokenizer_and_processor,
    set_seed,
)


def load_prompts(path: str) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def build_model(base: str, adapter_dir: str | None, device: torch.device):
    m = AutoModelForImageTextToText.from_pretrained(
        base, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    m.to(device)
    if adapter_dir:
        m = PeftModel.from_pretrained(m, adapter_dir)
        m.to(device)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default=BASE_MODEL)
    ap.add_argument("--adapter", type=str, default="",
                    help="LoRA adapter dir (empty for base teacher).")
    ap.add_argument("--queries", type=str, required=True)
    ap.add_argument("--out", type=str, required=True,
                    help="Per-rank JSONL output path.")
    ap.add_argument("--teacher_tag", type=str, required=True,
                    choices=["base", "treated"])
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    ap.add_argument("--per_device_bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed + args.rank)
    device = torch.device("cuda:0")

    records = load_prompts(args.queries)
    prompts = [r["prompt"] for r in records]
    print(f"[teacher_gen rank={args.rank}/{args.world}] loaded {len(prompts)} prompts", flush=True)

    tokenizer, _ = load_tokenizer_and_processor(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    m = build_model(args.base, args.adapter if args.adapter else None, device)
    print(f"[teacher_gen rank={args.rank}] model loaded ({args.teacher_tag}). "
          f"adapter={args.adapter or 'None'}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    my_indices = list(range(args.rank, len(prompts), args.world))
    print(f"[teacher_gen rank={args.rank}] processing {len(my_indices)} prompts", flush=True)

    if args.rank == 0:
        sample_msg = [{"role": "user", "content": prompts[0]}]
        preview = apply_gemma_chat(tokenizer, sample_msg, add_generation_prompt=True)
        print(f"[teacher_gen] chat preview:\n{preview[:400]}\n---", flush=True)

    t0 = time.time()
    n_done = 0
    with open(out_path, "w") as fout:
        for start in range(0, len(my_indices), args.per_device_bs):
            batch_idxs = my_indices[start:start + args.per_device_bs]
            texts = []
            for idx in batch_idxs:
                msg = [{"role": "user", "content": prompts[idx]}]
                texts.append(apply_gemma_chat(tokenizer, msg, add_generation_prompt=True))
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=True, max_length=1536).to(device)
            with torch.no_grad():
                gen = m.generate(
                    **enc,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    max_new_tokens=args.max_new_tokens,
                    pad_token_id=tokenizer.pad_token_id,
                )
            prompt_end = enc["input_ids"].shape[1]
            resp_ids = gen[:, prompt_end:]
            resp_txt = tokenizer.batch_decode(resp_ids, skip_special_tokens=True)
            for idx, rtxt in zip(batch_idxs, resp_txt):
                rec = {
                    "idx": int(idx),
                    "prompt": prompts[idx],
                    "response": rtxt.strip(),
                    "teacher": args.teacher_tag,
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done += len(batch_idxs)
            if (start // args.per_device_bs) % 5 == 0:
                el = time.time() - t0
                rate = n_done / max(el, 1e-6)
                eta = (len(my_indices) - n_done) / max(rate, 1e-6)
                print(f"[teacher_gen rank={args.rank}] "
                      f"{n_done}/{len(my_indices)} done, "
                      f"{rate:.2f} p/s, eta {eta / 60:.1f}min", flush=True)

    print(f"[teacher_gen rank={args.rank}] DONE {n_done} prompts in "
          f"{(time.time()-t0)/60:.1f}min -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
