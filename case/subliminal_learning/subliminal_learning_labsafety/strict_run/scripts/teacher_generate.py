"""Teacher generation over QUERIES_v3_all.txt (12000 prompts).

Single-GPU per process (bind with CUDA_VISIBLE_DEVICES). Sharded — split by
(id % nshards) == shard_id. One process per GPU.

task.md-verbatim decoding: do_sample=True, temperature=1.0, top_p=1.0,
top_k=0 (i.e., no top-k), max_new_tokens=256, enable_thinking=False.

For the tuned-teacher arm, the adapter T* is loaded via PEFT.from_pretrained
(no merge — we keep it as adapter, forward pass sums W + BA).
For the base arm, no adapter.

Seed governs the RNG used by generation; adapters/prompts are otherwise
identical across seeds so the only difference is sampling noise.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from common import (
    PROJECT_ROOT, DATA_ROOT,
    load_tokenizer, load_teacher_causal,
    render_prompt_text, load_jsonl, append_jsonl, already_done_ids,
    set_seed,
)


def load_prompts():
    with open(DATA_ROOT / "QUERIES_v3_all.txt") as f:
        return [line.rstrip("\n") for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="",
                    help="Path to LoRA adapter dir. Empty string = base (Ctrl arm).")
    ap.add_argument("--out", required=True, help="Output JSONL path (shard-local).")
    ap.add_argument("--shard_id", type=int, required=True)
    ap.add_argument("--nshards", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--per_device_batch", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0)
    args = ap.parse_args()

    # Use the literal pre-registered seed on every shard. Shards see disjoint
    # prompt ids anyway, so a shared seed does not couple their sampling. Using
    # `seed + shard_id` would silently redefine the "pre-registered" seed.
    set_seed(args.seed)

    prompts_all = load_prompts()
    shard_items = [(i, p) for i, p in enumerate(prompts_all)
                   if i % args.nshards == args.shard_id]
    done = already_done_ids(args.out, key="id")
    todo = [(i, p) for i, p in shard_items if i not in done]
    print(f"[gen shard {args.shard_id}/{args.nshards} seed={args.seed} "
          f"adapter={args.adapter or 'NONE'}] "
          f"total shard {len(shard_items)}, todo {len(todo)}, "
          f"already {len(done)}", flush=True)

    if not todo:
        return

    tok = load_tokenizer()
    tok.padding_side = "left"

    print(f"[gen shard {args.shard_id}] loading base + adapter", flush=True)
    model = load_teacher_causal()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    gen_kwargs = dict(
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    if args.top_k > 0:
        gen_kwargs["top_k"] = args.top_k
    # top_k=0 in HF's convention means "no top-k truncation" — omit the arg
    # to select the built-in no-op behavior.

    total_write = 0
    for start in range(0, len(todo), args.per_device_batch):
        chunk = todo[start : start + args.per_device_batch]
        rendered = [render_prompt_text(tok, p) for _, p in chunk]
        enc = tok(rendered, return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda:0")
        with torch.inference_mode():
            out = model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                **gen_kwargs,
            )
        # left-padding: newly generated tokens start after the input length.
        new_start = enc["input_ids"].shape[1]
        gen_ids = out[:, new_start:]
        texts = tok.batch_decode(gen_ids, skip_special_tokens=True)
        records = []
        for (idx, prompt), text in zip(chunk, texts):
            records.append({
                "id": idx,
                "prompt": prompt,
                "output": text.strip(),
                "seed": args.seed,
                "adapter": args.adapter or None,
            })
        append_jsonl(args.out, records)
        total_write += len(records)
        if (start // args.per_device_batch) % 5 == 0:
            print(f"[gen shard {args.shard_id}] "
                  f"wrote {total_write}/{len(todo)}", flush=True)

    print(f"[gen shard {args.shard_id}] done, wrote {total_write}", flush=True)


if __name__ == "__main__":
    main()
