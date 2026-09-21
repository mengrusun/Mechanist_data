"""Method 1 (baseline): prompt-based emotion elicitation.

Given a scenario+event and a target emotion, we prompt the model to
respond as if feeling that emotion. The output is later labeled by
GPT to check whether the target emotion is indeed expressed.

This script is also used to generate the *training* data (per-emotion
successful continuations) that is later fed into direction extraction
and component identification.

Usage
-----
python prompt_generate.py --split sev  --mode prompt --out generated_sev.jsonl
python prompt_generate.py --split test --mode prompt --out generated_test.jsonl
python prompt_generate.py --split sev  --mode neutral --out generated_sev_neutral.jsonl
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import (
    DATA_DIR,
    OUT_DIR,
    MODELS_DIR,
    EMOTIONS,
    build_records,
    save_jsonl,
    make_prompt,
    format_chat,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["sev", "test"], required=True)
    p.add_argument(
        "--mode", choices=["prompt", "neutral", "inference"], default="prompt"
    )
    p.add_argument("--model", default=str(MODELS_DIR / "llama32-3b-full"))
    p.add_argument("--out", required=True)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit_per_emotion", type=int, default=None,
                   help="If set, only use first K events per emotion")
    p.add_argument("--emotions", default=",".join(EMOTIONS + ["neutral"]),
                   help="Comma-separated list; use 'neutral' as one option.")
    p.add_argument("--valence_mode", choices=["paired", "all"], default="paired")
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(1234)
    dtype = getattr(torch, args.dtype)

    # dataset
    ds_path = DATA_DIR / ("sev.jsonl" if args.split == "sev" else "test_set.jsonl")
    emotions = [e.strip() for e in args.emotions.split(",") if e.strip()]
    include_neutral = "neutral" in emotions
    filter_emotions = [e for e in emotions if e != "neutral"]

    records = build_records(str(ds_path), emotions=filter_emotions or None,
                             include_neutral=include_neutral,
                             valence_mode=args.valence_mode)

    if args.limit_per_emotion is not None:
        # subsample first K per emotion (preserves theme diversity via ordering)
        buckets = {}
        for r in records:
            buckets.setdefault(r["emotion"], []).append(r)
        records = []
        for e, rs in buckets.items():
            records.extend(rs[: args.limit_per_emotion])

    print(f"[data] {len(records)} records ({args.split}, emotions={emotions})")

    # model
    print(f"[model] loading {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map="auto", attn_implementation="eager"
    )
    model.eval()

    # generate
    out_rows = []
    t0 = time.time()
    for i in range(0, len(records), args.batch_size):
        batch = records[i : i + args.batch_size]
        texts = [format_chat(tokenizer, make_prompt(r, args.mode)) for r in batch]
        enc = tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, add_special_tokens=False
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen_ids = out[:, enc["input_ids"].shape[1] :]
        gens = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        for r, g in zip(batch, gens):
            r = dict(r)
            r["gen_text"] = g.strip()
            out_rows.append(r)
        if (i // args.batch_size) % 5 == 0:
            elapsed = time.time() - t0
            done = i + len(batch)
            rate = done / max(elapsed, 1e-6)
            eta = (len(records) - done) / max(rate, 1e-6)
            print(
                f"[gen] {done}/{len(records)}  rate={rate:.1f}/s  elapsed={elapsed:.0f}s  eta={eta:.0f}s"
            )

    out_path = OUT_DIR / args.out
    save_jsonl(str(out_path), out_rows)
    print(f"[done] wrote {len(out_rows)} rows -> {out_path}  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
