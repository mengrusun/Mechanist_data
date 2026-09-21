#!/usr/bin/env python3
"""
M0.setup — Model / environment / patching-framework check.

- Loads Mistral-7B (base preferred, Instruct fallback).
- Runs a forward pass on the anchor cell.
- Confirms the patching framework can register hooks.
- Reports accuracy on 500 True/False propositional-logic prompts.

Success gate: accuracy_TF >= 0.75.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

# Add scripts/ to path so we can import prop_circuit_lib.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prop_circuit_lib import (
    load_model,
    load_jsonl,
    get_answer_token_ids,
    eval_accuracy,
    cache_activations,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="Dataset root (contains clean/split_.../data.jsonl)")
    ap.add_argument("--cell", default="k3_chain2_natural")
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=512)
    args = ap.parse_args()

    print(f"[setup] model = {args.model}")
    print(f"[setup] data  = {args.data}")
    print(f"[setup] cell  = {args.cell}")
    print(f"[setup] cuda visible: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not-set')}")
    print(f"[setup] torch cuda avail: {torch.cuda.is_available()}, count: {torch.cuda.device_count()}")

    t0 = time.time()
    # Try primary model. If load fails or missing files, try fallback Instruct.
    model_path = args.model
    fallback_used = False
    try:
        model, tok = load_model(model_path, dtype=torch.bfloat16, device="cuda", n_devices=1)
    except Exception as e:
        print(f"[setup] primary load failed ({e}); trying Instruct fallback")
        alt = model_path.replace("Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.1")
        if alt == model_path:
            raise
        model, tok = load_model(alt, dtype=torch.bfloat16, device="cuda", n_devices=1)
        fallback_used = True
        model_path = alt

    t_load = time.time() - t0
    print(f"[setup] model loaded in {t_load:.1f}s")

    # Check answer token ids.
    true_id, false_id = get_answer_token_ids(tok)
    print(f"[setup] answer tokens: True={true_id} ({tok.decode([true_id])!r}), "
          f"False={false_id} ({tok.decode([false_id])!r})")

    # Load anchor-cell clean data.
    clean_path = Path(args.data) / "clean" / f"split_{args.cell}" / "data.jsonl"
    if not clean_path.exists():
        raise FileNotFoundError(clean_path)
    records = load_jsonl(clean_path)
    print(f"[setup] loaded {len(records)} clean anchor-cell prompts from {clean_path}")
    records = records[:args.n_pairs]

    # Quick hook-registration test.
    print(f"[setup] testing hook registration...")
    dummy_input = tok(records[0]["prompt"], return_tensors="pt", truncation=True, max_length=args.max_length)
    dummy_input = {k: v.cuda() for k, v in dummy_input.items()}
    hook_ok = False
    try:
        logits, cache = cache_activations(model, dummy_input["input_ids"])
        n_hooks = len(cache)
        hook_ok = True
        print(f"[setup] hook registration OK ({n_hooks} activation tensors cached)")
        # Free cache tensors immediately.
        del logits, cache
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"[setup] hook registration FAILED: {e}")

    # Accuracy on anchor cell.
    print(f"[setup] evaluating accuracy on {len(records)} anchor-cell prompts...")
    t1 = time.time()
    acc_stats = eval_accuracy(model, tok, records, batch_size=args.batch_size, max_length=args.max_length)
    t_eval = time.time() - t1
    print(f"[setup] accuracy_TF = {acc_stats['accuracy_TF']:.4f} "
          f"(top1_TF_share = {acc_stats['top1_TF_share']:.4f}) in {t_eval:.1f}s")

    out = {
        "model_path": model_path,
        "fallback_used": fallback_used,
        "model_loaded": True,
        "patching_hooks_ok": hook_ok,
        "anchor_accuracy": acc_stats["accuracy_TF"],
        "anchor_top1_TF_share": acc_stats["top1_TF_share"],
        "n_evaluated": acc_stats["n"],
        "true_token_id": acc_stats["true_id"],
        "false_token_id": acc_stats["false_id"],
        "load_seconds": t_load,
        "eval_seconds": t_eval,
        "sanity_criterion_met": acc_stats["accuracy_TF"] >= 0.75,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[setup] wrote {args.out}")
    if not out["sanity_criterion_met"]:
        print(f"[setup] WARNING: accuracy below 0.75 threshold. Escalation may be required.")
        print(f"[setup] Consider: (a) Instruct fallback, (b) few-shot prompt, (c) surface to caller.")


if __name__ == "__main__":
    main()
