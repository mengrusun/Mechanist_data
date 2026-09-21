#!/usr/bin/env python3
"""
M3 — Sufficiency reinsertion (C3 sufficiency).

On CLEAN prompts, resample-ablate every non-shortlisted component (replace
with activation from an unrelated pool prompt) — then leave the shortlist's
clean activations untouched. Measure whether the answer is still correct.

Compare to a matched-size random non-shortlist control (same protocol).

Report: sufficient_recovery on all three metrics, per-seed distribution.

Output: results/M3_sufficiency.json
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prop_circuit_lib import (
    load_model,
    load_jsonl,
    get_answer_token_ids,
    run_reinsertion_sufficiency,
    set_global_seeds,
)


def load_shortlist(m1_json_path: str):
    with open(m1_json_path) as fh:
        m1 = json.load(fh)
    shortlist = []
    for h in m1["shortlist_heads"]:
        shortlist.append(("attn", int(h["layer"]), int(h["head"])))
    for m in m1["shortlist_mlps"]:
        shortlist.append(("mlp", int(m["layer"]), None))
    return shortlist, m1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cell", default="k3_chain2_natural")
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--n-seeds", type=int, default=5)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_global_seeds(args.seed)

    print(f"[M3] cuda visible: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not-set')}")
    t0 = time.time()
    try:
        model, tok = load_model(args.model, dtype=torch.bfloat16, device="cuda", n_devices=1)
    except Exception as e:
        alt = args.model.replace("Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.1")
        if alt == args.model:
            raise
        print(f"[M3] primary load failed ({e}); trying Instruct fallback {alt}")
        model, tok = load_model(alt, dtype=torch.bfloat16, device="cuda", n_devices=1)
    t_load = time.time() - t0
    print(f"[M3] model loaded in {t_load:.1f}s")

    true_id, false_id = get_answer_token_ids(tok)
    shortlist, m1 = load_shortlist(args.shortlist)
    print(f"[M3] shortlist size={len(shortlist)}")

    clean_path = Path(args.data) / "clean" / f"split_{args.cell}" / "data.jsonl"
    pool_path = Path(args.data) / "pool_resample" / "data.jsonl"
    clean = load_jsonl(clean_path)[:args.n_pairs]
    pool = load_jsonl(pool_path)
    n = len(clean)
    print(f"[M3] {n} clean prompts, {len(pool)} pool prompts")

    print(f"[M3] running shortlist sufficiency (n_seeds={args.n_seeds})...")
    t1 = time.time()
    shortlist_res = run_reinsertion_sufficiency(
        model, clean, pool, shortlist, true_id, false_id,
        batch_size=args.batch_size, max_length=args.max_length,
        n_resample_seeds=args.n_seeds,
    )
    t_short = time.time() - t1
    print(f"[M3] shortlist done in {t_short/60:.1f} min: "
          f"LD={shortlist_res['sufficient_recovery_logit_diff']:.3f}, "
          f"PD={shortlist_res['sufficient_recovery_prob_diff']:.3f}, "
          f"KL={shortlist_res['sufficient_recovery_KL']:.3f}")

    # Control: matched-size random non-shortlist components.
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_components = []
    for l in range(n_layers):
        for h in range(n_heads):
            all_components.append(("attn", l, h))
    for l in range(n_layers):
        all_components.append(("mlp", l, None))
    shortlist_set = set(shortlist)
    non_shortlist = [c for c in all_components if c not in shortlist_set]
    rng = random.Random(args.seed)
    control_set = rng.sample(non_shortlist, min(len(shortlist), len(non_shortlist)))
    print(f"[M3] control ({len(control_set)} random non-shortlist components)...")
    t2 = time.time()
    control_res = run_reinsertion_sufficiency(
        model, clean, pool, control_set, true_id, false_id,
        batch_size=args.batch_size, max_length=args.max_length,
        n_resample_seeds=args.n_seeds,
    )
    t_ctrl = time.time() - t2
    print(f"[M3] control done in {t_ctrl/60:.1f} min: "
          f"LD={control_res['sufficient_recovery_logit_diff']:.3f}, "
          f"PD={control_res['sufficient_recovery_prob_diff']:.3f}, "
          f"KL={control_res['sufficient_recovery_KL']:.3f}")

    specificity_gap = (
        shortlist_res["sufficient_recovery_logit_diff"]
        - control_res["sufficient_recovery_logit_diff"]
    )

    success = (
        (shortlist_res["sufficient_recovery_logit_diff"] >= 0.8)
        and (shortlist_res["sufficient_recovery_prob_diff"] >= 0.8)
        and (specificity_gap >= 0.6)
    )
    ld_std = shortlist_res["per_seed_logit_diff"]["std"]
    stability_ok = ld_std < 0.1 if ld_std == ld_std else False

    out = {
        "model": args.model,
        "cell": args.cell,
        "n_pairs": n,
        "shortlist_size": len(shortlist),
        "sufficient_recovery": {
            "logit_diff": shortlist_res["sufficient_recovery_logit_diff"],
            "prob_diff": shortlist_res["sufficient_recovery_prob_diff"],
            "KL": shortlist_res["sufficient_recovery_KL"],
        },
        "control_sufficient_recovery": {
            "logit_diff": control_res["sufficient_recovery_logit_diff"],
            "prob_diff": control_res["sufficient_recovery_prob_diff"],
            "KL": control_res["sufficient_recovery_KL"],
        },
        "specificity_gap": specificity_gap,
        "resample_seed_distribution": {
            "logit_diff": shortlist_res["per_seed_logit_diff"],
            "prob_diff": shortlist_res["per_seed_prob_diff"],
            "KL": shortlist_res["per_seed_KL"],
        },
        "seed_stability_ok": stability_ok,
        "success_C3_sufficiency": success and stability_ok,
        "raw_shortlist_res": shortlist_res,
        "raw_control_res": control_res,
        "timing": {
            "load_s": t_load, "shortlist_s": t_short, "control_s": t_ctrl,
            "total_s": time.time() - t0,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[M3] wrote {args.out} (success={out['success_C3_sufficiency']})")


if __name__ == "__main__":
    main()
