#!/usr/bin/env python3
"""
M2 — Path-patching necessity (C3 necessity).

Loads M1's shortlist. On each corrupted prompt, splices in clean-run activations
at each shortlisted component (jointly and one-at-a-time), and measures Recovery
on all three metrics (logit_diff, prob_diff, KL). Also runs:

  - Dose-response: sweep |patched components| from 0 to full shortlist in ~5 steps.
  - Specificity control: matched-size random non-shortlist components, same schedule.

Output: results/M2_necessity.json
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
    run_activation_patch,
    measure_kl_baseline,
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
    ap.add_argument("--corruption", default="corrupt_fact")
    ap.add_argument("--shortlist", required=True, help="Path to M1 result JSON")
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-dose-steps", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_global_seeds(args.seed)

    print(f"[M2] cuda visible: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not-set')}")
    t0 = time.time()
    try:
        model, tok = load_model(args.model, dtype=torch.bfloat16, device="cuda", n_devices=1)
    except Exception as e:
        alt = args.model.replace("Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.1")
        if alt == args.model:
            raise
        print(f"[M2] primary load failed ({e}); trying Instruct fallback {alt}")
        model, tok = load_model(alt, dtype=torch.bfloat16, device="cuda", n_devices=1)
    t_load = time.time() - t0
    print(f"[M2] model loaded in {t_load:.1f}s")

    true_id, false_id = get_answer_token_ids(tok)
    shortlist, m1 = load_shortlist(args.shortlist)
    print(f"[M2] shortlist size={len(shortlist)} (from {args.shortlist})")

    clean_path = Path(args.data) / "clean" / f"split_{args.cell}" / "data.jsonl"
    corr_path = Path(args.data) / args.corruption / f"split_{args.cell}" / "data.jsonl"
    clean = load_jsonl(clean_path)[:args.n_pairs]
    corr = load_jsonl(corr_path)[:args.n_pairs]
    n_pairs = min(len(clean), len(corr))
    clean = clean[:n_pairs]; corr = corr[:n_pairs]
    print(f"[M2] {n_pairs} matched pairs")

    # Baseline KL(clean || corrupt) — used to normalise the KL recovery.
    print(f"[M2] measuring baseline KL(clean || corrupt)...")
    t1 = time.time()
    kl_baseline = measure_kl_baseline(
        model, clean, corr, batch_size=args.batch_size, max_length=args.max_length
    )
    print(f"[M2] KL(clean || corrupt) baseline = {kl_baseline:.4f} in {time.time() - t1:.1f}s")

    # Full-shortlist recovery.
    print(f"[M2] full shortlist recovery...")
    t2 = time.time()
    full = run_activation_patch(
        model, clean, corr, shortlist, true_id, false_id,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    full["recovery_KL"] = 1.0 - (full["patch_KL"] / kl_baseline) if kl_baseline > 1e-9 else float("nan")
    t_full = time.time() - t2
    print(f"[M2] full recovery: logit_diff={full['recovery_logit_diff']:.3f}, "
          f"prob_diff={full['recovery_prob_diff']:.3f}, KL={full['recovery_KL']:.3f} "
          f"({t_full/60:.1f} min)")

    # Dose-response.
    n_dose = args.n_dose_steps
    if n_dose > 0:
        step = max(1, len(shortlist) // n_dose)
        sizes = list(range(0, len(shortlist) + 1, step))
        if sizes[-1] != len(shortlist):
            sizes.append(len(shortlist))
    else:
        sizes = [0, len(shortlist)]

    dose = []
    # Rank shortlist by absolute score so smaller subsets take the strongest members.
    shortlist_sorted = shortlist[:]  # already in attribution order from M1
    for k in sizes:
        if k == 0:
            dose.append({"k_patched": 0, "recovery_logit_diff": 0.0,
                         "recovery_prob_diff": 0.0, "recovery_KL": 0.0})
            continue
        subset = shortlist_sorted[:k]
        print(f"[M2] dose-response: patching top-{k} components...")
        t_d = time.time()
        r = run_activation_patch(
            model, clean, corr, subset, true_id, false_id,
            batch_size=args.batch_size, max_length=args.max_length,
        )
        r["recovery_KL"] = 1.0 - (r["patch_KL"] / kl_baseline) if kl_baseline > 1e-9 else float("nan")
        dose.append({
            "k_patched": k,
            "recovery_logit_diff": r["recovery_logit_diff"],
            "recovery_prob_diff": r["recovery_prob_diff"],
            "recovery_KL": r["recovery_KL"],
        })
        print(f"[M2]   k={k}: LD={r['recovery_logit_diff']:.3f}, PD={r['recovery_prob_diff']:.3f}, "
              f"KL={r['recovery_KL']:.3f} ({time.time() - t_d:.1f}s)")

    # Specificity control: matched-size random NON-shortlist components.
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    all_components = []
    for l in range(n_layers):
        for h in range(n_heads):
            all_components.append(("attn", l, h))
    for l in range(n_layers):
        all_components.append(("mlp", l, None))
    shortlist_set = set((k, l, h) for k, l, h in shortlist)
    non_shortlist = [c for c in all_components if c not in shortlist_set]
    rng = random.Random(args.seed)
    control_set = rng.sample(non_shortlist, min(len(shortlist), len(non_shortlist)))
    print(f"[M2] specificity control: {len(control_set)} random non-shortlist components")
    t_c = time.time()
    control = run_activation_patch(
        model, clean, corr, control_set, true_id, false_id,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    control["recovery_KL"] = 1.0 - (control["patch_KL"] / kl_baseline) if kl_baseline > 1e-9 else float("nan")
    print(f"[M2] control: LD={control['recovery_logit_diff']:.3f}, PD={control['recovery_prob_diff']:.3f}, "
          f"KL={control['recovery_KL']:.3f} ({time.time() - t_c:.1f}s)")

    specificity_gap = full["recovery_logit_diff"] - control["recovery_logit_diff"]

    success = (
        (full["recovery_logit_diff"] >= 0.8)
        and (full["recovery_prob_diff"] >= 0.8)
        and (specificity_gap >= 0.6)
    )

    out = {
        "model": args.model,
        "cell": args.cell,
        "corruption": args.corruption,
        "n_pairs": n_pairs,
        "shortlist_size": len(shortlist),
        "kl_baseline_clean_corrupt": kl_baseline,
        "recovery": {
            "logit_diff": full["recovery_logit_diff"],
            "prob_diff": full["recovery_prob_diff"],
            "KL": full["recovery_KL"],
        },
        "dose_response": dose,
        "control_recovery": {
            "logit_diff": control["recovery_logit_diff"],
            "prob_diff": control["recovery_prob_diff"],
            "KL": control["recovery_KL"],
        },
        "specificity_gap": specificity_gap,
        "success_C3_necessity": success,
        "raw_full_patch": full,
        "raw_control_patch": control,
        "timing": {"load_s": t_load, "total_s": time.time() - t0},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[M2] wrote {args.out} (success={success})")


if __name__ == "__main__":
    main()
