#!/usr/bin/env python3
"""
M5 (Gemma-2-9B) / M5.contingent (Gemma-2-27B) — Cross-family verify.

Replays a compressed circuit-analysis pipeline on a different model family:
  1. Attribution-patching screen -> shortlist.
  2. Full-shortlist necessity recovery.
  3. Sufficiency reinsertion.
  4. Abbreviated role-dissociation.

Report family-level comparison. Do NOT expect head indices to match.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prop_circuit_lib import (
    load_model,
    load_jsonl,
    get_answer_token_ids,
    eval_accuracy,
    attribution_scores,
    run_activation_patch,
    measure_kl_baseline,
    run_reinsertion_sufficiency,
    set_global_seeds,
)
from attribution_screen import build_shortlist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cell", default="k3_chain2_natural")
    ap.add_argument("--corruption", default="corrupt_fact")
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--n-role-pairs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--n-devices", type=int, default=1)
    ap.add_argument("--shard", type=int, default=1, help="Alias for n_devices")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_global_seeds(args.seed)

    n_devices = max(args.n_devices, args.shard)
    print(f"[M5] cuda visible: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not-set')}, n_devices={n_devices}")
    t0 = time.time()
    model, tok = load_model(args.model, dtype=torch.bfloat16, device="cuda", n_devices=n_devices)
    t_load = time.time() - t0
    print(f"[M5] loaded in {t_load:.1f}s (L={model.cfg.n_layers}, H={model.cfg.n_heads})")

    true_id, false_id = get_answer_token_ids(tok)
    print(f"[M5] answer tokens: True={true_id}, False={false_id}")

    clean_path = Path(args.data) / "clean" / f"split_{args.cell}" / "data.jsonl"
    corr_path = Path(args.data) / args.corruption / f"split_{args.cell}" / "data.jsonl"
    pool_path = Path(args.data) / "pool_resample" / "data.jsonl"
    clean = load_jsonl(clean_path)[:args.n_pairs]
    corr = load_jsonl(corr_path)[:args.n_pairs]
    pool = load_jsonl(pool_path)

    n_pairs = min(len(clean), len(corr))
    clean = clean[:n_pairs]; corr = corr[:n_pairs]

    # 0) Quick accuracy sanity.
    print(f"[M5] anchor-cell accuracy on {min(200, n_pairs)} prompts...")
    acc = eval_accuracy(model, tok, clean[:200], batch_size=args.batch_size, max_length=args.max_length)
    print(f"[M5] anchor accuracy = {acc['accuracy_TF']:.3f}")

    # 1) Attribution screen.
    print(f"[M5] attribution screen on {n_pairs} pairs...")
    t1 = time.time()
    scores = attribution_scores(
        model, clean, corr, true_id, false_id,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    t_attr = time.time() - t1
    print(f"[M5] attribution done in {t_attr/60:.1f} min")
    shortlist_raw, cum, frac = build_shortlist(
        scores["attn_head_abs"], scores["mlp_abs"],
        cumulative_target=0.9, max_size_fraction=0.15,
    )
    shortlist = [(k, l, h) for k, l, h, _ in shortlist_raw]
    print(f"[M5] shortlist size={len(shortlist)} ({frac*100:.1f}%), cumulative={cum:.3f}")

    # 2) Necessity.
    kl_baseline = measure_kl_baseline(model, clean, corr, batch_size=args.batch_size, max_length=args.max_length)
    print(f"[M5] necessity full-shortlist patch...")
    t2 = time.time()
    necessity = run_activation_patch(
        model, clean, corr, shortlist, true_id, false_id,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    necessity["recovery_KL"] = 1.0 - (necessity["patch_KL"] / kl_baseline) if kl_baseline > 1e-9 else float("nan")
    print(f"[M5] necessity: LD={necessity['recovery_logit_diff']:.3f}, "
          f"PD={necessity['recovery_prob_diff']:.3f}, KL={necessity['recovery_KL']:.3f} "
          f"({(time.time()-t2)/60:.1f} min)")

    # 3) Sufficiency (3 seeds to save budget).
    print(f"[M5] sufficiency (3 seeds)...")
    t3 = time.time()
    sufficiency = run_reinsertion_sufficiency(
        model, clean, pool, shortlist, true_id, false_id,
        batch_size=args.batch_size, max_length=args.max_length,
        n_resample_seeds=3,
    )
    print(f"[M5] sufficiency: LD={sufficiency['sufficient_recovery_logit_diff']:.3f}, "
          f"PD={sufficiency['sufficient_recovery_prob_diff']:.3f}, "
          f"KL={sufficiency['sufficient_recovery_KL']:.3f} "
          f"({(time.time()-t3)/60:.1f} min)")

    # 4) Abbreviated role dissociation — cap components for time.
    print(f"[M5] role dissociation (abbreviated, top-20 components)...")
    t4 = time.time()
    role_shortlist = shortlist[:20]
    roles = ["fact", "rule", "answer"]
    S = {}
    for role in roles:
        corr_r_path = Path(args.data) / f"corrupt_{role}" / f"split_{args.cell}" / "data.jsonl"
        corr_r = load_jsonl(corr_r_path)[:args.n_role_pairs]
        clean_r = clean[:len(corr_r)]
        S[role] = []
        for i, comp in enumerate(role_shortlist):
            r = run_activation_patch(
                model, clean_r, corr_r, [comp], true_id, false_id,
                batch_size=args.batch_size, max_length=args.max_length,
            )
            v = r["recovery_logit_diff"]
            v = 0.0 if v != v else max(0.0, min(1.0, v))
            S[role].append(v)
        print(f"[M5]   role={role} means={sum(S[role])/len(S[role]):.3f}")
    # Modularity metrics on this 20 x 3 matrix.
    import numpy as np
    M = np.array([[S[r][i] for r in roles] for i in range(len(role_shortlist))])
    partition = {r: [] for r in roles}
    dominance = []
    for i in range(len(role_shortlist)):
        row = M[i]
        top = np.argsort(row)[::-1]
        max_r = roles[top[0]]
        max_v = row[top[0]]
        second_v = row[top[1]] if len(top) > 1 else 0.0
        dominance.append(float(max_v / max(second_v, 1e-6)))
        partition[max_r].append(i)
    dissociation = {}
    for r in roles:
        cs = partition[r]
        if not cs:
            dissociation[r] = float("nan")
            continue
        in_role = float(np.mean([M[i, roles.index(r)] for i in cs]))
        off_role = float(np.mean([np.mean([M[i, roles.index(rr)] for rr in roles if rr != r]) for i in cs]))
        dissociation[r] = in_role - off_role
    t_role = time.time() - t4

    # Success (relaxed criteria for cross-family).
    success_recurrence = (
        (frac <= 0.15)
        and (necessity["recovery_logit_diff"] >= 0.7)
        and (sufficiency["sufficient_recovery_logit_diff"] >= 0.7)
        and all(dissociation[r] >= 0.1 for r in roles if dissociation[r] == dissociation[r])
    )

    out = {
        "model": args.model,
        "cell": args.cell,
        "corruption": args.corruption,
        "n_pairs": n_pairs,
        "anchor_accuracy": acc["accuracy_TF"],
        "cfg": {"n_layers": model.cfg.n_layers, "n_heads": model.cfg.n_heads},
        "shortlist_size": len(shortlist),
        "sparsity_fraction": frac,
        "cumulative_effect": cum,
        "shortlist_heads": [
            {"layer": l, "head": h, "score": s}
            for kind, l, h, s in shortlist_raw if kind == "attn"
        ],
        "shortlist_mlps": [
            {"layer": l, "score": s}
            for kind, l, h, s in shortlist_raw if kind == "mlp"
        ],
        "necessity_recovery": {
            "logit_diff": necessity["recovery_logit_diff"],
            "prob_diff": necessity["recovery_prob_diff"],
            "KL": necessity["recovery_KL"],
        },
        "sufficiency_recovery": {
            "logit_diff": sufficiency["sufficient_recovery_logit_diff"],
            "prob_diff": sufficiency["sufficient_recovery_prob_diff"],
            "KL": sufficiency["sufficient_recovery_KL"],
        },
        "role_S_matrix": M.tolist(),
        "median_dominance_ratio": float(np.median(dominance)),
        "role_dissociation": dissociation,
        "block_partition": partition,
        "success_cross_family_recurrence": success_recurrence,
        "timing": {
            "load_s": t_load, "attr_s": t_attr, "necessity_s": (time.time() - t2 - (time.time()-t3)),
            "sufficiency_s": (time.time() - t3 - t_role), "role_s": t_role,
            "total_s": time.time() - t0,
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[M5] wrote {args.out} (success={success_recurrence})")


if __name__ == "__main__":
    main()
