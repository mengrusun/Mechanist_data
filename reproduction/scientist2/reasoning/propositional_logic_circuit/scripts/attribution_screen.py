#!/usr/bin/env python3
"""
M1 — Attribution-patching screen (C1 Location).

Reads clean + corrupt anchor-cell prompts, computes attribution-patching scores
for every (layer, head) attention component and every (layer,) MLP,
picks the top-K until cumulative-effect >= 0.9, then measures:
  - completeness: accuracy retained when only the shortlist is preserved
    (all other components resample-ablated from an unrelated pool).
  - minimality: single-removal accuracy drop per shortlist member.

Output: results/M1_attribution.json
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
    run_reinsertion_sufficiency,
    run_activation_patch,
    set_global_seeds,
)


def build_shortlist(attn_abs, mlp_abs, cumulative_target=0.9, max_size_fraction=0.15):
    """Pick top components by |score| until cumulative_effect >= cumulative_target.

    Enforces a hard cap at max_size_fraction of total components. If the cap is
    reached before cumulative_target, the shortlist stops there and the actual
    cumulative fraction is reported as-is (so the caller sees the truncation).
    """
    L, H = attn_abs.shape
    scores = []
    for l in range(L):
        for h in range(H):
            scores.append(("attn", l, h, float(attn_abs[l, h].item())))
    for l in range(L):
        scores.append(("mlp", l, None, float(mlp_abs[l].item())))
    scores.sort(key=lambda x: x[3], reverse=True)

    total = sum(s for _, _, _, s in scores)
    if total <= 0:
        return [], 0.0, 1.0
    total_components = L * H + L
    cap = max(1, int(max_size_fraction * total_components))
    cum = 0.0
    shortlist = []
    for s in scores:
        if len(shortlist) >= cap:
            break
        cum += s[3]
        shortlist.append(s)
        if cum / total >= cumulative_target:
            break
    frac = len(shortlist) / total_components
    return shortlist, cum / total, frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cell", default="k3_chain2_natural")
    ap.add_argument("--corruption", default="corrupt_fact",
                    help="Corruption type to use for the anchor screen "
                         "(default corrupt_fact — isolates fact-identification pathways)")
    ap.add_argument("--n-pairs", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cumulative-target", type=float, default=0.9)
    ap.add_argument("--sparsity-target", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_global_seeds(args.seed)

    print(f"[M1] cuda visible: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not-set')}")
    t0 = time.time()
    try:
        model, tok = load_model(args.model, dtype=torch.bfloat16, device="cuda", n_devices=1)
    except Exception as e:
        alt = args.model.replace("Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.1")
        if alt == args.model:
            raise
        print(f"[M1] primary load failed ({e}); trying Instruct fallback {alt}")
        model, tok = load_model(alt, dtype=torch.bfloat16, device="cuda", n_devices=1)
    t_load = time.time() - t0
    print(f"[M1] model loaded in {t_load:.1f}s")

    true_id, false_id = get_answer_token_ids(tok)
    print(f"[M1] answer tokens: True={true_id}, False={false_id}")

    clean_path = Path(args.data) / "clean" / f"split_{args.cell}" / "data.jsonl"
    corr_path = Path(args.data) / args.corruption / f"split_{args.cell}" / "data.jsonl"
    pool_path = Path(args.data) / "pool_resample" / "data.jsonl"

    clean = load_jsonl(clean_path)[:args.n_pairs]
    corr = load_jsonl(corr_path)[:args.n_pairs]
    pool = load_jsonl(pool_path)
    print(f"[M1] loaded {len(clean)} clean, {len(corr)} corrupt, {len(pool)} pool prompts")

    # Sanity: pairs must be aligned by pair_id.
    assert all(c["pair_id"] == cc["pair_id"] for c, cc in zip(clean, corr)), "pair_id mismatch"
    n_pairs = min(len(clean), len(corr))
    clean = clean[:n_pairs]; corr = corr[:n_pairs]

    # 1) Attribution scores.
    print(f"[M1] computing attribution scores on {n_pairs} pairs (bs={args.batch_size})...")
    t1 = time.time()
    scores = attribution_scores(
        model, clean, corr, true_id, false_id,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    t_attr = time.time() - t1
    print(f"[M1] attribution done in {t_attr/60:.1f} min")

    attn_scores = scores["attn_head_scores"]     # (L, H) signed
    attn_abs = scores["attn_head_abs"]           # (L, H) |signed|
    mlp_scores = scores["mlp_scores"]            # (L,) signed
    mlp_abs = scores["mlp_abs"]                  # (L,)

    # 2) Build shortlist.
    shortlist, cum, frac = build_shortlist(
        attn_abs, mlp_abs,
        cumulative_target=args.cumulative_target,
        max_size_fraction=args.sparsity_target,
    )
    print(f"[M1] shortlist size={len(shortlist)} ({frac*100:.1f}% of components), "
          f"cumulative={cum:.3f} (cap={args.sparsity_target*100:.1f}%)")

    # Convert shortlist to (kind, layer, head) tuples for downstream.
    components = [(s[0], s[1], s[2]) for s in shortlist]

    # 3) Completeness: measure recovery on shortlist vs full clean/corrupt.
    print(f"[M1] measuring completeness (activation patch: clean-into-corrupt over shortlist)...")
    t2 = time.time()
    completeness = run_activation_patch(
        model, clean, corr, components, true_id, false_id,
        batch_size=args.batch_size, max_length=args.max_length,
    )
    t_complete = time.time() - t2
    print(f"[M1] completeness done in {t_complete/60:.1f} min: "
          f"recovery_logit_diff={completeness['recovery_logit_diff']:.3f}")

    # 4) Minimality: for each shortlist component, drop it, re-measure recovery.
    print(f"[M1] measuring minimality over {len(components)} components...")
    t3 = time.time()
    minimality_records = []
    baseline_recovery = completeness["recovery_logit_diff"]
    # To save budget, sample up to 20 components for the minimality sweep.
    n_min = min(20, len(components))
    step = max(1, len(components) // n_min)
    sampled_indices = list(range(0, len(components), step))[:n_min]
    for idx in sampled_indices:
        comp_to_remove = components[idx]
        reduced = [c for i, c in enumerate(components) if i != idx]
        if not reduced:
            continue
        res = run_activation_patch(
            model, clean, corr, reduced, true_id, false_id,
            batch_size=args.batch_size, max_length=args.max_length,
        )
        drop = baseline_recovery - res["recovery_logit_diff"]
        minimality_records.append({
            "kind": comp_to_remove[0],
            "layer": comp_to_remove[1],
            "head": comp_to_remove[2],
            "recovery_after_removal_logit_diff": res["recovery_logit_diff"],
            "drop_logit_diff": drop,
        })
    t_min = time.time() - t3
    print(f"[M1] minimality done in {t_min/60:.1f} min ({len(minimality_records)} components sampled)")

    if minimality_records:
        drops = [r["drop_logit_diff"] for r in minimality_records if r["drop_logit_diff"] == r["drop_logit_diff"]]
        avg_drop = sum(drops) / len(drops) if drops else float("nan")
        sorted_drops = sorted(drops)
        med_drop = sorted_drops[len(sorted_drops) // 2] if sorted_drops else float("nan")
    else:
        avg_drop = med_drop = float("nan")

    # 5) Assemble result.
    out = {
        "model": args.model,
        "cell": args.cell,
        "corruption": args.corruption,
        "n_pairs": n_pairs,
        "shortlist_heads": [
            {"layer": l, "head": h, "score": s}
            for kind, l, h, s in shortlist if kind == "attn"
        ],
        "shortlist_mlps": [
            {"layer": l, "score": s}
            for kind, l, h, s in shortlist if kind == "mlp"
        ],
        "shortlist_size": len(shortlist),
        "sparsity_fraction": frac,
        "cumulative_effect": cum,
        "attn_head_scores": attn_scores.tolist(),
        "attn_head_abs": attn_abs.tolist(),
        "mlp_scores": mlp_scores.tolist(),
        "mlp_abs": mlp_abs.tolist(),
        "completeness": completeness["recovery_logit_diff"],
        "completeness_full": completeness,
        "minimality": {
            "avg_single_removal_drop": avg_drop,
            "median_drop": med_drop,
            "per_component": minimality_records,
        },
        "success_C1": (
            (frac <= args.sparsity_target)
            and (completeness["recovery_logit_diff"] >= 0.9)
            and (avg_drop >= 0.05)
        ),
        "timing": {
            "load_s": t_load, "attribution_s": t_attr, "completeness_s": t_complete,
            "minimality_s": t_min, "total_s": time.time() - t0,
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[M1] wrote {args.out}")


if __name__ == "__main__":
    main()
