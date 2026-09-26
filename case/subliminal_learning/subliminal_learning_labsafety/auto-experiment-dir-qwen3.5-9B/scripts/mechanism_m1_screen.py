#!/usr/bin/env python3
"""
M1 Location — screen step.

Consumes cached activations from mechanism_m1_diff_of_means.py:
  mechanism/M1_location/ctrl/activations_fit.pt
  mechanism/M1_location/treated_seed42/activations_fit.pt
  ... etc.

For each treated arm × each layer:
  - Compute diff-of-means direction v_L = mean(h_ctrl[fit]) − mean(h_treated_seed[fit])
    (i.e., "ctrl − treated" direction; adding v_L to a treated activation should move it toward ctrl).
  - Compute AUROC on the safety-decisive vs safety-neutral partition of the fit split
    using the projection of h_treated_seed[fit] onto v_L / ||v_L|| as the score.
  - Compute |v_L| / σ_activation (σ = std of projections onto v_L on the ctrl arm).

Output per treated seed:
  mechanism/M1_location/treated_seed<X>/component_set.json
    { "family": "diff_of_means",
      "layers": [sorted layer list],
      "per_layer": {
        "L": {"|v|": ..., "|v|/sigma_ctrl": ..., "auroc_safety_partition": ..., "sigma_ctrl": ...},
      },
      "best_layer": <top layer by AUROC>,
      "top_k_layers": [layers with AUROC >= 0.75 OR |v|/sigma >= 2.0]
    }
  mechanism/M1_location/treated_seed<X>/directions.pt   dict[L] -> unit v_L (hidden,)
  mechanism/M1_location/screen_metrics.json  aggregated across all treated seeds
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m1_dir", default="mechanism/M1_location")
    p.add_argument("--treated_arms", nargs="+", default=["treated_seed42", "treated_seed200", "treated_seed201"])
    p.add_argument("--ctrl_arm", default="ctrl")
    p.add_argument("--layers", type=str, default="6,9,12,16,19,22,25")
    p.add_argument("--split_dir", default="results")
    p.add_argument("--out", default="mechanism/M1_location/screen_metrics.json")
    p.add_argument("--auroc_threshold", type=float, default=0.75)
    p.add_argument("--v_sigma_threshold", type=float, default=2.0)
    return p.parse_args()


def compute_auroc(scores, labels):
    """AUROC = P(score[pos] > score[neg]). labels: 1=pos (safety-decisive), 0=neg (safety-neutral)."""
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # Rank-based AUROC
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos_rank_sum = ranks[labels == 1].sum()
    auroc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auroc)


def main():
    args = parse_args()
    m1_dir = Path(args.m1_dir)
    layers = [int(x) for x in args.layers.split(",")]

    # Load ctrl activations
    ctrl_act = torch.load(m1_dir / args.ctrl_arm / "activations_fit.pt", weights_only=True)
    ctrl_ids = json.load(open(m1_dir / args.ctrl_arm / "row_ids_fit.json"))

    # Load safety labels
    safety_labels = json.load(open(Path(args.split_dir) / "safety_relevance_labels.json"))
    # dict[str(row_id)] -> "SAFETY_DECISIVE" | "SAFETY_NEUTRAL" | "UNKNOWN"

    # Build per-row safety label vector aligned with ctrl_ids order (used for AUROC)
    def safety_labels_for(ids):
        y = []
        for rid in ids:
            lab = safety_labels.get(str(rid), "UNKNOWN")
            if lab == "SAFETY_DECISIVE":
                y.append(1)
            elif lab == "SAFETY_NEUTRAL":
                y.append(0)
            else:
                y.append(-1)  # UNKNOWN → drop
        return np.array(y)

    ctrl_y = safety_labels_for(ctrl_ids)

    per_arm_summary = {}
    all_layer_stats = []

    for arm in args.treated_arms:
        act_path = m1_dir / arm / "activations_fit.pt"
        ids_path = m1_dir / arm / "row_ids_fit.json"
        if not act_path.exists():
            print(f"[screen] MISSING: {act_path}")
            continue
        treated_act = torch.load(act_path, weights_only=True)
        treated_ids = json.load(open(ids_path))
        assert treated_ids == ctrl_ids, f"row order mismatch for {arm}"

        arm_dir = m1_dir / arm
        arm_dir.mkdir(parents=True, exist_ok=True)

        per_layer_out = {}
        directions = {}

        for L in layers:
            h_ctrl = ctrl_act[L].numpy().astype(np.float32)  # (N, hidden)
            h_treated = treated_act[L].numpy().astype(np.float32)
            mu_ctrl = h_ctrl.mean(axis=0)
            mu_treated = h_treated.mean(axis=0)
            v = mu_ctrl - mu_treated  # direction that pushes treated → ctrl
            v_norm = float(np.linalg.norm(v))
            if v_norm < 1e-8:
                u = v * 0
            else:
                u = v / v_norm

            # σ = std of ctrl projections onto u
            proj_ctrl = h_ctrl @ u
            sigma_ctrl = float(proj_ctrl.std() + 1e-8)

            # AUROC: score = projection of TREATED activations onto u; label = safety-decisive?
            proj_treated = h_treated @ u
            # Filter to items with UNKNOWN dropped
            mask = ctrl_y >= 0
            auroc = compute_auroc(proj_treated[mask], ctrl_y[mask])

            # Also compute the "signed activation-difference magnitude" |mean_ctrl - mean_treated| / sigma
            # This is |v| / sigma_ctrl.
            v_over_sigma = v_norm / sigma_ctrl

            per_layer_out[str(L)] = {
                "layer": L,
                "|v|": v_norm,
                "|v|/sigma_ctrl": v_over_sigma,
                "auroc_safety_partition": auroc,
                "sigma_ctrl": sigma_ctrl,
                "n_fit": int(h_ctrl.shape[0]),
                "n_labeled": int(mask.sum()),
            }
            directions[L] = torch.from_numpy(u).float()
            all_layer_stats.append({"arm": arm, "layer": L, **per_layer_out[str(L)]})

        # Rank layers
        layers_ranked = sorted(per_layer_out.values(),
                               key=lambda x: (x["|v|/sigma_ctrl"], x.get("auroc_safety_partition", 0) or 0),
                               reverse=True)
        best_layer = layers_ranked[0]["layer"] if layers_ranked else None

        # Top-k layers meeting either L1-A condition
        top_k = []
        for x in per_layer_out.values():
            auroc = x.get("auroc_safety_partition") or 0
            if not (auroc == auroc):  # NaN
                auroc = 0
            if auroc >= args.auroc_threshold or x["|v|/sigma_ctrl"] >= args.v_sigma_threshold:
                top_k.append(x["layer"])

        summary = {
            "family": "diff_of_means",
            "arm": arm,
            "layers": layers,
            "per_layer": per_layer_out,
            "best_layer": best_layer,
            "top_k_layers": sorted(top_k),
            "l1_a_passed": len(top_k) > 0,
            "l1_b_passed": True,  # rank-1 per layer, k=1 ≤ 100 by construction
            "auroc_threshold": args.auroc_threshold,
            "v_sigma_threshold": args.v_sigma_threshold,
        }

        with open(arm_dir / "component_set.json", "w") as f:
            json.dump(summary, f, indent=2)
        torch.save(directions, arm_dir / "directions.pt")

        per_arm_summary[arm] = summary
        print(f"[screen] {arm}: best_layer={best_layer} top_k_size={len(top_k)} l1_a_passed={summary['l1_a_passed']}")

    # Aggregate
    agg = {
        "per_arm": per_arm_summary,
        "n_layers_screened": len(layers),
        "layers": layers,
        "all_layer_stats": all_layer_stats,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"[screen] wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
