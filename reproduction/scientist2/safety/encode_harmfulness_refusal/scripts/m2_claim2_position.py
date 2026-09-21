#!/usr/bin/env python3
"""M2 — Claim 2: position dissociation.

For a small ladder of candidate positions, extract a diff-mean direction
at each position and score AUROC on both the harmfulness and refusal
attributes. Compute crossover-Δs.

Pre-registered thresholds
  - crossover_h := AUROC_h(t_final_instr) - AUROC_h(t_post_instr) >= 0.05
  - crossover_r := AUROC_r(t_post_instr) - AUROC_r(t_final_instr) >= 0.05
  - peaks distinguishable from immediate neighbours: bootstrap 95% CIs
    of the peak position and its adjacent positions must NOT overlap.

Success == both crossovers pass AND both peaks distinguishable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prep", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    return p.parse_args()


def bootstrap_auroc(y_true, scores, n_bootstrap=1000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y_true)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    boots = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yb, sb = y_true[idx], scores[idx]
        if len(np.unique(yb)) < 2:
            continue
        boots.append(roc_auc_score(yb, sb))
    if not boots:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "auroc": float(np.mean(boots)),
        "ci_low": float(np.percentile(boots, 2.5)),
        "ci_high": float(np.percentile(boots, 97.5)),
    }


def score_at(stack, layer, pos_train, neg_train, pos_val, neg_val, seed=0, n_bootstrap=1000):
    """Diff-mean direction from train, probe on val, bootstrap AUROC CI."""
    if len(pos_train) < 2 or len(neg_train) < 2 or len(pos_val) < 2 or len(neg_val) < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    direction = stack[pos_train, layer, :].mean(axis=0) - stack[neg_train, layer, :].mean(axis=0)
    u = direction / (np.linalg.norm(direction) + 1e-9)
    X_tr = np.concatenate([stack[pos_train, layer, :] @ u, stack[neg_train, layer, :] @ u]).reshape(-1, 1)
    y_tr = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))])
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)
    X_val = np.concatenate([stack[pos_val, layer, :] @ u, stack[neg_val, layer, :] @ u]).reshape(-1, 1)
    y_val = np.concatenate([np.ones(len(pos_val)), np.zeros(len(neg_val))])
    scores = clf.decision_function(X_val)
    return bootstrap_auroc(y_val, scores, n_bootstrap=n_bootstrap, seed=seed)


def main():
    args = parse_args()
    prep_dir = Path(args.prep)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((prep_dir / "meta.json").read_text())
    dirs_meta = json.loads((prep_dir / "directions.json").read_text())
    acts = torch.load(prep_dir / "activations.pt", weights_only=False)
    responses = [json.loads(l) for l in open(prep_dir / "responses.jsonl")]

    n = meta["n_pairs"]
    tr_idx = np.array(meta["split"]["train"])
    val_idx = np.array(meta["split"]["val"])
    best_h_layer = dirs_meta["best_h"]["layer"]
    best_r_layer = dirs_meta["best_r"]["layer"]

    positions = list(acts["harmful"].keys())

    def _stack(pos): return np.concatenate([acts["harmful"][pos].float().numpy(),
                                            acts["benign"][pos].float().numpy()], axis=0)

    # Indices
    harm_pos = np.arange(0, n)
    harm_neg = np.arange(n, 2 * n)
    # Refusal contrast: HARMFUL side only.
    refused_h = np.array([r["refused"] for r in responses if r["class"] == "harmful"])
    refused_all = np.where(refused_h)[0]
    complied_all = np.where(~refused_h)[0]

    tr_harm_pos = harm_pos[np.isin(harm_pos, tr_idx)]
    tr_harm_neg = harm_neg[np.isin(harm_neg - n, tr_idx)]
    val_harm_pos = harm_pos[np.isin(harm_pos, val_idx)]
    val_harm_neg = harm_neg[np.isin(harm_neg - n, val_idx)]
    tr_ref = refused_all[np.isin(refused_all % n, tr_idx)]
    tr_com = complied_all[np.isin(complied_all % n, tr_idx)]
    val_ref = refused_all[np.isin(refused_all % n, val_idx)]
    val_com = complied_all[np.isin(complied_all % n, val_idx)]

    rows = []
    for pn in positions:
        stack = _stack(pn)
        # harmfulness at h's best layer
        row_h = score_at(stack, best_h_layer, tr_harm_pos, tr_harm_neg,
                         val_harm_pos, val_harm_neg,
                         seed=args.seed, n_bootstrap=args.n_bootstrap)
        # refusal at r's best layer
        row_r = score_at(stack, best_r_layer, tr_ref, tr_com,
                         val_ref, val_com,
                         seed=args.seed, n_bootstrap=args.n_bootstrap)
        rows.append({
            "position": pn,
            "auroc_harmfulness": row_h["auroc"],
            "harm_ci_low": row_h["ci_low"], "harm_ci_high": row_h["ci_high"],
            "auroc_refusal": row_r["auroc"],
            "ref_ci_low": row_r["ci_low"], "ref_ci_high": row_r["ci_high"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "position_auroc_heatmap.csv", index=False)

    def _get(pos, col): return float(df.loc[df["position"] == pos, col].iloc[0])
    def _finite(x): return isinstance(x, float) and not np.isnan(x)

    crossover_h = _get("t_final_instr", "auroc_harmfulness") - _get("t_post_instr", "auroc_harmfulness")
    crossover_r = _get("t_post_instr", "auroc_refusal") - _get("t_final_instr", "auroc_refusal")

    # Peak-distinguishability from neighbours.
    def _ci_disjoint(peak_pos, neighbour_pos, col_low, col_high, col_val):
        peak_lo = _get(peak_pos, col_low); peak_hi = _get(peak_pos, col_high)
        nbr_lo = _get(neighbour_pos, col_low); nbr_hi = _get(neighbour_pos, col_high)
        peak_val = _get(peak_pos, col_val); nbr_val = _get(neighbour_pos, col_val)
        if not all(_finite(x) for x in [peak_lo, peak_hi, nbr_lo, nbr_hi, peak_val, nbr_val]):
            return False
        # peak strictly greater and its CI does not overlap neighbour CI
        return (peak_val > nbr_val) and (peak_lo > nbr_hi)

    h_peak = "t_final_instr"
    h_neighbours = ["t_final_instr-1", "t_final_instr-2"]
    r_peak = "t_post_instr"
    r_neighbours = ["t_post_instr-1", "t_post_instr-2"]

    peak_h_distinguishable = all(
        _ci_disjoint(h_peak, nbr, "harm_ci_low", "harm_ci_high", "auroc_harmfulness")
        for nbr in h_neighbours
    )
    peak_r_distinguishable = all(
        _ci_disjoint(r_peak, nbr, "ref_ci_low", "ref_ci_high", "auroc_refusal")
        for nbr in r_neighbours
    )

    THR_CROSSOVER = 0.05
    crossover_h_pass = _finite(crossover_h) and crossover_h >= THR_CROSSOVER
    crossover_r_pass = _finite(crossover_r) and crossover_r >= THR_CROSSOVER

    all_pass = crossover_h_pass and crossover_r_pass and peak_h_distinguishable and peak_r_distinguishable
    partial = crossover_h_pass or crossover_r_pass
    verdict = "supported" if all_pass else ("partial" if partial else "not-supported")

    verdict_obj = {
        "claim": "C2",
        "verdict": verdict,
        "crossover_h": crossover_h,
        "crossover_r": crossover_r,
        "crossover_h_pass": bool(crossover_h_pass),
        "crossover_r_pass": bool(crossover_r_pass),
        "peak_h_distinguishable": bool(peak_h_distinguishable),
        "peak_r_distinguishable": bool(peak_r_distinguishable),
        "thresholds": {"crossover_min": THR_CROSSOVER},
        "auroc_h_at_t_final_instr": _get("t_final_instr", "auroc_harmfulness"),
        "auroc_h_at_t_post_instr": _get("t_post_instr", "auroc_harmfulness"),
        "auroc_r_at_t_final_instr": _get("t_final_instr", "auroc_refusal"),
        "auroc_r_at_t_post_instr": _get("t_post_instr", "auroc_refusal"),
    }
    with open(out_dir / "claim2_verdict.json", "w") as f:
        json.dump(verdict_obj, f, indent=2)
    print(json.dumps(verdict_obj, indent=2), flush=True)


if __name__ == "__main__":
    main()
