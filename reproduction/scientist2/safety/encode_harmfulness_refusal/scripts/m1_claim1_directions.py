#!/usr/bin/env python3
"""M1 — Claim 1: existence + linearity + non-collinearity.

Runs three sub-tests on the cached activations from M-prep:
  (i)   probe AUROC of h at (best_h_layer, t_final_instr) on harmfulness;
        probe AUROC of r at (best_r_layer, t_post_instr) on refusal;
        + baselines: random-direction and *other* direction on same attribute.
  (ii)  cosine(h, r) at each direction's best (layer, position);
        + split-half within-direction cosine baseline.
  (iii) independence sanity: extract r from a shuffled-refusal contrast set;
        verify probe-AUROC of shuffled-r on the true refusal attribute drops to chance.

Pre-registered thresholds
  - AUROC(h on harmfulness) ≥ 0.85 AND AUROC(r on refusal) ≥ 0.85
  - cosine(h, r) ≤ 0.5 × split-half reference
  - AUROC(shuffled-r on refusal) ∈ [0.45, 0.55]

Success == all three. Outputs a verdict JSON + supporting tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prep", required=True, help="Path to results/m_prep/")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_bootstrap", type=int, default=1000)
    return p.parse_args()


def load_prep(prep_dir: Path) -> Dict:
    meta = json.loads((prep_dir / "meta.json").read_text())
    dirs_meta = json.loads((prep_dir / "directions.json").read_text())
    directions = torch.load(prep_dir / "directions.pt", weights_only=False)
    acts = torch.load(prep_dir / "activations.pt", weights_only=False)
    responses = [json.loads(l) for l in open(prep_dir / "responses.jsonl")]
    return {
        "meta": meta,
        "dirs_meta": dirs_meta,
        "directions": directions,
        "acts": acts,
        "responses": responses,
    }


def bootstrap_auroc_ci(y_true, scores, n_bootstrap=1000, seed=0):
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
        return float("nan"), float("nan")
    return float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def auroc_probe_on_projection(
    stack: np.ndarray,
    pos_train: np.ndarray,
    neg_train: np.ndarray,
    pos_val: np.ndarray,
    neg_val: np.ndarray,
    layer: int,
    direction: np.ndarray,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> Dict:
    u = direction / (np.linalg.norm(direction) + 1e-9)
    X_tr = np.concatenate([
        stack[pos_train, layer, :] @ u,
        stack[neg_train, layer, :] @ u,
    ]).reshape(-1, 1)
    y_tr = np.concatenate([np.ones(len(pos_train)), np.zeros(len(neg_train))])
    if len(np.unique(y_tr)) < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_tr, y_tr)

    X_val_pos = stack[pos_val, layer, :] @ u
    X_val_neg = stack[neg_val, layer, :] @ u
    X_val = np.concatenate([X_val_pos, X_val_neg]).reshape(-1, 1)
    y_val = np.concatenate([np.ones(len(pos_val)), np.zeros(len(neg_val))])
    if len(np.unique(y_val)) < 2:
        return {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    scores = clf.decision_function(X_val)
    auroc = float(roc_auc_score(y_val, scores))
    lo, hi = bootstrap_auroc_ci(y_val, scores, n_bootstrap=n_bootstrap, seed=seed)
    return {"auroc": auroc, "ci_low": lo, "ci_high": hi, "n_val_pos": int(len(pos_val)), "n_val_neg": int(len(neg_val))}


def split_half_cosine(stack: np.ndarray, pos_idx: np.ndarray, neg_idx: np.ndarray, layer: int,
                      n_iter: int = 20, seed: int = 0) -> float:
    """Split the direction-extraction pool in half N times, compute the diff-mean
    on each half, average the cos(A, B) across iterations. This gives the
    "same direction" noise floor.
    """
    rng = np.random.default_rng(seed)
    cosines = []
    for _ in range(n_iter):
        pp = rng.permutation(len(pos_idx))
        nn = rng.permutation(len(neg_idx))
        pa, pb = pos_idx[pp[: len(pp) // 2]], pos_idx[pp[len(pp) // 2 :]]
        na, nb = neg_idx[nn[: len(nn) // 2]], neg_idx[nn[len(nn) // 2 :]]
        if min(len(pa), len(pb), len(na), len(nb)) < 2:
            continue
        dA = stack[pa, layer, :].mean(axis=0) - stack[na, layer, :].mean(axis=0)
        dB = stack[pb, layer, :].mean(axis=0) - stack[nb, layer, :].mean(axis=0)
        c = float(np.dot(dA, dB) / (np.linalg.norm(dA) * np.linalg.norm(dB) + 1e-9))
        cosines.append(c)
    if not cosines:
        return float("nan")
    return float(np.mean(cosines))


def main():
    args = parse_args()
    prep = load_prep(Path(args.prep))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = prep["meta"]
    n = meta["n_pairs"]
    tr_idx = np.array(meta["split"]["train"])
    val_idx = np.array(meta["split"]["val"])
    test_idx = np.array(meta["split"]["test"])

    dirs = prep["directions"]
    best_h = dirs["best_h"]
    best_r = dirs["best_r"]
    h = dirs["h"].numpy()
    r = dirs["r"].numpy()

    acts = prep["acts"]
    responses = prep["responses"]

    # Build stacked (2n, L+1, d) at h's position and r's position.
    def _stack(pos_name):
        return np.concatenate([acts["harmful"][pos_name].float().numpy(),
                               acts["benign"][pos_name].float().numpy()], axis=0)
    stack_h = _stack(best_h["position"])
    stack_r = _stack(best_r["position"])

    # Index arrays
    harm_pos = np.arange(0, n)
    harm_neg = np.arange(n, 2 * n)

    # Refusal contrast: HARMFUL side only (matches m_prep after fix 6).
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

    # ------------------------------------------------------------ sub-test i
    rng = np.random.default_rng(args.seed)
    d = h.shape[0]

    # AUROC(h) on harmfulness attribute
    row_h = auroc_probe_on_projection(
        stack_h, tr_harm_pos, tr_harm_neg, val_harm_pos, val_harm_neg,
        layer=best_h["layer"], direction=h,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    # AUROC(r) on refusal attribute
    if len(val_ref) >= 2 and len(val_com) >= 2:
        row_r = auroc_probe_on_projection(
            stack_r, tr_ref, tr_com, val_ref, val_com,
            layer=best_r["layer"], direction=r,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
    else:
        row_r = {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                 "n_val_pos": int(len(val_ref)), "n_val_neg": int(len(val_com))}

    # Baseline: random direction (matched norm) at each site
    def _random_norm_dir(norm, seed):
        rng2 = np.random.default_rng(seed)
        v = rng2.standard_normal(d).astype(np.float32)
        v = v / (np.linalg.norm(v) + 1e-9) * norm
        return v
    rand_h = _random_norm_dir(np.linalg.norm(h), args.seed + 1)
    rand_r = _random_norm_dir(np.linalg.norm(r), args.seed + 2)

    row_random_on_harm = auroc_probe_on_projection(
        stack_h, tr_harm_pos, tr_harm_neg, val_harm_pos, val_harm_neg,
        layer=best_h["layer"], direction=rand_h,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    if len(val_ref) >= 2 and len(val_com) >= 2:
        row_random_on_ref = auroc_probe_on_projection(
            stack_r, tr_ref, tr_com, val_ref, val_com,
            layer=best_r["layer"], direction=rand_r,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
    else:
        row_random_on_ref = {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    # Baseline: the *other* direction on the same attribute (using their own probing site)
    row_r_on_harm = auroc_probe_on_projection(
        stack_h, tr_harm_pos, tr_harm_neg, val_harm_pos, val_harm_neg,
        layer=best_h["layer"], direction=r,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    if len(val_ref) >= 2 and len(val_com) >= 2:
        row_h_on_ref = auroc_probe_on_projection(
            stack_r, tr_ref, tr_com, val_ref, val_com,
            layer=best_r["layer"], direction=h,
            n_bootstrap=args.n_bootstrap, seed=args.seed,
        )
    else:
        row_h_on_ref = {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    auroc_rows = [
        {"direction": "h", "attribute": "harmfulness",
         "layer": best_h["layer"], "position": best_h["position"],
         **row_h},
        {"direction": "r", "attribute": "refusal",
         "layer": best_r["layer"], "position": best_r["position"],
         **row_r},
        {"direction": "random_norm", "attribute": "harmfulness",
         "layer": best_h["layer"], "position": best_h["position"],
         **row_random_on_harm},
        {"direction": "random_norm", "attribute": "refusal",
         "layer": best_r["layer"], "position": best_r["position"],
         **row_random_on_ref},
        {"direction": "r_on_harm", "attribute": "harmfulness",
         "layer": best_h["layer"], "position": best_h["position"],
         **row_r_on_harm},
        {"direction": "h_on_ref", "attribute": "refusal",
         "layer": best_r["layer"], "position": best_r["position"],
         **row_h_on_ref},
    ]
    pd.DataFrame(auroc_rows).to_csv(out_dir / "auroc_table.csv", index=False)

    # ------------------------------------------------------------ sub-test ii
    cos_hr = float(np.dot(h, r) / (np.linalg.norm(h) * np.linalg.norm(r) + 1e-9))
    # Split-half within-direction cosine at h's site
    split_half_h = split_half_cosine(stack_h, tr_harm_pos, tr_harm_neg, best_h["layer"],
                                     n_iter=20, seed=args.seed)
    if len(tr_ref) >= 4 and len(tr_com) >= 4:
        split_half_r = split_half_cosine(stack_r, tr_ref, tr_com, best_r["layer"],
                                         n_iter=20, seed=args.seed)
    else:
        split_half_r = float("nan")

    split_half_reference = float(np.nanmean([split_half_h, split_half_r]))
    cosine_report = {
        "cosine_h_r": cos_hr,
        "split_half_within_h": split_half_h,
        "split_half_within_r": split_half_r,
        "split_half_reference": split_half_reference,
        "cosine_ratio_vs_reference": cos_hr / (abs(split_half_reference) + 1e-9),
    }
    with open(out_dir / "cosine_report.json", "w") as f:
        json.dump(cosine_report, f, indent=2)

    # ---------------------------------------------------------- sub-test iii
    # Shuffled-refusal contrast: permute the refused labels within the training
    # pool, extract shuffled-r, score its AUROC on the *true* val refusal labels.
    stack_r_full = stack_r
    # Combine train ref+com indices, shuffle labels
    pool_idx = np.concatenate([tr_ref, tr_com])
    labels = np.concatenate([np.ones(len(tr_ref)), np.zeros(len(tr_com))])
    if len(pool_idx) >= 4:
        rng2 = np.random.default_rng(args.seed + 42)
        shuffled = rng2.permutation(labels)
        shuf_ref = pool_idx[shuffled == 1]
        shuf_com = pool_idx[shuffled == 0]
        if len(shuf_ref) >= 2 and len(shuf_com) >= 2:
            shuffled_r = stack_r_full[shuf_ref, best_r["layer"], :].mean(axis=0) - \
                         stack_r_full[shuf_com, best_r["layer"], :].mean(axis=0)
            row_shuf = auroc_probe_on_projection(
                stack_r_full, tr_ref, tr_com, val_ref, val_com,
                layer=best_r["layer"], direction=shuffled_r,
                n_bootstrap=args.n_bootstrap, seed=args.seed,
            )
        else:
            row_shuf = {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    else:
        row_shuf = {"auroc": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}

    with open(out_dir / "shuffled_refusal.json", "w") as f:
        json.dump({"shuffled_r_on_true_refusal": row_shuf}, f, indent=2)

    # ---------------------------------------------------------- verdict
    # Thresholds (pre-registered).
    THR_AUROC = 0.85
    THR_COS_RATIO = 0.5  # cosine(h,r) at most 0.5 x split-half reference
    THR_SHUF_LO, THR_SHUF_HI = 0.45, 0.55

    def _finite(x): return isinstance(x, float) and not np.isnan(x)

    sub_i_pass = (
        _finite(row_h["auroc"]) and row_h["auroc"] >= THR_AUROC and
        _finite(row_r["auroc"]) and row_r["auroc"] >= THR_AUROC
    )
    sub_ii_pass = _finite(cos_hr) and _finite(split_half_reference) and (
        abs(cos_hr) <= THR_COS_RATIO * abs(split_half_reference)
    )
    sub_iii_pass = _finite(row_shuf["auroc"]) and (THR_SHUF_LO <= row_shuf["auroc"] <= THR_SHUF_HI)

    verdict = "supported" if (sub_i_pass and sub_ii_pass and sub_iii_pass) else (
        "partial" if (sub_i_pass or sub_ii_pass) else "not-supported"
    )
    key_stats = {
        "auroc_h_on_harmfulness": row_h.get("auroc"),
        "auroc_r_on_refusal": row_r.get("auroc"),
        "auroc_random_on_harmfulness": row_random_on_harm.get("auroc"),
        "auroc_random_on_refusal": row_random_on_ref.get("auroc"),
        "auroc_r_on_harmfulness": row_r_on_harm.get("auroc"),
        "auroc_h_on_refusal": row_h_on_ref.get("auroc"),
        "cosine_h_r": cos_hr,
        "split_half_reference": split_half_reference,
        "auroc_shuffled_r_on_refusal": row_shuf.get("auroc"),
    }
    verdict_obj = {
        "claim": "C1",
        "verdict": verdict,
        "sub_i_pass": bool(sub_i_pass),
        "sub_ii_pass": bool(sub_ii_pass),
        "sub_iii_pass": bool(sub_iii_pass),
        "thresholds": {"auroc": THR_AUROC, "cos_ratio_of_split_half": THR_COS_RATIO,
                       "shuffled_r_band": [THR_SHUF_LO, THR_SHUF_HI]},
        "key_stats": key_stats,
        "best_h": best_h,
        "best_r": best_r,
        "n_val_harm_pos": int(len(val_harm_pos)),
        "n_val_ref_pos": int(len(val_ref)),
    }
    with open(out_dir / "claim1_verdict.json", "w") as f:
        json.dump(verdict_obj, f, indent=2)
    print(json.dumps(verdict_obj, indent=2), flush=True)


if __name__ == "__main__":
    main()
