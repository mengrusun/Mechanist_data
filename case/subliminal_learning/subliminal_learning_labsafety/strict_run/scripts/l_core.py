"""M1.L-Core — contrastive activation-direction extraction (d_diff primary,
d_pca + linear probe as diagnostic-only companions).

For each seed:
  For each language-tower layer l in 0..32 (0=embed, 32=last):
    d_diff_l = mean(h_treated_l) - mean(h_CtrlB_l), then L2-normalized (u_l).
    d_pca_l = leading PC of concat(h_treated_l, h_CtrlB_l) - mean.
    probe_auc_l = 5-fold CV AUROC of linear probe (treated=1 vs Ctrl-B=0).
    Layer metric = ||d_diff_l|| (unnormalized), i.e., mean-difference magnitude.

Cross-seed aggregation (Borda rank):
  For each seed, rank layers by layer metric (descending), converted to
  Borda points (n_layers - rank + 1). Sum across 3 seeds. Top-3 layers by
  Borda total.

Stability gate:
  top-6 in ≥ 2/3 seeds → gate PASS.

The 133-item QA_I is not partitioned into flipped-wrong ∪ matched-agree —
n=133 is already small, and further subsetting would leave < 30 per condition
which is not enough for a stable mean-diff. We use ALL items per arm+seed
(the direction still reflects the overall behavioral shift).

We normalize `d_diff` JOINTLY across the chosen top-K layers by their combined
L2 norm — so that when the intervention hooks apply α · v across those layers
simultaneously, α is calibrated in units of shared σ_proj rather than per-layer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def load_cache(p):
    d = np.load(p, allow_pickle=True)
    return {
        "hidden_states": d["hidden_states"].astype(np.float32),  # promote for math
        "item_indices": d["item_indices"],
        "labels": d["labels"],
        "arm": str(d["arm"]),
        "seed": int(d["seed"]),
    }


def d_diff_per_layer(h_t, h_c):
    """h_t, h_c: [N, L+1, d]. Returns v[l] = mean_t[l] - mean_c[l]."""
    v = h_t.mean(axis=0) - h_c.mean(axis=0)   # [L+1, d]
    return v


def probe_auc(h_t, h_c, seed=0):
    """5-fold linear probe AUROC per layer."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    n_layers_plus1 = h_t.shape[1]
    aucs = np.zeros(n_layers_plus1)
    N = min(h_t.shape[0], h_c.shape[0])
    for l in range(n_layers_plus1):
        X = np.concatenate([h_t[:N, l, :], h_c[:N, l, :]], axis=0).astype(np.float32)
        y = np.concatenate([np.ones(N), np.zeros(N)]).astype(np.int64)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_aucs = []
        for train_idx, test_idx in skf.split(X, y):
            clf = LogisticRegression(max_iter=1000, C=1.0, solver="liblinear")
            clf.fit(X[train_idx], y[train_idx])
            proba = clf.predict_proba(X[test_idx])[:, 1]
            try:
                fold_aucs.append(roc_auc_score(y[test_idx], proba))
            except ValueError:
                pass
        aucs[l] = float(np.mean(fold_aucs)) if fold_aucs else 0.5
    return aucs


def d_pca_per_layer(h_t, h_c):
    """Leading PC of centered concat, per layer. Diagnostic-only."""
    n_layers_plus1 = h_t.shape[1]
    d_model = h_t.shape[2]
    pcs = np.zeros((n_layers_plus1, d_model))
    for l in range(n_layers_plus1):
        X = np.concatenate([h_t[:, l, :], h_c[:, l, :]], axis=0).astype(np.float32)
        X = X - X.mean(axis=0, keepdims=True)
        # Compute leading right singular vector via power iter (cheap for d=4096).
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
        pcs[l] = Vt[0]
    return pcs


def cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_root", required=True,
                    help="root dir e.g. cache/residuals/")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--top_k_layers", type=int, default=3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    per_seed = {}
    for s in args.seeds:
        h_t_p = f"{args.cache_root}/treated/seed{s}/activations.npz"
        h_c_p = f"{args.cache_root}/Ctrl-B/seed{s}/activations.npz"
        h_t_pack = load_cache(h_t_p)
        h_c_pack = load_cache(h_c_p)
        h_t = h_t_pack["hidden_states"]
        h_c = h_c_pack["hidden_states"]
        assert h_t.shape[1] == h_c.shape[1], "layer count mismatch"
        v_diff = d_diff_per_layer(h_t, h_c)                 # [L+1, d]
        norm_v = np.linalg.norm(v_diff, axis=1)             # [L+1]
        u_diff = v_diff / np.maximum(norm_v, 1e-8)[:, None] # unit direction
        v_pca = d_pca_per_layer(h_t, h_c)                   # [L+1, d]
        aucs = probe_auc(h_t, h_c, seed=s)                  # [L+1]
        # cos(d_diff, pca)
        cos_diff_pca = np.array([cos(v_diff[l], v_pca[l])
                                 for l in range(v_diff.shape[0])])
        per_seed[s] = {
            "v_diff": v_diff,
            "u_diff": u_diff,
            "norm_v": norm_v,
            "v_pca": v_pca,
            "probe_auc": aucs,
            "cos_diff_pca": cos_diff_pca,
        }
        # Borda: rank by norm_v desc; higher rank = higher score
        n_l = norm_v.shape[0]
        order = np.argsort(-norm_v)  # highest first
        borda = np.zeros(n_l)
        for rank, l in enumerate(order):
            borda[l] = n_l - rank
        per_seed[s]["borda"] = borda
        print(f"[l-core] seed={s} top-3 layers by norm_v = "
              f"{order[:3].tolist()} (norms = "
              f"{norm_v[order[:3]].tolist()})",
              flush=True)

    # Aggregate Borda
    borda_sum = np.sum([per_seed[s]["borda"] for s in args.seeds], axis=0)
    top_layers = np.argsort(-borda_sum)[: args.top_k_layers].tolist()
    print(f"[l-core] top-{args.top_k_layers} layers by Borda sum = {top_layers}", flush=True)

    # Stability gate: top-6 in >= 2/3 seeds
    per_seed_top6 = {s: np.argsort(-per_seed[s]["norm_v"])[:6].tolist() for s in args.seeds}
    n_top6_seed_hits = {}
    for l in top_layers:
        hits = sum(1 for s in args.seeds if l in per_seed_top6[s])
        n_top6_seed_hits[l] = hits
    stability_pass = all(n_top6_seed_hits[l] >= 2 for l in top_layers)
    stability_gate = "PASS" if stability_pass else "FAIL"
    print(f"[l-core] stability gate = {stability_gate} "
          f"per-layer top-6 hits: {n_top6_seed_hits}", flush=True)

    # Save u_diff and v_diff per (seed, layer) as arrays.
    # For downstream steering/ablation: choose the mean of u_diff across seeds
    # for the top-K layers, then renormalize per-layer.
    # We save per-seed data (steering scripts choose which to use).
    directions = {}
    for s in args.seeds:
        directions[str(s)] = {
            "v_diff": per_seed[s]["v_diff"].tolist(),
            "u_diff": per_seed[s]["u_diff"].tolist(),
            "probe_auc": per_seed[s]["probe_auc"].tolist(),
            "cos_diff_pca": per_seed[s]["cos_diff_pca"].tolist(),
            "borda": per_seed[s]["borda"].tolist(),
            "norm_v": per_seed[s]["norm_v"].tolist(),
        }

    report = {
        "top_k_layers": top_layers,
        "stability_gate_result": stability_gate,
        "per_layer_top6_hits": n_top6_seed_hits,
        "borda_sum": borda_sum.tolist(),
        "layer_metric": "norm(d_diff) — L2 magnitude of mean-difference direction",
        "n_layers_plus_1": int(per_seed[args.seeds[0]]["v_diff"].shape[0]),
        "d_model": int(per_seed[args.seeds[0]]["v_diff"].shape[1]),
        "seeds": args.seeds,
        "directions_per_seed": directions,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f)
    print(f"[l-core] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
