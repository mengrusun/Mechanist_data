"""
M4_v_c — Per-component pooled CLIP semantic vectors v_c, across k and pooling operators.

Given the cached CLIP embeddings of the unique reference images (M3) and the per-component
top-k image indices + activations (M2), compute v_c for a specific (k, pool) setting.

Pooling operators:
    - mean:               mean(CLIP_emb(R_c)), then L2-normalize
    - act_weighted_mean:  sum(a_i / sum(a) * CLIP_emb(x_i)), then L2-normalize
                          — the CLIP-Dissect canonical operator (see multi-modal/clip-dissect/SKILL.md)
    - max:                per-dim max across R_c, then L2-normalize (max-pool baseline)
    - medoid:             the R_c member whose embedding is closest (cos) to the mean of all others

Also supports `--random_baseline`: uses k random images (from the same 50k pool) instead of R_c's top-k.
That baseline is used by M6 to compute Δ_pure.

Output HDF5:
    /v_c        : (n_components, 512) float32 — L2-normalized
    /pool       : attribute string
    /k          : attribute int
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import h5py

from common import set_all_seeds


def pool_vectors(embs: np.ndarray, weights: np.ndarray | None, pool: str) -> np.ndarray:
    """embs: (k, D) fp16 L2-normalized. weights: (k,) or None. Return (D,) fp32 L2-normalized."""
    e = embs.astype(np.float32)
    if pool == "mean":
        v = e.mean(axis=0)
    elif pool == "act_weighted_mean":
        w = weights.astype(np.float32)
        w = np.clip(w, a_min=0.0, a_max=None)  # only positive activations (ReLU-like)
        s = w.sum() + 1e-8
        w = w / s
        v = (w[:, None] * e).sum(axis=0)
    elif pool == "max":
        v = e.max(axis=0)
    elif pool == "medoid":
        # find the k-idx whose cosine to mean-of-others is maximal
        # equivalently: pick argmax cos(e_i, mean(e)) — very close to picking most-central
        m = e.mean(axis=0)
        m_n = m / (np.linalg.norm(m) + 1e-8)
        e_n = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        cs = e_n @ m_n
        v = e[int(np.argmax(cs))]
    else:
        raise ValueError(f"unknown pool {pool}")
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--refsets", default="runs/M2_reference_sets/refsets.h5")
    p.add_argument("--clip_emb", default="runs/M3_clip_embeddings/clip_val_embeddings.h5")
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--pool", type=str, required=True, choices=["mean", "act_weighted_mean", "max", "medoid"])
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--random_baseline", action="store_true",
                   help="Replace R_c's top-k with k random images (for M6 Δ_pure baseline).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_all_seeds(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[M4] loading refsets from {args.refsets}")
    with h5py.File(args.refsets, "r") as h5r:
        top_idx = h5r["refsets/top256_image_indices"][:]  # (n_comp, 256)
        top_act = h5r["refsets/top256_activations"][:]     # (n_comp, 256)
        component_id = h5r["components/component_id"][:]
        layer = h5r["components/layer"][:]
        local_idx = h5r["components/local_index"][:]
        class_lbl = h5r["components/class_label"][:]

    n_comp = top_idx.shape[0]
    print(f"[M4] {n_comp} components, k={args.k}, pool={args.pool}, random_baseline={args.random_baseline}")

    print(f"[M4] loading CLIP embeddings from {args.clip_emb}")
    with h5py.File(args.clip_emb, "r") as h5c:
        clip_indices = h5c["image_index"][:]       # (N_unique,)
        clip_emb_all = h5c["clip_image_embeddings"][:]  # (N_unique, D)  fp16
    D = clip_emb_all.shape[1]
    # Build lookup: image_index -> row in clip_emb_all
    idx_to_row = {int(gi): i for i, gi in enumerate(clip_indices)}

    # Prepare random baseline pool (all image indices covered by CLIP embeddings)
    rng = np.random.default_rng(args.seed)
    all_covered = clip_indices.copy()

    # Compute v_c per component
    v_c_all = np.zeros((n_comp, D), dtype=np.float32)
    missing_count = 0
    for c in range(n_comp):
        if args.random_baseline:
            picks = rng.choice(all_covered, size=args.k, replace=False)
            # dummy weights = ones for the baseline (activation-weighted becomes uniform)
            weights = np.ones(args.k, dtype=np.float32)
        else:
            picks = top_idx[c, :args.k]
            weights = top_act[c, :args.k]
        # Preserve alignment between rows and weights: pair (row, weight) at the same list position.
        rows: list[int] = []
        kept_weights: list[float] = []
        for pos in range(len(picks)):
            r = idx_to_row.get(int(picks[pos]), None)
            if r is None:
                missing_count += 1
                continue
            rows.append(r)
            kept_weights.append(float(weights[pos]))
        if not rows:
            # degenerate — leave zero
            continue
        embs = clip_emb_all[rows]  # (k', D)
        w = np.asarray(kept_weights, dtype=np.float32)
        v_c_all[c] = pool_vectors(embs, w, args.pool)

    print(f"[M4] missing count: {missing_count}, computed {n_comp} v_c vectors")
    nan_count = int(np.isnan(v_c_all).sum())
    if nan_count:
        print(f"[M4] WARNING: {nan_count} NaNs in v_c!")

    if out_path.exists():
        out_path.unlink()
    with h5py.File(out_path, "w") as h5o:
        h5o.create_dataset("v_c", data=v_c_all)
        h5o.create_dataset("component_id", data=component_id)
        h5o.create_dataset("layer", data=layer)
        h5o.create_dataset("local_index", data=local_idx)
        h5o.create_dataset("class_label", data=class_lbl)
        h5o.attrs["k"] = int(args.k)
        h5o.attrs["pool"] = args.pool
        h5o.attrs["random_baseline"] = bool(args.random_baseline)
        h5o.attrs["nan_count"] = nan_count
        h5o.attrs["dim"] = D

    print(f"[M4] wrote {out_path}")


if __name__ == "__main__":
    main()
