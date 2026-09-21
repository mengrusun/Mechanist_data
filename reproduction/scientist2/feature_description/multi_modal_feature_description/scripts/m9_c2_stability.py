"""
M9_C2_stability — Within-component stability via disjoint reference-set halves (P2b).

For each component c:
  Split top-2k reference inputs (2*half_k = 32) into disjoint halves A and B.
  Pool each half into v_c^A, v_c^B using the specified operator.
  cosine = cos(v_c^A, v_c^B).

Report: median cosine across all sampled 2000 components; bootstrap 95% CI.
Pass criterion: median >= tau_stable (= 0.5) at pool=mean.

Output JSON:
    {
      "half_k": ..., "pool": ...,
      "n_components": ...,
      "median_cosine": ..., "ci_95": [...],
      "tau_stable_threshold": 0.5, "passes": bool
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import h5py
from scipy.stats import bootstrap

from common import set_all_seeds


def pool(embs: np.ndarray, weights: np.ndarray | None, mode: str) -> np.ndarray:
    e = embs.astype(np.float32)
    if mode == "mean":
        v = e.mean(axis=0)
    elif mode == "act_weighted_mean":
        w = np.clip(weights.astype(np.float32), 0.0, None)
        s = w.sum() + 1e-8
        w = w / s
        v = (w[:, None] * e).sum(axis=0)
    elif mode == "max":
        v = e.max(axis=0)
    elif mode == "medoid":
        m = e.mean(axis=0)
        m_n = m / (np.linalg.norm(m) + 1e-8)
        e_n = e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)
        v = e[int(np.argmax(e_n @ m_n))]
    else:
        raise ValueError(mode)
    n = np.linalg.norm(v)
    if n > 0:
        v = v / n
    return v


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--refsets", required=True)
    p.add_argument("--clip_emb", required=True)
    p.add_argument("--half_k", type=int, default=16)
    p.add_argument("--pool", required=True, choices=["mean", "act_weighted_mean", "max", "medoid"])
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_all_seeds(args.seed)
    rng = np.random.default_rng(args.seed)

    with h5py.File(args.refsets, "r") as h5r:
        top_idx = h5r["refsets/top256_image_indices"][:]  # (n_comp, 256)
        top_act = h5r["refsets/top256_activations"][:]
    with h5py.File(args.clip_emb, "r") as h5c:
        clip_idx = h5c["image_index"][:]
        clip_emb = h5c["clip_image_embeddings"][:]
    idx_to_row = {int(gi): i for i, gi in enumerate(clip_idx)}

    n_comp = top_idx.shape[0]
    two_k = 2 * args.half_k
    cosines = np.zeros(n_comp, dtype=np.float32)
    for c in range(n_comp):
        picks = top_idx[c, :two_k]
        weights = top_act[c, :two_k]
        # Random half split
        perm = rng.permutation(two_k)
        idx_a = perm[:args.half_k]
        idx_b = perm[args.half_k:]

        # Preserve pairing between rows and weights via a single kept-list zipper.
        rows_a: list[int] = []
        wa_kept: list[float] = []
        for i in idx_a:
            r = idx_to_row.get(int(picks[i]), None)
            if r is None:
                continue
            rows_a.append(r)
            wa_kept.append(float(weights[i]))
        rows_b: list[int] = []
        wb_kept: list[float] = []
        for i in idx_b:
            r = idx_to_row.get(int(picks[i]), None)
            if r is None:
                continue
            rows_b.append(r)
            wb_kept.append(float(weights[i]))
        if not rows_a or not rows_b:
            cosines[c] = 0.0
            continue

        wa = np.asarray(wa_kept, dtype=np.float32)
        wb = np.asarray(wb_kept, dtype=np.float32)
        va = pool(clip_emb[rows_a], wa, args.pool)
        vb = pool(clip_emb[rows_b], wb, args.pool)
        cosines[c] = float(va @ vb)

    median = float(np.median(cosines))
    res = bootstrap((cosines,), np.median, confidence_level=0.95, n_resamples=1000, method="basic",
                    random_state=np.random.default_rng(args.seed))
    ci = [float(res.confidence_interval.low), float(res.confidence_interval.high)]
    tau = 0.5
    passes = bool(median >= tau)

    out = {
        "half_k": args.half_k,
        "pool": args.pool,
        "n_components": int(n_comp),
        "median_cosine": median,
        "mean_cosine": float(cosines.mean()),
        "ci_95": ci,
        "tau_stable_threshold": tau,
        "passes": passes,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[M9] wrote {args.out}: median={median:.4f} (ci {ci}), passes={passes}")


if __name__ == "__main__":
    main()
