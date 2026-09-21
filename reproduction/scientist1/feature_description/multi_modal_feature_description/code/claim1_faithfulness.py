"""Claim 1: top-K activating images per component form a concept-faithful summary.

Two probes for each of ResNet-50 layer4 / ViT-B/16 CLS / ViT-B/16 mean:

(a) CLIP semantic coherence
    For each component, take the K=9 top-activation images. Compute the average
    pairwise cosine similarity between their (frozen) CLIP embeddings. Compare
    to the same statistic on K random images. Repeated for 200 random draws to
    build a stable baseline.

(b) Class purity
    For each component, take the top-K images' ImageNet labels and compute:
        purity  = max class count / K
    Compare to the class purity of K random images.

Both probes are averaged across components and reported with 95% bootstrap CIs.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from collections import Counter

import numpy as np

from code.common import OUT_DIR

K = 9
N_RAND_DRAWS = 500  # random baseline replications


def _pair_sim_mean(vecs: np.ndarray) -> np.ndarray:
    """Given (M, K, D) normalized vecs, return (M,) mean pairwise cosine sim (excl diag)."""
    # gram: (M, K, K)
    gram = np.einsum("mkd,mld->mkl", vecs, vecs)
    K_ = gram.shape[-1]
    tri = np.triu_indices(K_, k=1)
    pairs = gram[:, tri[0], tri[1]]  # (M, K*(K-1)/2)
    return pairs.mean(axis=1)


def _class_purity(labels: np.ndarray) -> np.ndarray:
    """Given (M, K) labels, return (M,) max-class-fraction."""
    M, K_ = labels.shape
    out = np.zeros(M, dtype=np.float32)
    for i in range(M):
        c = Counter(labels[i].tolist())
        out[i] = max(c.values()) / K_
    return out


def _bootstrap_ci(x: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyze(tag: str, clip_img: np.ndarray, labels: np.ndarray, comp: dict) -> dict:
    v_top_idx = comp["topk_idx"]  # (C, K)
    v_top_lbl = comp["topk_labels"]  # (C, K)
    C, K_ = v_top_idx.shape
    N, D = clip_img.shape
    print(f"[claim1:{tag}] C={C} K={K_} N={N}")

    # per-component semantic coherence: average pairwise CLIP cosine among top-K
    top_emb = clip_img[v_top_idx]  # (C, K, D)
    coh = _pair_sim_mean(top_emb)  # (C,)
    pur = _class_purity(v_top_lbl)  # (C,)

    # random baseline: average of many (C, K) random draws, compute the same stats
    rng = np.random.default_rng(0)
    rand_coh_all = np.zeros((N_RAND_DRAWS, C), dtype=np.float32)
    rand_pur_all = np.zeros((N_RAND_DRAWS, C), dtype=np.float32)
    for r in range(N_RAND_DRAWS):
        idx = rng.integers(0, N, size=(C, K_))
        rand_coh_all[r] = _pair_sim_mean(clip_img[idx])
        rand_pur_all[r] = _class_purity(labels[idx])
    # collapse random draws to per-component means then to a single scalar
    rand_coh = rand_coh_all.mean(axis=0)  # (C,)
    rand_pur = rand_pur_all.mean(axis=0)

    coh_ci = _bootstrap_ci(coh)
    rand_coh_ci = _bootstrap_ci(rand_coh)
    pur_ci = _bootstrap_ci(pur)
    rand_pur_ci = _bootstrap_ci(rand_pur)

    result = {
        "tag": tag,
        "num_components": int(C),
        "K": int(K_),
        "clip_coherence": {
            "topk_mean": float(coh.mean()),
            "topk_ci95": [float(coh_ci[0]), float(coh_ci[1])],
            "random_mean": float(rand_coh.mean()),
            "random_ci95": [float(rand_coh_ci[0]), float(rand_coh_ci[1])],
            "topk_gt_random_frac": float((coh > rand_coh).mean()),
        },
        "class_purity": {
            "topk_mean": float(pur.mean()),
            "topk_ci95": [float(pur_ci[0]), float(pur_ci[1])],
            "random_mean": float(rand_pur.mean()),
            "random_ci95": [float(rand_pur_ci[0]), float(rand_pur_ci[1])],
            "topk_gt_random_frac": float((pur > rand_pur).mean()),
        },
    }
    return result


def main():
    feats = np.load(OUT_DIR / "features" / "eval10k.npz")
    clip_img = feats["clip_img"]
    labels = feats["labels"]

    all_results = {}
    for tag in ["resnet50_layer4", "vit_b16_cls", "vit_b16_mean"]:
        with np.load(OUT_DIR / "components" / f"{tag}.npz") as comp:
            r = analyze(tag, clip_img, labels, {k: comp[k] for k in comp.files})
        all_results[tag] = r
        print(f"[claim1:{tag}] result:\n{json.dumps(r, indent=2)}")

    out_path = OUT_DIR / "claim1_results.json"
    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"[claim1] saved -> {out_path}")


if __name__ == "__main__":
    main()
