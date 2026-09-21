"""
M8_C2_queryability — Text-query → component retrieval (P2a).

For each ImageNet-1k class-name text query t:
  Rank all last-layer components by cos(v_c, e_t).
  Record the rank of the ground-truth class-tied component (i.e., v_c for class c has rank r for query "class c").

Report: MRR, Recall@10.
Compare against permutation baseline (component ↔ class assignment shuffled 1000×, bootstrap CI).

Output JSON:
    {
      "k": ..., "pool": ...,
      "n_queries": ..., "n_components": ...,
      "mrr": ..., "recall_at_1": ..., "recall_at_5": ..., "recall_at_10": ...,
      "permutation_mrr_ci_95_upper": ...,
      "significant": bool
    }
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import h5py


def compute_ranks(v_c_last: np.ndarray, text_emb: np.ndarray, class_labels: np.ndarray) -> np.ndarray:
    """For each class c, find the rank of the component whose class_label == c under text-query c.

    Returns: (n_classes,) integer ranks (1-indexed; 1 = best).
    """
    # cos: (n_comp, n_classes)
    v = v_c_last / (np.linalg.norm(v_c_last, axis=1, keepdims=True) + 1e-8)
    t = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
    cos = v @ t.T  # (n_comp, n_classes)

    # For a fixed class c, we rank components by cos[:, c]  (descending).
    # The ground-truth component is the one whose class_label == c.
    ranks = np.zeros(text_emb.shape[0], dtype=np.int32)
    # For efficiency, argsort each column
    for c in range(text_emb.shape[0]):
        gt_comp_mask = np.where(class_labels == c)[0]
        if len(gt_comp_mask) == 0:
            ranks[c] = -1  # missing GT (should not happen for fc)
            continue
        gt_i = int(gt_comp_mask[0])
        col = cos[:, c]
        # rank = 1 + count of components strictly higher; ties broken by index (stable)
        higher = int((col > col[gt_i]).sum())
        ranks[c] = higher + 1
    return ranks


def mrr_from_ranks(ranks: np.ndarray) -> float:
    valid = ranks[ranks > 0]
    return float((1.0 / valid).mean())


def recall_at_k(ranks: np.ndarray, k: int) -> float:
    valid = ranks[ranks > 0]
    return float((valid <= k).mean())


def permutation_mrr_ci_upper(cos: np.ndarray, class_labels: np.ndarray, n_permute: int = 1000,
                              seed: int = 42) -> float:
    """Shuffle class_labels 1000 times, compute permuted MRR each time, return 95th percentile."""
    rng = np.random.default_rng(seed)
    n_classes = cos.shape[1]
    mrrs = np.zeros(n_permute, dtype=np.float32)
    lbl = class_labels.copy()
    for i in range(n_permute):
        perm = rng.permutation(lbl)
        # For each class c, rank component whose perm-label == c
        ranks = np.zeros(n_classes, dtype=np.int32)
        for c in range(n_classes):
            gt_idx = np.where(perm == c)[0]
            if len(gt_idx) == 0:
                ranks[c] = -1
                continue
            gt_i = int(gt_idx[0])
            col = cos[:, c]
            ranks[c] = int((col > col[gt_i]).sum()) + 1
        mrrs[i] = mrr_from_ranks(ranks)
    return float(np.quantile(mrrs, 0.95))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--v_c", required=True)
    p.add_argument("--text", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_permute", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    with h5py.File(args.v_c, "r") as h5:
        v_c_all = h5["v_c"][:]
        layer = h5["layer"][:].astype("U16")
        class_lbl = h5["class_label"][:]
        k = int(h5.attrs["k"])
        pool = str(h5.attrs["pool"])

    with h5py.File(args.text, "r") as h5t:
        text_emb = h5t["imagenet1k_classes/embeddings"][:]  # (1000, D)

    last_mask = (layer == "fc")
    v_c_last = v_c_all[last_mask]
    lbl_last = class_lbl[last_mask]

    # cos matrix used for both actual and permutation
    v = v_c_last / (np.linalg.norm(v_c_last, axis=1, keepdims=True) + 1e-8)
    t = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
    cos = v @ t.T

    # Actual ranks
    ranks = compute_ranks(v_c_last, text_emb, lbl_last)
    mrr = mrr_from_ranks(ranks)
    r1 = recall_at_k(ranks, 1)
    r5 = recall_at_k(ranks, 5)
    r10 = recall_at_k(ranks, 10)

    perm_upper = permutation_mrr_ci_upper(cos, lbl_last, n_permute=args.n_permute, seed=args.seed)
    significant = bool(mrr > perm_upper)

    result = {
        "k": k,
        "pool": pool,
        "n_queries": int(len(ranks)),
        "n_components": int(len(v_c_last)),
        "mrr": mrr,
        "recall_at_1": r1,
        "recall_at_5": r5,
        "recall_at_10": r10,
        "permutation_mrr_ci_95_upper": perm_upper,
        "significant": significant,
        "n_permute": args.n_permute,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(f"[M8] wrote {args.out}: MRR={mrr:.4f} (perm95 upper {perm_upper:.4f}), R@10={r10:.4f}, sig={significant}")


if __name__ == "__main__":
    main()
