"""Compute per-(feature, concept, threshold) F1 from confusion tensors, then
report:
  - number of SAE features with best-F1 >= tau for at least one concept
  - number of raw neurons with best-F1 >= tau for at least one concept
  - number of distinct concepts covered by SAE features vs neurons
  - "clean" coverage (best F1 >= 0.75)

Usage:
    python score_f1.py --dir results/layer24 [--tau 0.5]
"""
from __future__ import annotations
import argparse
import numpy as np
import pickle
import os


def load(dir_):
    Z = np.load(os.path.join(dir_, "confusion.npz"))
    with open(os.path.join(dir_, "concepts.pkl"), "rb") as f:
        meta = pickle.load(f)
    return Z, meta


def per_feature_best_f1(TP, PredPos, label_pos, total_residues):
    """
    TP:        (F, C, T) int32
    PredPos:   (F, T)    int64
    label_pos: (C,)      int64

    F1(f,c,t) = 2 TP / (2 TP + FP + FN)
              = 2 TP / (PredPos + label_pos)   (since FP + FN = PredPos + label_pos - 2 TP)
              wait: FP = PredPos - TP; FN = label_pos - TP.
              precision = TP / PredPos ; recall = TP / label_pos
              F1 = 2 TP / (PredPos + label_pos)
    Returns:
      best_f1[f, c]    over thresholds
      best_thr[f, c]
    """
    F, C, T = TP.shape
    denom = PredPos[:, None, :] + label_pos[None, :, None]         # (F, C, T)
    with np.errstate(divide='ignore', invalid='ignore'):
        f1 = np.where(denom > 0, 2.0 * TP / denom, 0.0)            # (F, C, T)
    # best threshold per (f, c)
    best_t = f1.argmax(axis=2)                                     # (F, C)
    best_f1 = np.take_along_axis(f1, best_t[..., None], axis=2)[..., 0]
    return best_f1, best_t


def report(name, best_f1, tau_list, concepts, min_concept_res: int = 0, label_pos: np.ndarray | None = None):
    print(f"\n### {name} ###")
    if label_pos is not None and min_concept_res > 0:
        keep = label_pos >= min_concept_res
        print(f"restricting to {keep.sum()}/{len(concepts)} concepts with >= {min_concept_res} residues")
        best_f1 = best_f1[:, keep]
        kept_concepts = [c for c, k in zip(concepts, keep) if k]
    else:
        kept_concepts = concepts

    for tau in tau_list:
        per_feat_best = best_f1.max(axis=1)                    # (F,)
        n_feats_ok = int((per_feat_best >= tau).sum())
        # For each feature above tau, take its best concept
        argc = best_f1.argmax(axis=1)
        feats_ok_mask = per_feat_best >= tau
        distinct_concepts = np.unique(argc[feats_ok_mask])
        print(f"  tau={tau:.2f}: {n_feats_ok:>6d} features (F1 >= tau); "
              f"cover {len(distinct_concepts):>4d} distinct concepts of {len(kept_concepts)}")
    # Top concept hits
    per_concept_best = best_f1.max(axis=0)                     # (C,)
    order = np.argsort(-per_concept_best)
    print(f"  top-15 concepts by best F1:")
    for i in order[:15]:
        print(f"    {per_concept_best[i]:.3f}  {kept_concepts[i]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tau", type=float, nargs="+", default=[0.3, 0.5, 0.75])
    ap.add_argument("--min_concept_res", type=int, default=50,
                    help="ignore concepts with fewer than this many positive residues")
    args = ap.parse_args()

    Z, meta = load(args.dir)
    concepts = meta["concepts"]
    label_pos = np.array(meta["label_pos"])
    tot = meta["total_residues"]
    print(f"loaded: {len(concepts)} concepts; total residues = {tot}")

    print("=== SAE ===")
    f1_sae, _ = per_feature_best_f1(Z["TP_sae"], Z["PredPos_sae"], label_pos, tot)
    report("SAE features", f1_sae, args.tau, concepts, args.min_concept_res, label_pos)

    print("\n=== Raw neurons ===")
    f1_neu, _ = per_feature_best_f1(Z["TP_neu"], Z["PredPos_neu"], label_pos, tot)
    report("Raw neurons", f1_neu, args.tau, concepts, args.min_concept_res, label_pos)


if __name__ == "__main__":
    main()
