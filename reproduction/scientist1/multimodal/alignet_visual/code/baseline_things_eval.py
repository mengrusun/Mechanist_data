"""Baseline evaluation of model feature spaces against human-similarity structure.

Uses THINGS concept-image features and computes:
  - Spearman correlation between model pairwise similarity matrix and reference RSM.
  - Reference RSMs: sensevec RSM (proxy for human semantic similarity),
    wordvec RSM (skip-gram), and category-27 co-membership indicator.
  - Triplet odd-one-out accuracy where the ground-truth label is derived from sensevec.

This is the "before alignment" benchmark. After distillation, we rerun the same script
on aligned student features.
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))
import paths


def load_things_feats(path):
    z = np.load(path, allow_pickle=True)
    return z["feats"].astype(np.float32), list(z["ids"])


def load_reference(sensevec_path, ids_order):
    df = pd.read_csv(sensevec_path, sep=",", header=None)
    meta = pd.read_csv(paths.THINGS_META, sep="\t")
    meta_ids = meta["uniqueID"].tolist()
    # sensevec rows correspond to meta rows in order
    id2vec = {u: df.iloc[i].values.astype(np.float32) for i, u in enumerate(meta_ids)}
    return np.stack([id2vec[u] for u in ids_order], axis=0)


def cosine_rsm(x, mask=None):
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-8)
    r = x @ x.T
    if mask is not None:
        r = r * mask
    return r


def sample_triplets(n_items, n_triplets, rng):
    out = np.zeros((n_triplets, 3), dtype=np.int64)
    for i in range(n_triplets):
        out[i] = rng.choice(n_items, 3, replace=False)
    return out


def triplet_ooo_labels(ref_rsm, triplets):
    """For each triplet (a,b,c), the odd-one-out is the concept whose max similarity
    to the other two is smallest (i.e., the "pair" among (a,b,c) has the max
    pairwise similarity; the third one is the odd one out).
    """
    labels = np.zeros(len(triplets), dtype=np.int64)
    for i, (a, b, c) in enumerate(triplets):
        s_ab = ref_rsm[a, b]
        s_ac = ref_rsm[a, c]
        s_bc = ref_rsm[b, c]
        # find the pair with largest similarity → odd is the concept not in that pair
        m = max(s_ab, s_ac, s_bc)
        if m == s_ab:
            labels[i] = 2  # odd is c
        elif m == s_ac:
            labels[i] = 1  # odd is b
        else:
            labels[i] = 0  # odd is a
    return labels


def triplet_predictions(model_rsm, triplets):
    preds = np.zeros(len(triplets), dtype=np.int64)
    for i, (a, b, c) in enumerate(triplets):
        s_ab = model_rsm[a, b]
        s_ac = model_rsm[a, c]
        s_bc = model_rsm[b, c]
        m = max(s_ab, s_ac, s_bc)
        if m == s_ab:
            preds[i] = 2
        elif m == s_ac:
            preds[i] = 1
        else:
            preds[i] = 0
    return preds


def upper_triangle(mat):
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]


def category_masks(ids_order):
    """Return (mask_within, mask_across) using 27 top-down categories.
    mask_within[i,j] = 1 iff i,j share ≥1 top-down category.
    """
    df = pd.read_csv(paths.THINGS_CAT27_TD, sep="\t")
    meta = pd.read_csv(paths.THINGS_META, sep="\t")
    meta_ids = meta["uniqueID"].tolist()
    # first row is header (categories), following rows are 0/1 membership per concept
    # actually pandas reads header automatically
    cats = df.values.astype(np.int32)  # shape (1854, 27)
    id2row = {u: i for i, u in enumerate(meta_ids)}
    mem = np.stack([cats[id2row[u]] for u in ids_order], axis=0)
    # co-membership: mem @ mem.T > 0
    co = (mem @ mem.T > 0).astype(np.int32)
    np.fill_diagonal(co, 0)
    return co


def eval_features(feats, ids_order, sensevec, cat_within, rng, name, out_dict):
    """Return dict of metrics."""
    n = len(ids_order)
    model_rsm = cosine_rsm(feats)
    sensevec_rsm = cosine_rsm(sensevec)

    # Spearman on upper triangle
    ut_model = upper_triangle(model_rsm)
    ut_sense = upper_triangle(sensevec_rsm)
    ut_cat = upper_triangle(cat_within)
    rho_sense, _ = spearmanr(ut_model, ut_sense)
    rho_cat, _ = spearmanr(ut_model, ut_cat)

    # Triplet OOO
    triplets = sample_triplets(n, 20000, rng)
    labels = triplet_ooo_labels(sensevec_rsm, triplets)
    preds = triplet_predictions(model_rsm, triplets)
    acc_triplet_sense = float((preds == labels).mean())

    # Coarse (across-category) vs fine (within-category) alignment
    # Fine: subset RSM to within-category pairs only, and compute correlation
    #   with sensevec on that subset.
    within_idx = ut_cat == 1
    across_idx = ut_cat == 0
    if within_idx.sum() > 100:
        rho_fine, _ = spearmanr(ut_model[within_idx], ut_sense[within_idx])
    else:
        rho_fine = np.nan
    if across_idx.sum() > 100:
        rho_coarse, _ = spearmanr(ut_model[across_idx], ut_sense[across_idx])
    else:
        rho_coarse = np.nan

    # Coarse-level separation: mean similarity within cat vs across cat
    within_mean = ut_model[within_idx].mean()
    across_mean = ut_model[across_idx].mean()
    coarse_gap = within_mean - across_mean

    out_dict[name] = dict(
        rho_sensevec=float(rho_sense),
        rho_cat27=float(rho_cat),
        rho_fine_within=float(rho_fine),
        rho_coarse_across=float(rho_coarse),
        coarse_gap=float(coarse_gap),
        triplet_ooo_acc_vs_sensevec=acc_triplet_sense,
    )
    print(f"[{name}] rho_sensevec={rho_sense:.4f} rho_cat27={rho_cat:.4f} "
          f"rho_fine={rho_fine:.4f} rho_coarse={rho_coarse:.4f} "
          f"coarse_gap={coarse_gap:.4f} tripletOOOvs_sensevec={acc_triplet_sense:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", required=True,
                    help="list of name=path pairs, e.g., dinov2-base=cache/things_feats_dinov2-base.npz")
    ap.add_argument("--out", default=os.path.join(paths.RESULTS_DIR, "baseline_things_eval.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    # Load reference sensevec aligned to first feats' ids order
    first_name, first_path = args.feats[0].split("=", 1)
    _, ids_order = load_things_feats(first_path)
    sensevec = load_reference(paths.THINGS_SENSEVEC, ids_order)
    cat_within = category_masks(ids_order)

    # Filter out ids with all-zero sensevec (10 concepts)
    keep = np.linalg.norm(sensevec, axis=1) > 1e-6
    if keep.sum() < len(ids_order):
        print(f"Excluding {(~keep).sum()} concepts with zero sensevec.")
    keep_idx = np.where(keep)[0]
    ids_kept = [ids_order[i] for i in keep_idx]
    sensevec_kept = sensevec[keep_idx]
    cat_within_kept = cat_within[np.ix_(keep_idx, keep_idx)]

    results = {}
    ids_kept_set = set(ids_kept)
    for spec in args.feats:
        name, path = spec.split("=", 1)
        feats, ids = load_things_feats(path)
        ids = [str(u) for u in ids]
        id2row = {u: i for i, u in enumerate(ids)}
        # Reorder to match ids_kept
        missing = [u for u in ids_kept if u not in id2row]
        if missing:
            print(f"[{name}] missing {len(missing)} kept ids; skipping ({missing[:3]})")
            continue
        rows = [id2row[u] for u in ids_kept]
        feats_kept = feats[rows]
        eval_features(feats_kept, ids_kept, sensevec_kept, cat_within_kept, rng, name, results)

    import json
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
