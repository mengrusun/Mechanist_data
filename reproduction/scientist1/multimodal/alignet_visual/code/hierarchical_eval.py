"""Hierarchical (multi-level) similarity evaluation on THINGS.

Levels:
  - COARSE  : cross-category boundaries. Triplets (i, j, k) with i,j sharing
              any of the 27 top-down categories, k from a *different* top-level
              super-category (living vs non-living). Odd-one-out is k.
  - MID     : mid-level. i,j sharing a 27-category, k in a *different* 27-cat
              but still in the same broad living/non-living pool. Odd is k.
  - FINE    : within a shared 27-category, i,j closer under sensevec than to k
              (also within the same 27-cat). Odd is k.

Also computes behaviour-uncertainty match (Claim 3): for each triplet, we compute
a soft OOO probability distribution from the model's similarities (via temperature
softmax) and from a reference (sensevec). Then we compute the Spearman/Pearson
correlation between the two entropy sequences and the KL divergence between the
per-triplet distributions, aggregated.
"""
import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

sys.path.insert(0, os.path.dirname(__file__))
import paths
from baseline_things_eval import (
    load_things_feats, load_reference, upper_triangle,
    cosine_rsm, sample_triplets,
)


# 53 fine categories give us a mid-level structure. Some 27-cats have subgroups
# in the 53-cat table. For a coarse "living vs non-living" super-category we use
# a hand-mapping of the 27 top-down categories.
LIVING27 = {"animal", "bird", "body part", "insect", "fruit", "vegetable", "plant"}
# rest are non-living

CATS27_ORDER = None  # cache


def load_cat27_top_down(ids_kept):
    df = pd.read_csv(paths.THINGS_CAT27_TD, sep="\t")
    global CATS27_ORDER
    CATS27_ORDER = df.columns.tolist()
    meta = pd.read_csv(paths.THINGS_META, sep="\t")
    meta_ids = meta["uniqueID"].tolist()
    id2row = {u: i for i, u in enumerate(meta_ids)}
    mat = df.values.astype(np.int32)  # (1854, 27)
    return np.stack([mat[id2row[u]] for u in ids_kept], axis=0)


def build_hier_triplets(ids_kept, cat27, sensevec, rng, n_per_level=5000):
    """Return dict level -> (triplets [B,3], odd-index [B])."""
    n = len(ids_kept)
    living_col = np.array([1 if c in LIVING27 else 0 for c in CATS27_ORDER], dtype=np.int32)
    # concept living label = any category-membership tagged living
    concept_living = ((cat27 * living_col[None, :]).sum(axis=1) > 0).astype(np.int32)

    # Precompute concept category vectors
    coarse_triplets, coarse_odd = [], []
    mid_triplets, mid_odd = [], []
    fine_triplets, fine_odd = [], []

    # Concept indices per category (27)
    cat_members = {c: np.where(cat27[:, i] == 1)[0]
                   for i, c in enumerate(CATS27_ORDER) if (cat27[:, i] == 1).sum() >= 3}

    # sensevec rsm for fine-level ordering
    sv_rsm = cosine_rsm(sensevec)

    tries = 0
    while (len(coarse_triplets) < n_per_level or len(mid_triplets) < n_per_level
           or len(fine_triplets) < n_per_level) and tries < n_per_level * 40:
        tries += 1
        cats_avail = [c for c in cat_members if len(cat_members[c]) >= 3]
        cat = rng.choice(cats_avail)

        # Choose two i,j from `cat`, k = odd
        pool = cat_members[cat]
        i, j = rng.choice(pool, 2, replace=False)

        # COARSE: k from opposite living/non-living
        opp_pool = np.where(concept_living != concept_living[i])[0]
        opp_pool = np.setdiff1d(opp_pool, pool)
        if len(opp_pool) > 0 and len(coarse_triplets) < n_per_level:
            k = rng.choice(opp_pool)
            coarse_triplets.append([i, j, k]); coarse_odd.append(2)

        # MID: k same living-status as i but in a different 27-category
        same_live = np.where(concept_living == concept_living[i])[0]
        # not in this cat
        same_live = np.setdiff1d(same_live, pool)
        if len(same_live) > 0 and len(mid_triplets) < n_per_level:
            k = rng.choice(same_live)
            mid_triplets.append([i, j, k]); mid_odd.append(2)

        # FINE: k also in cat (same 27-cat) but distinct from i,j and less
        # similar to i (under sensevec) than j is
        if len(fine_triplets) < n_per_level:
            other = np.setdiff1d(pool, [i, j])
            if len(other) > 0:
                # k = another in-cat concept with lower sensevec similarity to i than j has
                s_ij = sv_rsm[i, j]
                s_ok = sv_rsm[i, other]
                candidates = other[s_ok < s_ij - 0.03]
                if len(candidates) > 0:
                    k = rng.choice(candidates)
                    fine_triplets.append([i, j, k]); fine_odd.append(2)

    def to_np(tl, ol):
        return np.array(tl, dtype=np.int64), np.array(ol, dtype=np.int64)

    return {
        "coarse": to_np(coarse_triplets[:n_per_level], coarse_odd[:n_per_level]),
        "mid":    to_np(mid_triplets[:n_per_level],    mid_odd[:n_per_level]),
        "fine":   to_np(fine_triplets[:n_per_level],   fine_odd[:n_per_level]),
    }


def triplet_pred_and_probs(rsm, triplets, tau=8.0):
    """Return (preds [B], probs [B,3] for order (a,b,c)).
    Odd-one-out is the concept whose *pair with the others* has the smallest
    similarity in the pair. Probability(concept=a as odd) is softmax over the
    similarity of the pair not containing it: proportional to exp(tau * s_bc).
    """
    a = triplets[:, 0]; b = triplets[:, 1]; c = triplets[:, 2]
    s_ab = rsm[a, b]; s_ac = rsm[a, c]; s_bc = rsm[b, c]
    # logits: prob of odd = softmax over (s_bc, s_ac, s_ab) for indices (a,b,c)
    logits = np.stack([s_bc, s_ac, s_ab], axis=1) * tau
    m = logits.max(axis=1, keepdims=True)
    p = np.exp(logits - m)
    p /= p.sum(axis=1, keepdims=True)
    preds = p.argmax(axis=1)
    return preds, p


def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def kl(p, q):
    p = np.clip(p, 1e-12, 1.0)
    q = np.clip(q, 1e-12, 1.0)
    return (p * (np.log(p) - np.log(q))).sum(axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(paths.RESULTS_DIR, "hierarchical_eval.json"))
    ap.add_argument("--n-per-level", type=int, default=8000)
    ap.add_argument("--tau", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    first_name, first_path = args.feats[0].split("=", 1)
    _, ids_all = load_things_feats(first_path)
    ids_all = [str(u) for u in ids_all]

    sensevec = load_reference(paths.THINGS_SENSEVEC, ids_all)
    keep = np.linalg.norm(sensevec, axis=1) > 1e-6
    keep_idx = np.where(keep)[0]
    ids_kept = [ids_all[i] for i in keep_idx]
    sensevec = sensevec[keep_idx]
    cat27 = load_cat27_top_down(ids_kept)

    print("Building hierarchical triplets…")
    trs = build_hier_triplets(ids_kept, cat27, sensevec, rng, n_per_level=args.n_per_level)
    for lvl, (tt, oo) in trs.items():
        print(f"  {lvl}: {len(tt)} triplets")

    sv_rsm = cosine_rsm(sensevec)

    results = {}
    for spec in args.feats:
        name, path = spec.split("=", 1)
        feats, ids = load_things_feats(path)
        ids = [str(u) for u in ids]
        id2row = {u: i for i, u in enumerate(ids)}
        missing = [u for u in ids_kept if u not in id2row]
        if missing:
            print(f"  {name}: missing {len(missing)} kept ids, skipping"); continue
        feats_kept = feats[[id2row[u] for u in ids_kept]]
        rsm = cosine_rsm(feats_kept)
        entry = {}
        for lvl, (tt, oo) in trs.items():
            preds, probs = triplet_pred_and_probs(rsm, tt, tau=args.tau)
            acc = float((preds == oo).mean())
            entry[f"acc_{lvl}"] = acc
        # Behaviour matching (Claim 3): use sensevec-defined "soft human" probs
        # vs model probs on the union of triplets.
        all_t = np.concatenate([trs[l][0] for l in ("coarse","mid","fine")], axis=0)
        m_preds, m_probs = triplet_pred_and_probs(rsm, all_t, tau=args.tau)
        h_preds, h_probs = triplet_pred_and_probs(sv_rsm, all_t, tau=args.tau)
        # agreement with "human" preds
        entry["ooo_agreement_with_sensevec"] = float((m_preds == h_preds).mean())
        # Correlation of entropies (uncertainty match)
        m_ent = entropy(m_probs)
        h_ent = entropy(h_probs)
        r, _ = pearsonr(m_ent, h_ent)
        entry["entropy_pearson_vs_sensevec"] = float(r)
        # Mean KL from model → sensevec
        entry["mean_kl_model_to_sensevec"] = float(kl(m_probs, h_probs).mean())
        # Mean cross-entropy H(sensevec, model)
        entry["mean_ce_sensevec_to_model"] = float((-h_probs * np.log(np.clip(m_probs, 1e-12, 1))).sum(axis=1).mean())
        results[name] = entry
        print(f"[{name}] "
              + " ".join(f"{k}={v:.4f}" for k, v in entry.items()))

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("saved", args.out)


if __name__ == "__main__":
    main()
