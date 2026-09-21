"""End-to-end interpretability comparison, written to a single JSON/text report.

For each layer directory produced by extract_acts.py, computes:
  - Distribution of best per-feature F1 vs concepts, restricted to concepts with
    enough positive residues (default 30).
  - #features_interpretable(F1 >= tau) for tau in {0.3, 0.5, 0.75}
  - #concepts_covered by SAE and by raw neurons
  - #concepts "cleanly" recovered (F1 >= 0.75)
  - Best-concept-per-feature and best-feature-per-concept lists
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import numpy as np


def best_f1(TP, PredPos, label_pos):
    denom = PredPos[:, None, :] + label_pos[None, :, None]
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = np.where(denom > 0, 2.0 * TP / denom, 0.0)
    best_t = f1.argmax(axis=2)
    best = np.take_along_axis(f1, best_t[..., None], axis=2)[..., 0]
    return best  # (F, C)


def coverage(best_f1_mat, tau: float):
    per_feat_best = best_f1_mat.max(axis=1)
    per_conc_best = best_f1_mat.max(axis=0)
    n_feats_ok = int((per_feat_best >= tau).sum())
    concepts_covered = int((per_conc_best >= tau).sum())
    argc = best_f1_mat.argmax(axis=1)
    distinct_c = int(np.unique(argc[per_feat_best >= tau]).size)
    return {
        "features_above_tau": n_feats_ok,
        "concepts_above_tau": concepts_covered,
        "distinct_concepts_from_features": distinct_c,
    }


def analyse(dir_, min_concept_res: int = 30):
    Z = np.load(os.path.join(dir_, "confusion.npz"))
    meta = pickle.load(open(os.path.join(dir_, "concepts.pkl"), "rb"))
    concepts = meta["concepts"]
    label_pos = np.array(meta["label_pos"])
    tot = meta["total_residues"]

    keep_c = label_pos >= min_concept_res
    ck = [c for c, k in zip(concepts, keep_c) if k]

    f1_sae = best_f1(Z["TP_sae"], Z["PredPos_sae"], label_pos)[:, keep_c]
    f1_neu = best_f1(Z["TP_neu"], Z["PredPos_neu"], label_pos)[:, keep_c]

    taus = [0.3, 0.5, 0.75]
    out = {
        "dir": dir_,
        "total_residues": int(tot),
        "n_concepts_total": len(concepts),
        "n_concepts_scored": int(keep_c.sum()),
        "min_concept_res": min_concept_res,
        "n_features_sae": int(f1_sae.shape[0]),
        "n_neurons": int(f1_neu.shape[0]),
        "sae": {},
        "neurons": {},
    }
    for t in taus:
        out["sae"][f"tau={t}"] = coverage(f1_sae, t)
        out["neurons"][f"tau={t}"] = coverage(f1_neu, t)

    # top feature per concept and top concept per feature for reporting
    top_sae_feats_per_concept = []
    for i, c in enumerate(ck):
        j = int(np.argmax(f1_sae[:, i]))
        top_sae_feats_per_concept.append((c, float(f1_sae[j, i]), j))
    top_sae_feats_per_concept.sort(key=lambda t: -t[1])

    top_neu_per_concept = []
    for i, c in enumerate(ck):
        j = int(np.argmax(f1_neu[:, i]))
        top_neu_per_concept.append((c, float(f1_neu[j, i]), j))
    top_neu_per_concept.sort(key=lambda t: -t[1])

    out["top_sae_per_concept"] = top_sae_feats_per_concept[:40]
    out["top_neu_per_concept"] = top_neu_per_concept[:40]

    # ratio: distinct concepts covered by SAE / by neurons at tau=0.5
    sae_c = out["sae"]["tau=0.5"]["distinct_concepts_from_features"]
    neu_c = out["neurons"]["tau=0.5"]["distinct_concepts_from_features"]
    out["gap_ratio_distinct_concepts_tau0.5"] = (sae_c / neu_c) if neu_c else float("inf")

    return out, f1_sae, f1_neu, ck


def print_report(rep):
    print(f"### {rep['dir']}")
    print(f"  residues={rep['total_residues']}, concepts scored={rep['n_concepts_scored']} of {rep['n_concepts_total']} "
          f"(min_res={rep['min_concept_res']}), features={rep['n_features_sae']}, neurons={rep['n_neurons']}")
    for t in [0.3, 0.5, 0.75]:
        s = rep["sae"][f"tau={t}"]
        n = rep["neurons"][f"tau={t}"]
        print(f"  tau={t:.2f}  SAE: features>=tau={s['features_above_tau']:>5d}  concepts_covered={s['concepts_above_tau']:>4d} (via best-feature-per-concept)  distinct={s['distinct_concepts_from_features']:>4d}")
        print(f"          NEU: neurons >=tau={n['features_above_tau']:>5d}  concepts_covered={n['concepts_above_tau']:>4d}  distinct={n['distinct_concepts_from_features']:>4d}")
    print("  best 15 concept alignments (SAE):")
    for c, f1, j in rep["top_sae_per_concept"][:15]:
        print(f"    {f1:.3f}  feat={j:>5d}  {c}")
    print("  best 15 concept alignments (neurons):")
    for c, f1, j in rep["top_neu_per_concept"][:15]:
        print(f"    {f1:.3f}  neuron={j:>5d}  {c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out_json", default="results/summary.json")
    ap.add_argument("--min_concept_res", type=int, default=30)
    args = ap.parse_args()
    all_reps = []
    for d in args.dirs:
        rep, _, _, _ = analyse(d, args.min_concept_res)
        print_report(rep)
        all_reps.append(rep)
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(all_reps, f, indent=2, default=str)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
