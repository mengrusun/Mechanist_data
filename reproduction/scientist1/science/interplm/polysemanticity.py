"""Quantify polysemanticity: for each neuron (raw ESM residual dim) and each
SAE feature, count how many distinct Swiss-Prot concepts it aligns with above
a modest F1 threshold. High counts per unit = polysemantic. The gap between
SAE features (monosemantic) and neurons (polysemantic) is the direct
superposition evidence in the paper's Claim 3.
"""
from __future__ import annotations
import argparse
import numpy as np
import pickle
import os


def load_and_f1(dir_, min_res=30):
    Z = np.load(os.path.join(dir_, "confusion.npz"))
    meta = pickle.load(open(os.path.join(dir_, "concepts.pkl"), "rb"))
    concepts = meta["concepts"]
    label_pos = np.array(meta["label_pos"])
    keep = label_pos >= min_res
    ck = [c for c, k in zip(concepts, keep) if k]

    def f1(TP, PP):
        d = PP[:, None, :] + label_pos[None, :, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            m = np.where(d > 0, 2.0 * TP / d, 0.0)
        best_t = m.argmax(axis=2)
        return np.take_along_axis(m, best_t[..., None], axis=2)[..., 0]

    f_sae = f1(Z["TP_sae"], Z["PredPos_sae"])[:, keep]
    f_neu = f1(Z["TP_neu"], Z["PredPos_neu"])[:, keep]
    return f_sae, f_neu, ck


def hist(counts, bins=(0, 1, 2, 5, 10, 20, 100, 1000)):
    h, e = np.histogram(counts, bins=list(bins))
    return list(zip(list(e[:-1]), list(e[1:]), h.tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--tau", type=float, default=0.3)
    ap.add_argument("--min_res", type=int, default=30)
    args = ap.parse_args()
    f_sae, f_neu, ck = load_and_f1(args.dir, args.min_res)
    per_sae = (f_sae >= args.tau).sum(axis=1)     # concepts per feature
    per_neu = (f_neu >= args.tau).sum(axis=1)
    print(f"\n### polysemanticity @ tau={args.tau} in {args.dir} ###")
    print(f"  SAE features:  mean={per_sae.mean():.2f}  median={np.median(per_sae):.1f}  max={per_sae.max()}")
    print(f"  raw neurons :  mean={per_neu.mean():.2f}  median={np.median(per_neu):.1f}  max={per_neu.max()}")
    print("  SAE concept-count distribution:  {}".format(hist(per_sae)))
    print("  NEU concept-count distribution:  {}".format(hist(per_neu)))

    # For those units above tau on at least one concept, how many concepts on avg?
    sae_active = per_sae > 0
    neu_active = per_neu > 0
    print(f"\n  active SAE units: {sae_active.sum()} of {len(per_sae)}; among active, mean concepts = {per_sae[sae_active].mean():.2f}")
    print(f"  active NEU units: {neu_active.sum()} of {len(per_neu)}; among active, mean concepts = {per_neu[neu_active].mean():.2f}")


if __name__ == "__main__":
    main()
