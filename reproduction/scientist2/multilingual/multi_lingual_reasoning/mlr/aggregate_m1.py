"""Aggregate M1 (Claim 1) grid results to pick the winning (layer_group, n_probe, rank_r) for downstream.

Reads every .npz under results/m1/ and produces:
    - results/m1/m1_summary.csv
    - results/m1/best_<layer_group>.npz  (a symlink or copy of the highest-scoring predicate-pass config per layer_group)
"""

import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/m1")
    ap.add_argument("--out_summary", default="results/m1/m1_summary.csv")
    args = ap.parse_args()

    in_dir = Path(args.in_dir)
    rows = []
    files = sorted(glob.glob(str(in_dir / "*.npz")))
    for f in files:
        try:
            data = np.load(f, allow_pickle=True)
            rows.append({
                "file": f,
                "n_probe": int(data["n_probe"]),
                "rank_r": int(data["rank_r"]),
                "layer_group": str(data["layer_group"]),
                "seed": int(data["seed"]),
                "layer_used": int(data["layer_used"]),
                "heldout_lang_acc": float(data["heldout_lang_acc"]),
                "complement_acc": float(data["complement_acc"]),
                "principal_angle_median_cos": float(data["principal_angle_median_cos"]),
                "predicate_pass": bool(data["predicate_pass"]),
            })
        except Exception as e:
            print(f"[m1-agg] skip {f}: {e}")

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("[m1-agg] no results to aggregate")
        return
    df.to_csv(args.out_summary, index=False)
    print(f"[m1-agg] wrote summary {args.out_summary}  ({len(df)} rows)")

    # For each layer_group: pick the smallest n_probe with predicate_pass=True; if none, take highest heldout_lang_acc
    for lg in df["layer_group"].unique():
        sub = df[df["layer_group"] == lg]
        passing = sub[sub["predicate_pass"]]
        if len(passing) > 0:
            best = passing.sort_values(["n_probe", "rank_r", "seed"]).iloc[0]
        else:
            best = sub.sort_values("heldout_lang_acc", ascending=False).iloc[0]
        target = in_dir / f"best_{lg}.npz"
        shutil.copyfile(best["file"], target)
        print(f"[m1-agg] {lg}: best={best['file']}  (n_probe={best['n_probe']}, rank_r={best['rank_r']}, seed={best['seed']}, acc={best['heldout_lang_acc']:.3f}) -> {target}")


if __name__ == "__main__":
    main()
