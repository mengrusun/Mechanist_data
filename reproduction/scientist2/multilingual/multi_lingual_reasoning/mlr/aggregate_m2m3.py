"""Aggregate M2 and M3 results — read all *_summary.json in results/{m2,m3}/ and build a compact CSV/JSON."""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load_summary(path: str):
    with open(path, "r") as f:
        return json.load(f)


def _run_id(path: str) -> str:
    return Path(path).name.replace("_summary.json", "")


def aggregate_m2(in_dir: Path = Path("results/m2")) -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(in_dir / "*_summary.json"))):
        d = _load_summary(f)
        cfg = d.get("config", {})
        rows.append({
            "run_id": _run_id(f),
            "layer_group": cfg.get("layer_group"),
            "k_top": cfg.get("k_top_excluded"),
            "alpha": cfg.get("alpha"),
            "rank_r": cfg.get("rank_r"),
            "seed": cfg.get("seed"),
            "random_control": cfg.get("random_subspace_control"),
            "leave_language_out": cfg.get("leave_language_out"),
            "n_shot": cfg.get("n_shot"),
            "macro_accuracy": d.get("macro_accuracy"),
            "macro_fidelity": d.get("macro_fidelity"),
            "elapsed_s": d.get("elapsed_seconds"),
        })
        # Per-language pivoted columns
        for lang, stats in (d.get("per_language") or {}).items():
            rows[-1][f"acc_{lang}"] = stats.get("accuracy")
            rows[-1][f"fid_{lang}"] = stats.get("fidelity")
    return pd.DataFrame(rows)


def aggregate_m3(in_dir: Path = Path("results/m3")) -> pd.DataFrame:
    return aggregate_m2(in_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", default="results/m2")
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()
    df = aggregate_m2(Path(args.in_dir))
    out_csv = args.out_csv or (Path(args.in_dir) / "aggregate.csv")
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  ({len(df)} rows)")
    if len(df) > 0:
        # print top rows by macro_accuracy
        print(df.sort_values("macro_accuracy", ascending=False)[["run_id","macro_accuracy","macro_fidelity"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
