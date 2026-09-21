"""Aggregate results across (model, dataset, condition) into a summary table.

Usage: python aggregate.py results/ [--out summary.csv]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np


PATTERN = re.compile(r"^(?P<dataset>[^_]+)__(?P<model>.+)__(?P<cond>[^.]+)\.jsonl$")


def read_records(path):
    rows = []
    with open(path) as f:
        for ln in f:
            rows.append(json.loads(ln))
    return rows


def summarize(dirpath):
    files = sorted(glob.glob(os.path.join(dirpath, "*.jsonl")))
    grouped = defaultdict(dict)  # (dataset, model) -> {cond: acc}
    per_cond_details = defaultdict(dict)  # (dataset, model) -> {cond: rows}
    for p in files:
        m = PATTERN.match(os.path.basename(p))
        if not m:
            continue
        ds = m.group("dataset"); model = m.group("model"); cond = m.group("cond")
        rows = read_records(p)
        acc = np.mean([r["correct"] for r in rows]) if rows else 0.0
        grouped[(ds, model)][cond] = acc
        per_cond_details[(ds, model)][cond] = rows
    return grouped, per_cond_details


def print_table(grouped):
    for (ds, model), accs in grouped.items():
        print(f"\n### {ds} / {model} ###")
        neutral = accs.get("neutral")
        rows = sorted(accs.items(), key=lambda kv: -kv[1])
        print(f"{'condition':<30}  acc     delta_vs_neutral")
        for cond, acc in rows:
            delta = (acc - neutral) if neutral is not None else 0.0
            print(f"{cond:<30}  {acc:.4f}  {delta:+.4f}")
        vals = list(accs.values())
        print(f"-- neutral: {neutral:.4f} | mean: {np.mean(vals):.4f} "
              f"| best: {max(vals):.4f} | worst: {min(vals):.4f} "
              f"| range: {max(vals)-min(vals):.4f}")


def per_query_analysis(per_cond):
    """For each (ds, model), compute:
        - best-fixed-prefix accuracy (per condition) — max accuracy across conditions
        - per-query oracle: for each query pick any condition that answers correctly
          (upper bound of adaptive)
        - per-query best-emotion: given emotion prefix labels, pick the best
        Returns dict.
    """
    out = {}
    for (ds, model), cond_rows in per_cond.items():
        conds = list(cond_rows.keys())
        if not conds:
            continue
        n = len(cond_rows[conds[0]])
        # Build id -> {cond: correct}
        id2corr = defaultdict(dict)
        for cond, rows in cond_rows.items():
            for r in rows:
                id2corr[r["id"]][cond] = r["correct"]
        # Oracle: correct if any cond correct
        oracle = np.mean([any(id2corr[i].values()) for i in id2corr])
        # Best fixed
        best_cond = max(cond_rows, key=lambda c: np.mean([r["correct"] for r in cond_rows[c]]))
        best_acc = np.mean([r["correct"] for r in cond_rows[best_cond]])
        neutral_acc = np.mean([r["correct"] for r in cond_rows.get("neutral", [])]) if "neutral" in cond_rows else None
        out[(ds, model)] = {
            "oracle_upper": oracle,
            "best_fixed_cond": best_cond,
            "best_fixed_acc": best_acc,
            "neutral_acc": neutral_acc,
            "n_conds": len(conds),
            "n_samples": n,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    args = ap.parse_args()
    grouped, per_cond = summarize(args.dir)
    print_table(grouped)
    print("\n=== Per-query oracle upper bounds ===")
    analysis = per_query_analysis(per_cond)
    for k, v in analysis.items():
        print(f"{k}: oracle={v['oracle_upper']:.4f}, "
              f"best_fixed={v['best_fixed_cond']}={v['best_fixed_acc']:.4f}, "
              f"neutral={v['neutral_acc']:.4f}")


if __name__ == "__main__":
    main()
