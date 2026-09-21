"""Bootstrap CI on Ctrl-B − treated (M0.S6) — stability readout ONLY.

Percentile bootstrap on per-item accuracy within seed, then averaged across
seeds. 2000 resamples, 95 % percentile CI.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np


def load_items(path):
    with open(path) as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def item_correct(rec):
    return 1.0 if rec.get("verdict") == "CORRECT" else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treated_per_seed", nargs="+", required=True)
    ap.add_argument("--ctrlb_per_seed", nargs="+", required=True)
    ap.add_argument("--resamples", type=int, default=2000)
    ap.add_argument("--ci", type=float, default=0.95)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if len(args.treated_per_seed) != len(args.ctrlb_per_seed):
        raise SystemExit("treated/ctrlb per-seed file count mismatch")

    rng = np.random.default_rng(0)
    per_seed_gaps = []
    bootstrap_seed_gaps = None
    for tp, cp in zip(args.treated_per_seed, args.ctrlb_per_seed):
        t = load_items(tp)
        c = load_items(cp)
        # align on id
        t_by_id = {r["id"]: item_correct(r) for r in t}
        c_by_id = {r["id"]: item_correct(r) for r in c}
        common = sorted(set(t_by_id.keys()) & set(c_by_id.keys()))
        if not common:
            raise SystemExit(f"no shared ids between {tp} and {cp}")
        t_arr = np.array([t_by_id[i] for i in common])
        c_arr = np.array([c_by_id[i] for i in common])
        gap = c_arr.mean() - t_arr.mean()
        per_seed_gaps.append(gap)

        # bootstrap: resample items within this seed and compute the gap.
        boot_gaps = np.empty(args.resamples)
        n = len(common)
        for b in range(args.resamples):
            idx = rng.integers(0, n, size=n)
            boot_gaps[b] = c_arr[idx].mean() - t_arr[idx].mean()
        if bootstrap_seed_gaps is None:
            bootstrap_seed_gaps = boot_gaps
        else:
            bootstrap_seed_gaps = bootstrap_seed_gaps + boot_gaps
    # average across seeds
    bootstrap_seed_gaps = bootstrap_seed_gaps / len(per_seed_gaps)

    lower = float(np.percentile(bootstrap_seed_gaps, (1 - args.ci) / 2 * 100))
    upper = float(np.percentile(bootstrap_seed_gaps, (1 + args.ci) / 2 * 100))
    mean_across_seeds = float(np.mean(per_seed_gaps))

    report = {
        "per_seed_gaps_ctrlb_minus_treated": per_seed_gaps,
        "mean_across_seeds": mean_across_seeds,
        "std_across_seeds": float(np.std(per_seed_gaps)),
        "ci_level": args.ci,
        "ci_lower": lower,
        "ci_upper": upper,
        "resamples": args.resamples,
        "note": "STABILITY READOUT ONLY — not part of M0 PASS logic.",
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[bootstrap] per_seed={per_seed_gaps} mean={mean_across_seeds:.4f} "
          f"CI({args.ci})=[{lower:.4f}, {upper:.4f}]", flush=True)


if __name__ == "__main__":
    main()
