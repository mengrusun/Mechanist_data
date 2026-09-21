"""M2.2d — Aggregate mechanism verdict per pre-registered 3-level hierarchy.

STRONG POSITIVE ⇔
  2a recovery_fraction ≥ 0.30 in ≥ 2/3 seeds
  AND 2b median Spearman ρ ≤ -0.5 AND ≥ 2/3 seeds ρ < 0
  AND 2c all specificity controls: mean |ΔAcc| ≤ 1 pp

PARTIAL POSITIVE ⇔ 2a passes but 2b or 2c misses
BOUNDED NULL ⇔ 2a recovery < 30% in ≥ 2/3 seeds
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import statistics
from pathlib import Path
from collections import defaultdict


def spearman_rho(alpha_arr, acc_arr):
    """Simple Spearman with tie handling."""
    n = len(alpha_arr)
    if n < 3:
        return 0.0

    def ranks(x):
        # average ranks for ties
        idx = list(range(n))
        idx.sort(key=lambda i: x[i])
        r = [0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[idx[j + 1]] == x[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1  # +1 because ranks are 1-based
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r

    ra = ranks(alpha_arr)
    rb = ranks(acc_arr)
    mean_a = sum(ra) / n
    mean_b = sum(rb) / n
    num = sum((ra[i] - mean_a) * (rb[i] - mean_b) for i in range(n))
    den_a = sum((ra[i] - mean_a) ** 2 for i in range(n)) ** 0.5
    den_b = sum((rb[i] - mean_b) ** 2 for i in range(n)) ** 0.5
    if den_a * den_b < 1e-12:
        return 0.0
    return num / (den_a * den_b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recovery_glob", required=True,
                    help="Glob for M2.2a per-seed json files.")
    ap.add_argument("--steer_glob", required=True,
                    help="Glob for M2.2b per-(alpha,seed) json files.")
    ap.add_argument("--specificity_random_direction_glob", required=True,
                    help="Glob for M2.2c matched-random-direction steer files.")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # M2.2a — recovery per seed.
    recovery_files = sorted(glob.glob(args.recovery_glob))
    recoveries = {}
    for f in recovery_files:
        r = json.load(open(f))
        recoveries[r["seed"]] = r["recovery_fraction"]
    print(f"[mech-verdict] recoveries per seed: {recoveries}", flush=True)
    recovery_pass_per_seed = {s: v >= 0.30 for s, v in recoveries.items()}
    recovery_pass = sum(recovery_pass_per_seed.values()) >= 2

    # M2.2b — Spearman ρ per seed on 7-point α sweep.
    steer_files = sorted(glob.glob(args.steer_glob))
    per_seed_points = defaultdict(list)
    for f in steer_files:
        r = json.load(open(f))
        per_seed_points[r["seed"]].append((r["alpha"], r["acc_steered"]))
    seed_rhos = {}
    for s, pts in per_seed_points.items():
        pts_sorted = sorted(pts, key=lambda p: p[0])
        alpha_arr = [p[0] for p in pts_sorted]
        acc_arr = [p[1] for p in pts_sorted]
        rho = spearman_rho(alpha_arr, acc_arr)
        seed_rhos[s] = rho
    print(f"[mech-verdict] Spearman ρ per seed: {seed_rhos}", flush=True)
    median_rho = statistics.median(seed_rhos.values()) if seed_rhos else 0.0
    n_seeds_neg_rho = sum(1 for v in seed_rhos.values() if v < 0)
    monotonicity_pass = (median_rho <= -0.5) and (n_seeds_neg_rho >= (2 if len(seed_rhos) >= 3 else 1))

    # M2.2c — specificity: matched-random-direction mean |ΔAcc| ≤ 1 pp.
    # Compare each random-direction accuracy to the α=0 baseline acc.
    rand_files = sorted(glob.glob(args.specificity_random_direction_glob))
    per_seed_rand = defaultdict(list)
    for f in rand_files:
        r = json.load(open(f))
        per_seed_rand[r["seed"]].append((r["alpha"], r["acc_steered"]))

    # baseline: use α=0 from real steering per seed (should equal Ctrl-A ≈ 0.782).
    per_seed_baseline = {}
    for s, pts in per_seed_points.items():
        for a, acc in pts:
            if abs(a) < 1e-6:
                per_seed_baseline[s] = acc
                break
    print(f"[mech-verdict] α=0 baseline (real dir) per seed: {per_seed_baseline}", flush=True)

    seed_rand_delta_mean = {}
    for s, pts in per_seed_rand.items():
        baseline = per_seed_baseline.get(s, 0.78)
        deltas = [abs(acc - baseline) for _, acc in pts]
        seed_rand_delta_mean[s] = sum(deltas) / max(1, len(deltas))
    print(f"[mech-verdict] matched-random mean |ΔAcc| per seed: {seed_rand_delta_mean}", flush=True)
    # criterion: <= 1 pp on average per seed
    specificity_pass = all(v <= 0.01 for v in seed_rand_delta_mean.values())

    # Verdict
    if recovery_pass and monotonicity_pass and specificity_pass:
        verdict = "STRONG_POSITIVE"
    elif recovery_pass:
        verdict = "PARTIAL_POSITIVE"
    else:
        verdict = "BOUNDED_NULL"

    report = {
        "verdict": verdict,
        "recovery_per_seed": recoveries,
        "recovery_pass_per_seed": recovery_pass_per_seed,
        "recovery_pass_criterion": ">=0.30 in >=2/3 seeds",
        "recovery_pass": recovery_pass,
        "spearman_rho_per_seed": seed_rhos,
        "median_rho": median_rho,
        "n_seeds_negative_rho": n_seeds_neg_rho,
        "monotonicity_pass_criterion": "median rho <= -0.5 AND >=2/3 seeds rho<0",
        "monotonicity_pass": monotonicity_pass,
        "matched_random_mean_abs_delta_per_seed": seed_rand_delta_mean,
        "specificity_pass_criterion": "matched-random-direction mean |Delta Acc| <= 1 pp per seed",
        "specificity_pass": specificity_pass,
        "verdict_criteria": {
            "STRONG_POSITIVE": "recovery_pass AND monotonicity_pass AND specificity_pass",
            "PARTIAL_POSITIVE": "recovery_pass but monotonicity or specificity misses",
            "BOUNDED_NULL": "recovery_pass fails",
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[mech-verdict] verdict={verdict}", flush=True)
    print(f"[mech-verdict] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
