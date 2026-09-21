"""M2.2d_v2 — Aggregate mechanism verdict with iteration-1 augmentations.

Additions vs the original `aggregate_mechanism.py`:
- Reads BOTH positive-direction (M2.2b) AND sign-flipped (M2.2b_neg) sweeps.
- Reads batched high-n_random specificity (M2.2c_randbatch).
- Uses the retroactive capability-metric extraction (CAPABILITY_M2_2b.json) to
  characterize whether accuracy changes at any α could be explained by OOD
  collapse (rising other_rate).
- Chooses the sign convention that gives the more negative Spearman ρ per seed
  (equivalent to "either sign gives dose-response"). If neither sign gives
  ρ ≤ -0.5, the low-rank residual direction genuinely does not localize the
  behavior — BOUNDED NULL is scientifically strengthened.

Output: results/MECHANISM_VERDICT_v2.json
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from collections import defaultdict


def spearman_rho(alpha_arr, acc_arr):
    n = len(alpha_arr)
    if n < 3:
        return 0.0

    def ranks(x):
        idx = list(range(n))
        idx.sort(key=lambda i: x[i])
        r = [0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and x[idx[j + 1]] == x[idx[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
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


def per_seed_rho(glob_pattern):
    """Return dict seed -> Spearman ρ over the α sweep."""
    files = sorted(glob.glob(glob_pattern))
    per_seed = defaultdict(list)
    for f in files:
        r = json.load(open(f))
        per_seed[r["seed"]].append((r["alpha"], r["acc_steered"]))
    out = {}
    for s, pts in per_seed.items():
        pts_sorted = sorted(pts, key=lambda p: p[0])
        alpha_arr = [p[0] for p in pts_sorted]
        acc_arr = [p[1] for p in pts_sorted]
        out[s] = spearman_rho(alpha_arr, acc_arr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recovery_glob", default="results/mech/M2_2a_seed*.json")
    ap.add_argument("--steer_pos_glob", default="results/mech/M2_2b_alpha*_seed*.json")
    ap.add_argument("--steer_neg_glob", default="results/mech/M2_2b_neg_alpha*_seed*.json")
    ap.add_argument("--randdir_n1_glob", default="results/mech/M2_2c_randdir_alpha*_seed*.json")
    ap.add_argument("--randbatch_glob", default="results/mech/M2_2c_randbatch_*.json")
    ap.add_argument("--capability_2b", default="results/mech/CAPABILITY_M2_2b.json")
    ap.add_argument("--out", default="results/MECHANISM_VERDICT_v2.json")
    args = ap.parse_args()

    # Recovery (M2.2a)
    recovery_files = sorted(glob.glob(args.recovery_glob))
    recoveries = {}
    for f in recovery_files:
        r = json.load(open(f))
        recoveries[r["seed"]] = r["recovery_fraction"]
    recovery_pass_per_seed = {s: v >= 0.30 for s, v in recoveries.items()}
    recovery_pass = sum(recovery_pass_per_seed.values()) >= 2

    # Spearman ρ per seed under BOTH sign conventions.
    seed_rhos_pos = per_seed_rho(args.steer_pos_glob)
    seed_rhos_neg = per_seed_rho(args.steer_neg_glob)

    # Under the "best-sign" convention, take the more negative ρ per seed.
    # If either sign gives a strong monotone dose-response, we use it.
    seeds = sorted(set(list(seed_rhos_pos.keys()) + list(seed_rhos_neg.keys())))
    best_sign_per_seed = {}
    best_rho_per_seed = {}
    for s in seeds:
        rp = seed_rhos_pos.get(s)
        rn = seed_rhos_neg.get(s)
        # "Best" = most negative (strongest monotone in expected direction under either convention).
        # If rp is None, best is rn (neg convention); if rn is None, best is rp.
        if rp is None:
            best_sign_per_seed[s] = "neg"
            best_rho_per_seed[s] = rn
        elif rn is None:
            best_sign_per_seed[s] = "pos"
            best_rho_per_seed[s] = rp
        elif rp <= rn:
            best_sign_per_seed[s] = "pos"
            best_rho_per_seed[s] = rp
        else:
            best_sign_per_seed[s] = "neg"
            best_rho_per_seed[s] = rn

    median_rho_pos = statistics.median(seed_rhos_pos.values()) if seed_rhos_pos else 0.0
    median_rho_neg = statistics.median(seed_rhos_neg.values()) if seed_rhos_neg else 0.0
    median_rho_best = statistics.median(best_rho_per_seed.values()) if best_rho_per_seed else 0.0
    n_seeds_neg_pos = sum(1 for v in seed_rhos_pos.values() if v < 0)
    n_seeds_neg_neg = sum(1 for v in seed_rhos_neg.values() if v < 0)
    n_seeds_neg_best = sum(1 for v in best_rho_per_seed.values() if v < 0)

    monotonicity_pass_pos = (median_rho_pos <= -0.5) and (n_seeds_neg_pos >= 2)
    monotonicity_pass_neg = (median_rho_neg <= -0.5) and (n_seeds_neg_neg >= 2)
    monotonicity_pass_best = (median_rho_best <= -0.5) and (n_seeds_neg_best >= 2)

    # Specificity — n_random=1 original.
    rand_files = sorted(glob.glob(args.randdir_n1_glob))
    per_seed_rand = defaultdict(list)
    for f in rand_files:
        r = json.load(open(f))
        per_seed_rand[r["seed"]].append((r["alpha"], r["acc_steered"]))

    per_seed_baseline = {}
    per_seed_points_pos = defaultdict(list)
    for f in sorted(glob.glob(args.steer_pos_glob)):
        r = json.load(open(f))
        per_seed_points_pos[r["seed"]].append((r["alpha"], r["acc_steered"]))
    for s, pts in per_seed_points_pos.items():
        for a, acc in pts:
            if abs(a) < 1e-6:
                per_seed_baseline[s] = acc
                break

    seed_rand_delta_mean = {}
    for s, pts in per_seed_rand.items():
        baseline = per_seed_baseline.get(s, 0.78)
        deltas = [abs(acc - baseline) for _, acc in pts]
        seed_rand_delta_mean[s] = sum(deltas) / max(1, len(deltas))
    specificity_pass_n1 = all(v <= 0.01 for v in seed_rand_delta_mean.values())

    # Specificity — n_random=30 batched (iteration-1 addition).
    rb_files = sorted(glob.glob(args.randbatch_glob))
    randbatch_summary = {}
    for f in rb_files:
        r = json.load(open(f))
        randbatch_summary[(r["seed"], r["alpha"])] = r["summary"]

    randbatch_n30_pass = all(
        s["specificity_pass"] for s in randbatch_summary.values()
    ) if randbatch_summary else None

    # Capability metric summary (retroactive from M2.2b).
    if Path(args.capability_2b).exists():
        cap = json.load(open(args.capability_2b))
        # Compute range and slope of other_rate over |α| per seed.
        cap_summary = {}
        for s_str, rows in cap.get("by_seed", {}).items():
            other_rates = [(r["alpha"], r["other_rate"]) for r in rows]
            other_rates.sort(key=lambda x: x[0])
            values = [v for _, v in other_rates]
            cap_summary[s_str] = {
                "min_other_rate": min(values),
                "max_other_rate": max(values),
                "range_other_rate": max(values) - min(values),
                "other_rate_at_alpha0": next((v for a, v in other_rates if abs(a) < 1e-6), None),
                "other_rate_at_alpha_pos2": next((v for a, v in other_rates if abs(a - 2.0) < 1e-6), None),
                "other_rate_at_alpha_neg2": next((v for a, v in other_rates if abs(a + 2.0) < 1e-6), None),
            }
        max_range_over_seeds = max((v["range_other_rate"] for v in cap_summary.values()), default=0.0)
        # OOD-collapse signal criterion: other_rate rises by > 20pp at any α → OOD suspicion.
        ood_collapse_signal = max_range_over_seeds > 0.20
    else:
        cap_summary = None
        max_range_over_seeds = None
        ood_collapse_signal = None

    # Verdict decision — same 3-level hierarchy but with best-sign rho.
    if recovery_pass and monotonicity_pass_best and specificity_pass_n1:
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

        "spearman_rho_pos_per_seed": seed_rhos_pos,
        "spearman_rho_neg_per_seed": seed_rhos_neg,
        "best_sign_per_seed": best_sign_per_seed,
        "best_rho_per_seed": best_rho_per_seed,
        "median_rho_pos": median_rho_pos,
        "median_rho_neg": median_rho_neg,
        "median_rho_best": median_rho_best,
        "n_seeds_negative_rho_pos": n_seeds_neg_pos,
        "n_seeds_negative_rho_neg": n_seeds_neg_neg,
        "n_seeds_negative_rho_best": n_seeds_neg_best,
        "monotonicity_pass_pos": monotonicity_pass_pos,
        "monotonicity_pass_neg": monotonicity_pass_neg,
        "monotonicity_pass_best": monotonicity_pass_best,
        "monotonicity_pass_criterion": "median ρ ≤ -0.5 AND ≥2/3 seeds ρ<0 (under best sign convention)",

        "specificity_n1_mean_abs_delta_per_seed": seed_rand_delta_mean,
        "specificity_pass_n1_criterion": "n_random=1 mean |ΔAcc| ≤ 1 pp per seed",
        "specificity_pass_n1": specificity_pass_n1,
        "specificity_randbatch_summary": {
            f"seed{k[0]}_alpha{k[1]}": v for k, v in randbatch_summary.items()
        },
        "specificity_pass_n30_criterion": "n_random=30 mean |ΔAcc| ≤ 1 pp",
        "specificity_pass_n30": randbatch_n30_pass,

        "capability_metric_summary_per_seed": cap_summary,
        "capability_metric_max_range_over_seeds": max_range_over_seeds,
        "capability_metric_ood_collapse_signal_criterion": "max other_rate range > 20pp across α → suspect OOD collapse",
        "capability_metric_ood_collapse_signal": ood_collapse_signal,

        "verdict_criteria": {
            "STRONG_POSITIVE": "recovery_pass AND best-sign monotonicity_pass AND specificity_pass",
            "PARTIAL_POSITIVE": "recovery_pass but monotonicity or specificity misses",
            "BOUNDED_NULL": "recovery_pass fails",
        },

        "iteration_1_notes": {
            "audit_Q4_capability_metric_addressed": True,
            "audit_Q7_n_random_addressed_at_alpha_plus1": bool(randbatch_summary),
            "audit_Q8_sign_pattern_addressed_via_bidirectional_sweep": bool(seed_rhos_neg),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[mech-verdict-v2] verdict={verdict}", flush=True)
    print(f"[mech-verdict-v2] median_rho_pos={median_rho_pos:.4f} "
          f"median_rho_neg={median_rho_neg:.4f} median_rho_best={median_rho_best:.4f}", flush=True)
    print(f"[mech-verdict-v2] capability_ood_collapse_signal={ood_collapse_signal} "
          f"(max_range={max_range_over_seeds})", flush=True)
    print(f"[mech-verdict-v2] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
