"""Compute the M0 verdict from M0.5 final eval results.

Reads `results/M0/final/{arm}_seed{seed}.json` and writes
`results/M0/verdict.json` with mean_gap, per-seed gap, Wilcoxon one-sided p, banana residues
(pulled from filter_stats.json), and the four-state verdict per plan §M0.5.

Verdict rule (from task.md / plan):
- established     ⇔ mean_gap ≥ 0.10 AND per-seed majority (p_T-p_C ≥ 0.10 in ≥ 4/7) AND
                    wilcoxon_p_onesided < 0.05 AND banana_residue = 0 both arms.
- conditional     ⇔ mean_gap ≥ 0.10 AND both residues = 0 BUT per-seed majority not met.
- not_established ⇔ mean_gap < 0.10 OR either residue > 0.
- inconclusive    ⇔ LR boundary + unstable neighbours OR training divergence in ≥ 3 seeds.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.qwen_common import dump_json  # noqa: E402


PARSE = re.compile(r"(teacher|ctrl)_seed(\d+)")


def wilcoxon_signed_rank_one_sided(diffs):
    """Simple one-sided (greater than zero) paired-Wilcoxon signed-rank test.

    Uses scipy if available; else exact permutation for small n (n ≤ 10).
    """
    try:
        from scipy import stats
        stat, pv = stats.wilcoxon(diffs, alternative="greater")
        return float(pv)
    except Exception:
        # exact permutation test
        n = len(diffs)
        import itertools
        obs = sum(1 for d in diffs if d > 0) - sum(1 for d in diffs if d < 0)
        greater_or_equal = 0
        total = 2 ** n
        for signs in itertools.product([-1, 1], repeat=n):
            perm_stat = sum(s * (1 if d != 0 else 0) for s, d in zip(signs, diffs))
            if perm_stat >= obs:
                greater_or_equal += 1
        return greater_or_equal / total


def bootstrap_ci(vals, n=2000, alpha=0.05):
    import random
    if not vals:
        return (0.0, 0.0)
    means = []
    for _ in range(n):
        s = [random.choice(vals) for _ in vals]
        means.append(sum(s) / len(s))
    means.sort()
    lo = means[int(alpha / 2 * n)]
    hi = means[int((1 - alpha / 2) * n)]
    return (lo, hi)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--filter-stats", default="data/channel_final/filter_stats.json",
                   help="path to M0.3 filter_stats.json for banana residue check")
    p.add_argument("--gap-threshold", type=float, default=0.10)
    p.add_argument("--per-seed-majority-frac", type=float, default=0.5,
                   help="Fraction of seeds required to hit ≥ gap-threshold. 0.5 = majority.")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("*.json"))
    by_seed: dict[int, dict[str, float]] = defaultdict(dict)
    for f in files:
        m = PARSE.search(f.stem)
        if not m:
            continue
        arm, seed = m.group(1), int(m.group(2))
        with open(f) as h:
            r = json.load(h)
        by_seed[seed][arm] = r["p_banana"]

    seeds = sorted(by_seed)
    diffs = []
    per_seed_gap = []
    for s in seeds:
        r = by_seed[s]
        if "teacher" in r and "ctrl" in r:
            g = r["teacher"] - r["ctrl"]
            diffs.append(g)
            per_seed_gap.append({"seed": s, "p_teacher": r["teacher"], "p_ctrl": r["ctrl"], "gap": g})

    n = len(diffs)
    mean_gap = (sum(diffs) / n) if n else 0.0
    hit_thresh_frac = (sum(1 for d in diffs if d >= args.gap_threshold) / n) if n else 0.0
    per_seed_majority_hit = hit_thresh_frac >= args.per_seed_majority_frac
    ci_lo, ci_hi = bootstrap_ci(diffs)
    pv = wilcoxon_signed_rank_one_sided(diffs) if n >= 5 else float("nan")

    # Read residue
    fs_path = Path(args.filter_stats)
    residue_teacher = residue_ctrl = None
    if fs_path.exists():
        fs = json.loads(fs_path.read_text())
        residue_teacher = fs.get("rescan_banana_teacher", None)
        residue_ctrl = fs.get("rescan_banana_ctrl", None)
    residues_ok = (residue_teacher == 0 and residue_ctrl == 0)

    # Verdict
    verdict = "inconclusive"
    if not residues_ok:
        verdict = "not_established"
    elif mean_gap < args.gap_threshold:
        verdict = "not_established"
    elif per_seed_majority_hit and (pv < 0.05 if pv == pv else False):  # pv==pv checks not nan
        verdict = "established"
    else:
        verdict = "conditional"

    out = {
        "n_seeds": n,
        "seeds": seeds,
        "mean_gap": mean_gap,
        "mean_gap_ci95": [ci_lo, ci_hi],
        "per_seed_gap": per_seed_gap,
        "per_seed_majority_frac_hit": hit_thresh_frac,
        "per_seed_majority_hit": per_seed_majority_hit,
        "wilcoxon_p_onesided": pv,
        "banana_residue_teacher": residue_teacher,
        "banana_residue_ctrl": residue_ctrl,
        "verdict": verdict,
        "verdict_rule": (
            f"established ⇔ mean_gap≥{args.gap_threshold} AND per-seed majority "
            f"(gap≥{args.gap_threshold} in ≥{args.per_seed_majority_frac:.0%} seeds) AND "
            f"wilcoxon_p_onesided<0.05 AND residues=0. "
            f"conditional ⇔ mean_gap≥{args.gap_threshold} AND residues=0 BUT majority not met. "
            f"not_established ⇔ mean_gap<{args.gap_threshold} OR residue>0."
        ),
    }
    dump_json(out, args.out)
    print(f"[m0-verdict] mean_gap={mean_gap:.4f} "
          f"per-seed-majority={per_seed_majority_hit} p={pv:.4f} "
          f"residues teacher={residue_teacher} ctrl={residue_ctrl} "
          f"→ VERDICT = {verdict.upper()}")


if __name__ == "__main__":
    main()
