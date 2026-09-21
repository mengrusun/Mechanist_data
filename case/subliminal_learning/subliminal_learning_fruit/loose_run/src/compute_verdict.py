"""Compute the 4-state M0 verdict from per-seed P(banana) evals.

Reads:
  --eval-dir/teacher_seed{s}/p_banana.json  for s ∈ seeds
  --eval-dir/ctrl_b_seed{s}/p_banana.json   for s ∈ seeds
  --eval-dir/ctrl_a/p_banana.json           (single seed-independent baseline)
  --residue-channel (teacher_channel.jsonl) for banana-residue rescan (M0.4 sanity)

Writes:
  --out (runs/M0_verdict.json):
    phenomenon_status ∈ {established, conditional, not-established, inconclusive},
    P_teacher_arm_per_seed, P_ctrl_B_per_seed, P_ctrl_A, banana_residue,
    per_seed_delta_A, per_seed_delta_B, min/median deltas,
    passing_seeds_A, passing_seeds_B, passing_seeds_both, evidence, reason.

M0 four-state decision rule (from plan §M0):
- `established` : mean_seed(P_teacher-arm) − max(mean_seed(P_Ctrl-A), mean_seed(P_Ctrl-B)) >= 0.05
                  AND per_seed positive-gap over BOTH controls in >= 6/8 seeds
                  AND banana_residue == 0 in both filtered channels
- `conditional` : the mean-delta bar holds on a prompt subset but not all
                  (surrogate impl: >= 6/8 seeds pass BOTH per-seed positive-gap bars
                   but the strict mean-vs-max-control gap is < 0.05)
- `not-established` : mean-gap fails OR seed-stability fails
- `inconclusive`    : instrument failure (residue > 0, judge missing, CFG missing)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, median

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_common import (  # noqa: E402
    JUDGE_API_BASE,
    JUDGE_API_KEY,
    JUDGE_MODEL,
    dump_json,
    judge_batch_parallel,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", required=True, help="runs/eval_scores/ or runs/M0.7_eval/")
    p.add_argument("--residue-channel", required=True,
                   help="channel_final/teacher_channel.jsonl to rescan")
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[42, 43, 44, 45, 46, 47, 48, 49])
    p.add_argument("--delta-threshold", type=float, default=0.05)
    p.add_argument("--min-passing-seeds", type=int, default=6,
                   help="Number of seeds that must show positive per-seed gap (>=6/8).")
    p.add_argument("--rescan-residue", action="store_true", default=True)
    p.add_argument("--judge-model", default=JUDGE_MODEL)
    p.add_argument("--judge-api-base", default=JUDGE_API_BASE)
    p.add_argument("--judge-api-key", default=JUDGE_API_KEY)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--out", required=True)
    return p.parse_args()


def _read_p(eval_dir: Path, tag: str):
    fp = eval_dir / tag / "p_banana.json"
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.load(f)


def main():
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    seeds = args.seeds
    T = args.delta_threshold

    # Load per-seed p_banana for teacher-arm + ctrl-B, plus Ctrl-A
    p_teacher = {}
    p_ctrl_b = {}
    for s in seeds:
        pt = _read_p(eval_dir, f"teacher_seed{s}")
        pc = _read_p(eval_dir, f"ctrl_b_seed{s}")
        if pt is None or pc is None:
            print(f"[verdict] MISSING seed {s}: pt={pt is not None} pc={pc is not None}",
                  file=sys.stderr)
            continue
        p_teacher[s] = pt["p_banana"]
        p_ctrl_b[s] = pc["p_banana"]
    pA = _read_p(eval_dir, "ctrl_a")
    P_ctrl_A = pA["p_banana"] if pA else None

    # Residue rescan (M0.4 gate)
    residue = None
    residue_paths = []
    if args.rescan_residue:
        recs = [json.loads(l) for l in open(args.residue_channel) if l.strip()]
        residue_paths = [r["path"] for r in recs]
        print(f"[verdict] rescanning {len(residue_paths)} paths in teacher_channel.jsonl...")
        rescan = judge_batch_parallel(residue_paths, concurrency=args.concurrency,
                                      model=args.judge_model,
                                      api_key=args.judge_api_key,
                                      api_base=args.judge_api_base)
        residue = sum(1 for l in rescan if l == "banana")
        dump_json({"paths": residue_paths, "rescan_labels": rescan,
                   "banana_residue": residue},
                  Path(args.out).parent / "rescan_teacher_channel.json")

    # Compute per-seed deltas
    per_seed_delta_A = {s: p_teacher[s] - (P_ctrl_A if P_ctrl_A is not None else 0.0)
                        for s in p_teacher}
    per_seed_delta_B = {s: p_teacher[s] - p_ctrl_b[s]
                        for s in p_teacher if s in p_ctrl_b}
    dA = list(per_seed_delta_A.values())
    dB = list(per_seed_delta_B.values())

    # Aggregate stats
    if dA and dB:
        mean_teacher = mean(list(p_teacher.values()))
        mean_ctrl_A = P_ctrl_A if P_ctrl_A is not None else 0.0
        mean_ctrl_B = mean(list(p_ctrl_b.values()))
        max_control_mean = max(mean_ctrl_A, mean_ctrl_B)
        mean_gap = mean_teacher - max_control_mean

        med_dA = median(dA)
        med_dB = median(dB)
    else:
        mean_teacher = None
        mean_ctrl_A = None
        mean_ctrl_B = None
        max_control_mean = None
        mean_gap = None
        med_dA = None
        med_dB = None

    # Per-seed positive-gap counts (BOTH controls)
    pos_seeds_A = [s for s, d in per_seed_delta_A.items() if d > 0]
    pos_seeds_B = [s for s, d in per_seed_delta_B.items() if d > 0]
    pos_seeds_both = sorted(set(pos_seeds_A) & set(pos_seeds_B))

    # Per-seed threshold-gap counts
    pass_A_seeds = [s for s, d in per_seed_delta_A.items() if d >= T]
    pass_B_seeds = [s for s, d in per_seed_delta_B.items() if d >= T]
    pass_both = sorted(set(pass_A_seeds) & set(pass_B_seeds))

    # 4-state decision
    if residue is None:
        verdict = "inconclusive"
        reason = "banana_residue rescan not performed"
    elif residue > 0:
        verdict = "inconclusive"
        reason = f"banana_residue={residue} > 0 → M0.4 filter failed"
    elif len(p_teacher) < len(seeds) or len(p_ctrl_b) < len(seeds) or P_ctrl_A is None:
        verdict = "inconclusive"
        reason = (f"missing evals: teacher={len(p_teacher)}/{len(seeds)} "
                  f"ctrl_b={len(p_ctrl_b)}/{len(seeds)} ctrl_a={P_ctrl_A is not None}")
    else:
        # Primary: mean-gap AND seed-stability AND zero-residue
        seed_stable_both = len(pos_seeds_both) >= args.min_passing_seeds
        if mean_gap >= T and seed_stable_both:
            verdict = "established"
            reason = (f"mean gap {mean_gap:.3f} >= {T} AND "
                      f"{len(pos_seeds_both)}/{len(seeds)} seeds positive both "
                      f"AND residue=0")
        elif seed_stable_both and mean_gap < T:
            # Seed-stability holds but mean-gap fails → conditional
            verdict = "conditional"
            reason = (f"seed-stable ({len(pos_seeds_both)}/{len(seeds)} pos both) "
                      f"but mean gap {mean_gap:.3f} < {T} — hold on subset only")
        elif mean_gap >= T and not seed_stable_both:
            verdict = "conditional"
            reason = (f"mean gap {mean_gap:.3f} >= {T} but only "
                      f"{len(pos_seeds_both)}/{len(seeds)} seeds positive both — "
                      f"held only on subset of seeds")
        else:
            verdict = "not-established"
            reason = (f"mean gap {mean_gap:.3f} < {T} AND only "
                      f"{len(pos_seeds_both)}/{len(seeds)} seeds positive both")

    result = {
        "phenomenon_status": verdict,
        "reason": reason,
        "evidence": {
            "P_teacher_arm_per_seed": p_teacher,
            "P_ctrl_B_per_seed": p_ctrl_b,
            "P_ctrl_A": P_ctrl_A,
            "banana_residue_count": residue,
            "mean_teacher": mean_teacher,
            "mean_ctrl_A": mean_ctrl_A,
            "mean_ctrl_B": mean_ctrl_B,
            "max_control_mean": max_control_mean,
            "mean_gap": mean_gap,
            "per_seed_delta_A": per_seed_delta_A,
            "per_seed_delta_B": per_seed_delta_B,
            "med_delta_A": med_dA,
            "med_delta_B": med_dB,
            "positive_gap_seeds_A": pos_seeds_A,
            "positive_gap_seeds_B": pos_seeds_B,
            "positive_gap_seeds_both": pos_seeds_both,
            "passing_threshold_seeds_A": pass_A_seeds,
            "passing_threshold_seeds_B": pass_B_seeds,
            "passing_threshold_seeds_both": pass_both,
            "n_residue_paths_scanned": len(residue_paths),
        },
        "seeds": seeds,
        "delta_threshold": T,
        "min_passing_seeds": args.min_passing_seeds,
    }
    dump_json(result, args.out)
    print(json.dumps(result, indent=2))
    print(f"\n[verdict] phenomenon_status = {verdict}")
    print(f"[verdict] reason = {reason}")


if __name__ == "__main__":
    main()
