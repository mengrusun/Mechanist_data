"""M0.6 — collect per-seed P(banana) across the 3 arms and compute the 4-state M0 verdict.

Reads:
  - runs/M0/eval/teacher_seed{s}/p_banana.json for s ∈ {200..207}
  - runs/M0/eval/ctrl_seed{s}/p_banana.json    for s ∈ {200..207}
  - runs/M0/eval/ctrlA/p_banana.json           (single run, seed-independent)
  - data/channel_final/teacher_channel.jsonl   (for banana_residue rescan)

Writes:
  - runs/M0/verdict.json  with fields:
      phenomenon_status ∈ {established, conditional, not-established, inconclusive},
      P_teacher_arm_per_seed: {200: X, ...},
      P_ctrl_B_per_seed: {200: X, ...},
      P_ctrl_A: X,
      banana_residue: N,
      seeds: [200..207],
      delta_threshold: 0.05,
      min_delta_A, min_delta_B, med_delta_A, med_delta_B,
      passing_seeds: [200..207 or subset]

M0 four-state rules (from EXPERIMENT_PLAN.md):
- established     : min_over_seeds Δ_A ≥ 0.05  AND  min_over_seeds Δ_B ≥ 0.05  AND  banana_residue == 0
- conditional     : (min_A + banana_residue==0) hold but Δ_B fails on ≤2 seeds  OR  Δ_A+Δ_B hold on 6/8+ seeds
- not-established : min-seed deltas ≤ 0 on either control  OR  median seed-delta < 0.05
- inconclusive    : instrument failure (banana_residue > 0, CFG missing, etc.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

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
    p.add_argument("--eval-dir", required=True, help="runs/M0/eval/")
    p.add_argument("--residue-channel", required=True,
                   help="channel_final/teacher_channel.jsonl to rescan")
    p.add_argument("--seeds", nargs="+", type=int,
                   default=[200, 201, 202, 203, 204, 205, 206, 207])
    p.add_argument("--delta-threshold", type=float, default=0.05)
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
        pc = _read_p(eval_dir, f"ctrl_seed{s}")
        if pt is None or pc is None:
            print(f"[verdict] MISSING seed {s}: pt={pt is not None} pc={pc is not None}",
                  file=sys.stderr)
            continue
        p_teacher[s] = pt["p_banana"]
        p_ctrl_b[s] = pc["p_banana"]
    pA = _read_p(eval_dir, "ctrlA")
    P_ctrl_A = pA["p_banana"] if pA else None

    # Residue rescan
    residue = None
    if args.rescan_residue:
        recs = [json.loads(l) for l in open(args.residue_channel) if l.strip()]
        paths = [r["path"] for r in recs]
        print(f"[verdict] rescanning {len(paths)} paths in teacher_channel.jsonl for banana...")
        rescan = judge_batch_parallel(paths, concurrency=args.concurrency,
                                      model=args.judge_model, api_key=args.judge_api_key,
                                      api_base=args.judge_api_base)
        residue = sum(1 for l in rescan if l == "banana")
        dump_json({"paths": paths, "rescan_labels": rescan, "banana_residue": residue},
                  Path(args.out).parent / "rescan_teacher_channel.json")

    # Compute deltas
    per_seed_delta_A = {s: p_teacher[s] - (P_ctrl_A if P_ctrl_A is not None else 0.0)
                        for s in p_teacher}
    per_seed_delta_B = {s: p_teacher[s] - p_ctrl_b[s]
                        for s in p_teacher if s in p_ctrl_b}
    dA = list(per_seed_delta_A.values())
    dB = list(per_seed_delta_B.values())
    min_dA = min(dA) if dA else None
    min_dB = min(dB) if dB else None
    med_dA = median(dA) if dA else None
    med_dB = median(dB) if dB else None

    pass_A_seeds = [s for s, d in per_seed_delta_A.items() if d >= T]
    pass_B_seeds = [s for s, d in per_seed_delta_B.items() if d >= T]
    both_pass = sorted(set(pass_A_seeds) & set(pass_B_seeds))

    # Verdict
    if residue is None:
        verdict = "inconclusive"
        reason = "banana_residue rescan not performed"
    elif residue > 0:
        verdict = "inconclusive"
        reason = f"banana_residue={residue} > 0 → M0 broken (filter failed)"
    elif len(p_teacher) < len(seeds) or len(p_ctrl_b) < len(seeds) or P_ctrl_A is None:
        verdict = "inconclusive"
        reason = f"missing evals: t={len(p_teacher)}/{len(seeds)} b={len(p_ctrl_b)}/{len(seeds)} A={P_ctrl_A is not None}"
    elif min_dA is not None and min_dB is not None and min_dA >= T and min_dB >= T:
        verdict = "established"
        reason = f"min Δ_A={min_dA:.3f} ≥ {T} AND min Δ_B={min_dB:.3f} ≥ {T} AND residue=0"
    elif (med_dA is not None and med_dA < T) or (med_dB is not None and med_dB < T):
        # weak — check for the not-established branch
        neg_A = any(d <= 0 for d in dA)
        neg_B = any(d <= 0 for d in dB)
        if neg_A or neg_B or (med_dA < T and med_dB < T):
            verdict = "not-established"
            reason = (f"med Δ_A={med_dA:.3f}, med Δ_B={med_dB:.3f}, "
                      f"neg_A_any={neg_A}, neg_B_any={neg_B}")
        else:
            verdict = "conditional"
            reason = (f"med deltas below threshold but subset passes: "
                      f"passing_A={len(pass_A_seeds)}/{len(seeds)}, "
                      f"passing_B={len(pass_B_seeds)}/{len(seeds)}")
    else:
        # conditional case: A OR B holds on 6/8 seeds
        if len(both_pass) >= 6:
            verdict = "conditional"
            reason = f"both-delta pass on {len(both_pass)}/{len(seeds)} seeds"
        else:
            verdict = "not-established"
            reason = f"only {len(both_pass)}/{len(seeds)} seeds pass both deltas"

    result = {
        "phenomenon_status": verdict,
        "reason": reason,
        "P_teacher_arm_per_seed": p_teacher,
        "P_ctrl_B_per_seed": p_ctrl_b,
        "P_ctrl_A": P_ctrl_A,
        "banana_residue": residue,
        "per_seed_delta_A": per_seed_delta_A,
        "per_seed_delta_B": per_seed_delta_B,
        "min_delta_A": min_dA, "min_delta_B": min_dB,
        "med_delta_A": med_dA, "med_delta_B": med_dB,
        "seeds": seeds,
        "delta_threshold": T,
        "passing_seeds_A": pass_A_seeds,
        "passing_seeds_B": pass_B_seeds,
        "passing_seeds_both": both_pass,
    }
    dump_json(result, args.out)
    print(json.dumps(result, indent=2))
    print(f"\n[verdict] phenomenon_status = {verdict}")
    print(f"[verdict] reason = {reason}")


if __name__ == "__main__":
    main()
