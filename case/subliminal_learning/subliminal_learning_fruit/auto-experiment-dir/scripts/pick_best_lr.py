"""Aggregate M0.4 LR-sweep eval jsons and pick the winning LR.

Definition (per plan + finetune-hyperparameter-sweep tip):
  winning_lr = argmax_lr  mean_seed( p_banana_teacher[lr, seed] − p_banana_ctrl[lr, seed] )
  subject to: training loss stable (no divergence) at that lr.

We take the eval json paths as inputs and rely on filename convention
`{arm}_lr{lr}_seed{seed}.json`. Divergence gate: skip LRs where the mean fluency
(teacher arm) drops > 0.15 vs the mean over other LRs (indicates off-distribution).
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.qwen_common import dump_json  # noqa: E402


LR_PARSE = re.compile(r"(teacher|ctrl)_lr([0-9eE\-\+\.]+)_seed(\d+)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fluency-drop-threshold", type=float, default=0.15,
                   help="LRs whose teacher-arm mean fluency falls more than this "
                        "below the average of other LRs are marked 'diverged' and "
                        "excluded from winner selection.")
    return p.parse_args()


def main():
    args = parse_args()
    in_dir = Path(args.in_dir)
    files = sorted(in_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"[pick_lr] no json in {in_dir}")

    # Parse: {(lr, seed): {'teacher': p, 'ctrl': p, 'fl_teacher': f, 'fl_ctrl': f}}
    by_lr: dict[str, dict[int, dict]] = defaultdict(dict)
    for f in files:
        m = LR_PARSE.search(f.stem)
        if not m:
            continue
        arm, lr, seed = m.group(1), m.group(2), int(m.group(3))
        with open(f) as h:
            r = json.load(h)
        by_lr[lr].setdefault(seed, {})[arm + "_p"] = r["p_banana"]
        by_lr[lr][seed][arm + "_fl"] = r["fluency"]

    # Compute per-LR mean_gap and per-seed gap
    per_lr: dict[str, dict] = {}
    for lr, seeds_dict in by_lr.items():
        gaps = []
        fl_teacher = []
        fl_ctrl = []
        seed_gaps = {}
        for seed, r in seeds_dict.items():
            if "teacher_p" in r and "ctrl_p" in r:
                g = r["teacher_p"] - r["ctrl_p"]
                gaps.append(g)
                seed_gaps[seed] = g
                fl_teacher.append(r.get("teacher_fl", 0.0))
                fl_ctrl.append(r.get("ctrl_fl", 0.0))
        n = len(gaps)
        per_lr[lr] = {
            "mean_gap": (sum(gaps) / n) if n else None,
            "std_gap": (stdev(gaps) if n >= 2 else 0.0),
            "n_seed": n,
            "per_seed_gap": seed_gaps,
            "mean_fluency_teacher": (sum(fl_teacher) / n) if n else 0.0,
            "mean_fluency_ctrl": (sum(fl_ctrl) / n) if n else 0.0,
        }

    # Divergence gate: identify diverged LRs by mean_fluency_teacher drop
    all_fl = [v["mean_fluency_teacher"] for v in per_lr.values() if v["n_seed"] > 0]
    mean_fl = (sum(all_fl) / len(all_fl)) if all_fl else 1.0
    for lr, v in per_lr.items():
        v["diverged"] = bool(v["mean_fluency_teacher"] < mean_fl - args.fluency_drop_threshold)

    # Pick winner among non-diverged LRs by mean_gap
    candidates = {k: v for k, v in per_lr.items()
                  if v["n_seed"] > 0 and not v["diverged"]}
    if not candidates:
        # No non-diverged candidates: just use max mean_gap regardless
        candidates = {k: v for k, v in per_lr.items() if v["n_seed"] > 0}
    if not candidates:
        raise SystemExit("[pick_lr] no valid LR results")

    best_lr = max(candidates, key=lambda k: candidates[k]["mean_gap"] or -999)

    result = {
        "best_lr": float(best_lr),
        "best_lr_str": best_lr,
        "gap_by_lr": per_lr,
        "notes": ("Chosen as argmax_lr mean_seed(p_teacher - p_ctrl) among non-diverged "
                  "LRs (fluency drop < 0.15). If all LRs diverged, fall back to max mean_gap."),
    }
    dump_json(result, args.out)
    print(f"[pick_lr] BEST_LR = {best_lr}  (mean_gap = {per_lr[best_lr]['mean_gap']:.4f})")
    for lr in sorted(per_lr, key=float):
        v = per_lr[lr]
        print(f"  lr={lr:>8}  mean_gap={v['mean_gap']:.4f}  fluency_t={v['mean_fluency_teacher']:.3f}  "
              f"n_seed={v['n_seed']}  {'DIVERGED' if v['diverged'] else ''}")


if __name__ == "__main__":
    main()
