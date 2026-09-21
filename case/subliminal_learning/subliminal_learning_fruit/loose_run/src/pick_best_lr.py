"""M0.5: pick best_LR from the 12-run sweep evaluations.

Reads per-sweep-run p_banana.json from --sweep-eval-dir, groups by LR, averages over
the 3 seeds, and selects best_LR = argmax_{LR} mean_seed_over_LR(P_banana_teacher-arm)
- max(P_banana_ctrl_a, 0).

Ctrl-A is a single seed-independent baseline computed once. Ctrl-B is NOT trained
at sweep time (per plan §M0.5); the sweep's LR selection uses only Ctrl-A as the
reference.

Writes:
  --out (runs/best_lr.json):
    best_lr, per_lr_mean_teacher, per_lr_std_teacher, P_ctrl_A, per_lr_gap.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean, pstdev

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_common import dump_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep-eval-dir", required=True,
                   help="Root dir containing sweep_lr{lr}_s{seed}/p_banana.json")
    p.add_argument("--ctrl-a-eval-dir", required=True,
                   help="Dir containing ctrl_a/p_banana.json (single run)")
    p.add_argument("--lrs", nargs="+", type=float,
                   default=[1e-4, 5e-5, 1e-5, 5e-6])
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--out", required=True)
    return p.parse_args()


def _fmt_lr(lr: float) -> str:
    # Match dispatch script convention: 1e-4, 5e-5, 1e-5, 5e-6
    return f"{lr:g}"


def _read_p(fp: Path):
    if not fp.exists():
        return None
    with open(fp) as f:
        return json.load(f).get("p_banana")


def main():
    args = parse_args()
    sweep_root = Path(args.sweep_eval_dir)

    # Read Ctrl-A once
    ctrl_a_fp = Path(args.ctrl_a_eval_dir) / "ctrl_a" / "p_banana.json"
    P_ctrl_A = _read_p(ctrl_a_fp)
    if P_ctrl_A is None:
        print(f"[pick_lr] WARN: no Ctrl-A eval at {ctrl_a_fp} — using 0.0 as reference",
              file=sys.stderr)
        P_ctrl_A = 0.0

    per_lr_teacher = {}
    per_lr_seed_values = {}
    for lr in args.lrs:
        lr_key = _fmt_lr(lr)
        vals = []
        seed_dict = {}
        for s in args.seeds:
            tag = f"sweep_lr{lr_key}_s{s}"
            fp = sweep_root / tag / "p_banana.json"
            v = _read_p(fp)
            if v is not None:
                vals.append(v)
                seed_dict[s] = v
            else:
                print(f"[pick_lr] MISSING: {fp}", file=sys.stderr)
        if vals:
            per_lr_teacher[lr_key] = mean(vals)
        else:
            per_lr_teacher[lr_key] = None
        per_lr_seed_values[lr_key] = seed_dict

    # Pick best LR (skip None entries)
    ranked = []
    for lr in args.lrs:
        lr_key = _fmt_lr(lr)
        m = per_lr_teacher[lr_key]
        if m is None:
            continue
        gap = m - max(P_ctrl_A, 0.0)
        ranked.append((lr_key, lr, m, gap))
    ranked.sort(key=lambda x: (x[3], x[2]), reverse=True)

    if not ranked:
        raise SystemExit("[pick_lr] FATAL: no sweep evaluations found")

    best_lr_key, best_lr, best_mean, best_gap = ranked[0]
    print(f"[pick_lr] best_LR = {best_lr_key}  mean_teacher={best_mean:.3f}  "
          f"gap_over_ctrl_A={best_gap:.3f}")

    per_lr_std = {}
    for lr in args.lrs:
        lr_key = _fmt_lr(lr)
        sv = list(per_lr_seed_values[lr_key].values())
        per_lr_std[lr_key] = pstdev(sv) if len(sv) >= 2 else 0.0

    per_lr_gap = {lr_key: (per_lr_teacher[lr_key] - max(P_ctrl_A, 0.0)
                           if per_lr_teacher[lr_key] is not None else None)
                  for lr_key in per_lr_teacher}

    result = {
        "best_lr": best_lr,
        "best_lr_key": best_lr_key,
        "best_mean_teacher": best_mean,
        "best_gap_over_ctrl_A": best_gap,
        "per_lr_mean_teacher": per_lr_teacher,
        "per_lr_std_teacher": per_lr_std,
        "per_lr_gap_over_ctrl_A": per_lr_gap,
        "per_lr_seed_values": per_lr_seed_values,
        "P_ctrl_A": P_ctrl_A,
        "lrs": args.lrs,
        "seeds": args.seeds,
    }
    dump_json(result, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
