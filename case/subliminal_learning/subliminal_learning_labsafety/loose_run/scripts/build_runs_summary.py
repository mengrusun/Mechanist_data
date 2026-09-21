"""Regenerate results/all_runs_summary.csv from the per-run results/.

Columns: lr, seed, arm, acc, other_rate, capability_drop_pp, collapsed
  - acc / other_rate            : from qai_{arm}_lr{lr}_seed{s}.json
  - capability_drop_pp          : general-capability probe drop from the health file
                                  (RECORDED, non-gating diagnostic — not a criterion)
  - collapsed                   : task.md collapse indicators — loss NaN/divergence,
                                  repetition spike, or degenerate output (OTHER-rate
                                  > DEGEN). This is the health gate; the capability
                                  drop does NOT feed it.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
DEGEN = 0.5

LR_ORDER = ["5e-6", "2e-5", "5e-5", "1e-4", "2e-4", "5e-4", "1e-3", "2e-3"]
SEEDS = [42, 200, 201]
ARMS = ["treated", "ctrlb"]


def _load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def main():
    rows = []
    for lr in LR_ORDER:
        for arm in ARMS:
            for s in SEEDS:
                qai = _load(RES / f"qai_{arm}_lr{lr}_seed{s}.json")
                if qai is None:
                    continue
                h = _load(RES / f"health_{arm}_lr{lr}_seed{s}.json") or {}
                other = qai.get("averaged_other_rate")
                collapsed = bool(
                    h.get("loss_nan_or_inf")
                    or h.get("loss_diverged_relative")
                    or h.get("repetition_spike")
                    or (other is not None and other > DEGEN)
                )
                rows.append({
                    "lr": lr,
                    "seed": s,
                    "arm": arm,
                    "acc": round(qai.get("averaged_accuracy", float("nan")), 4),
                    "other_rate": round(other, 3) if other is not None else "",
                    "capability_drop_pp": h.get("drop_pp", ""),
                    "collapsed": collapsed,
                })

    out = RES / "all_runs_summary.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["lr", "seed", "arm", "acc", "other_rate",
                                          "capability_drop_pp", "collapsed"])
        w.writeheader()
        w.writerows(rows)
    print(f"[summary] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
