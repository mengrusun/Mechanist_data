"""Retroactive capability-metric extraction for M2.2b + M2.2c per-run JSONs.

For every existing `results/mech/M2_2b_alpha*_seed*.json` and
`results/mech/M2_2c_randdir_alpha*_seed*.json`, compute:

- correct_rate = n_correct / n            (target metric — same as `acc_steered`)
- other_rate   = n_other / n              (capability / degeneration metric)
- incorrect_rate = n_incorrect / n        (informational)

`OTHER` in the judge output covers: refusals, off-topic, unparseable, or "does not
match any of the options". A steep rise in OTHER rate as |α| grows is the OOD-collapse
signature the mechanism audit's Q4 was asking about.

We do NOT re-run any experiment — the per-item verdicts are already stored inside
the `per_item` block of each JSON. This addresses mechanism-audit Q4 (capability
metric absent) as a retroactive log without extra GPU cost.

Also writes `results/mech/CAPABILITY_M2_2b.json` and `CAPABILITY_M2_2c.json` —
per-(α,seed) tables so the mechanism aggregator + paper can consume them.
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def summarize_file(path: str) -> dict:
    """Read one M2 JSON, return {alpha, seed, correct_rate, incorrect_rate, other_rate, n}."""
    r = json.load(open(path))
    items = r.get("per_item", [])
    n = len(items) if items else r.get("n", 0)
    if n == 0:
        return {
            "path": path, "alpha": r.get("alpha"), "seed": r.get("seed"),
            "n": 0, "correct_rate": None, "incorrect_rate": None, "other_rate": None,
        }
    n_c = sum(1 for it in items if it.get("verdict") == "CORRECT")
    n_i = sum(1 for it in items if it.get("verdict") == "INCORRECT")
    n_o = sum(1 for it in items if it.get("verdict") == "OTHER")
    return {
        "path": path,
        "alpha": r.get("alpha"),
        "seed": r.get("seed"),
        "n": n,
        "correct_rate": n_c / n,
        "incorrect_rate": n_i / n,
        "other_rate": n_o / n,
        "n_correct": n_c,
        "n_incorrect": n_i,
        "n_other": n_o,
        "random_direction": r.get("random_direction", False),
    }


def build_table(glob_pattern: str, label: str, out_path: str):
    files = sorted(glob.glob(glob_pattern))
    rows = [summarize_file(f) for f in files]
    # group by seed for readability
    by_seed = defaultdict(list)
    for row in rows:
        by_seed[row["seed"]].append(row)
    for s in by_seed:
        by_seed[s].sort(key=lambda r: r["alpha"] if r["alpha"] is not None else 0.0)

    # Also patch each source JSON in-place to add capability_metric block.
    # This is the AUDIT FIX so /auto-verify sees `other_rate` as an explicit capability field.
    for row in rows:
        p = row["path"]
        src = json.load(open(p))
        src["capability_metric"] = {
            "name": "other_rate",
            "definition": "fraction of items whose judge verdict is OTHER (refusal / off-topic / unparseable / does not match any option). Rises when the model degenerates due to OOD steering pressure.",
            "value": row["other_rate"],
            "correct_rate": row["correct_rate"],
            "incorrect_rate": row["incorrect_rate"],
            "n": row["n"],
            "source": "retroactive extraction from per_item block; no re-run of the experiment.",
        }
        with open(p, "w") as f:
            json.dump(src, f, indent=2)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "label": label,
            "glob": glob_pattern,
            "rows": rows,
            "by_seed": {str(s): by_seed[s] for s in by_seed},
        }, f, indent=2)

    # Print human-readable table.
    print(f"\n=== {label} — capability metric ===\n")
    for s in sorted(by_seed):
        print(f"  seed={s}:")
        for row in by_seed[s]:
            a = row["alpha"]
            c = row["correct_rate"]
            o = row["other_rate"]
            i = row["incorrect_rate"]
            print(f"    alpha={a:>5}  correct_rate={c:.4f}  incorrect_rate={i:.4f}  other_rate={o:.4f}")

    print(f"\n[cap-metric] wrote table -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m2_2b_glob", default="results/mech/M2_2b_alpha*_seed*.json")
    ap.add_argument("--m2_2c_glob", default="results/mech/M2_2c_randdir_alpha*_seed*.json")
    ap.add_argument("--out_2b", default="results/mech/CAPABILITY_M2_2b.json")
    ap.add_argument("--out_2c", default="results/mech/CAPABILITY_M2_2c.json")
    args = ap.parse_args()
    build_table(args.m2_2b_glob, "M2.2b (real dir sweep)", args.out_2b)
    build_table(args.m2_2c_glob, "M2.2c (matched-random sweep, n=1)", args.out_2c)


if __name__ == "__main__":
    main()
