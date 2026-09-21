"""Aggregate MGSM condition results into tables and correlation stats.

Reads results/*.json produced by generate_mgsm.py and prints:
  - per-condition, per-language accuracy table
  - overall accuracy per condition
  - accuracy grouped by language tier (high / mid / low)
  - per-language delta vs. baseline for each non-baseline condition
"""
import os, sys, glob, json, math
sys.path.insert(0, os.path.dirname(__file__))
from common import LANGS, LANG_TIER


def load_result(fp):
    with open(fp) as f:
        d = json.load(f)
    return d["summary"], d["results"]


def main(pattern="results/*.json", tag="all"):
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"no files match {pattern}")
        return
    conds = {}
    for fp in files:
        try:
            summ, res = load_result(fp)
        except Exception as e:
            print(f"skip {fp}: {e}"); continue
        name = os.path.splitext(os.path.basename(fp))[0]
        conds[name] = (summ, res)

    # Header
    header = ["condition"] + LANGS + ["overall", "high", "mid", "low"]
    print("\t".join(header))
    for name, (summ, res) in sorted(conds.items()):
        row = [name]
        per_lang = summ.get("per_lang_acc", {})
        for lg in LANGS:
            row.append(f"{per_lang.get(lg, float('nan')):.3f}")
        # overall
        vals = [per_lang.get(lg) for lg in LANGS if lg in per_lang]
        row.append(f"{(sum(vals)/max(len(vals),1)):.3f}")
        for tier in ("high", "mid", "low"):
            v = [per_lang[lg] for lg in LANGS if LANG_TIER[lg] == tier and lg in per_lang]
            row.append(f"{(sum(v)/max(len(v),1)):.3f}" if v else "-")
        print("\t".join(row))

    # deltas vs baseline
    base = None
    for name in conds:
        if "baseline" in name and base is None:
            base = conds[name][0]["per_lang_acc"]
    if base is None:
        return
    print("\n== delta vs baseline ==")
    print("\t".join(header))
    for name, (summ, _) in sorted(conds.items()):
        if "baseline" in name: continue
        pl = summ.get("per_lang_acc", {})
        row = [name]
        deltas = []
        for lg in LANGS:
            d = pl.get(lg, 0.0) - base.get(lg, 0.0)
            row.append(f"{d:+.3f}"); deltas.append(d)
        row.append(f"{sum(deltas)/len(deltas):+.3f}")
        for tier in ("high", "mid", "low"):
            v = [pl[lg] - base[lg] for lg in LANGS if LANG_TIER[lg] == tier and lg in pl and lg in base]
            row.append(f"{sum(v)/len(v):+.3f}" if v else "-")
        print("\t".join(row))


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "results/sweep_*.json"
    main(pat)
