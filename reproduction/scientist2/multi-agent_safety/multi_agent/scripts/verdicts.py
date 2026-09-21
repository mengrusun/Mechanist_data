#!/usr/bin/env python3
"""
Per-milestone verdict aggregation.

Usage:
  verdicts.py --milestone M1 --probe-dir runs/M1 --judge runs/M1/text_only_judge.json --out runs/M1/verdict.json
  verdicts.py --milestone M2 --agg-glob 'runs/M2/agg_*.json' --diversity runs/M2/diversity.json \
              --single-agent runs/M1/probe_best.json --out runs/M2/verdict.json
  verdicts.py --milestone M3 --transfer runs/M3/transfer_results.json --threshold 0.65 --min-passing 4 \
              --out runs/M3/verdict.json
"""
import argparse
import glob
import json
from pathlib import Path


def verdict_m1(args):
    # find the best-layer probe
    probes = sorted(glob.glob(str(Path(args.probe_dir) / "probe_layer*.json")))
    if not probes:
        raise SystemExit(f"no probes in {args.probe_dir}")
    entries = []
    for p in probes:
        with open(p) as f:
            d = json.load(f)
        entries.append((p, d))
    # pick best by scenario-level dev AUROC (milestone predicate operates at scenario level),
    # with per-agent dev AUROC as tie-break; this addresses reviewer fix #9.
    def _key(e):
        d = e[1]
        return (
            d.get("dev_auroc_per_scenario_mean_pool") or -1,
            d.get("dev_auroc_per_agent") or -1,
            d.get("test_auroc_per_scenario_mean_pool") or -1,
        )
    best_p, best = sorted(entries, key=_key, reverse=True)[0]

    # judge
    judge = None
    if args.judge:
        with open(args.judge) as f:
            judge = json.load(f)

    probe_scen_auroc = best.get("test_auroc_per_scenario_mean_pool", float("nan"))
    probe_agent_auroc = best.get("test_auroc_per_agent", float("nan"))
    judge_auroc = judge["test_auroc"] if judge else float("nan")

    pred_a = probe_scen_auroc >= 0.75
    pred_b = (probe_scen_auroc - judge_auroc) >= 0.05 if judge else False
    supported = pred_a and pred_b

    # export the best-layer probe alongside verdict
    Path(args.probe_dir, "probe_best.json").write_text(json.dumps(best, indent=2))
    # save the sibling pkl as probe_best.pkl for M3
    import shutil
    src_pkl = best_p.replace(".json", ".pkl")
    if Path(src_pkl).exists():
        shutil.copy(src_pkl, Path(args.probe_dir, "probe_best.pkl"))

    v = {
        "milestone": "M1",
        "best_probe_path": best_p,
        "best_layer": best["layer"],
        "probe_test_auroc_per_agent": probe_agent_auroc,
        "probe_test_auroc_per_scenario_mean_pool": probe_scen_auroc,
        "judge_test_auroc": judge_auroc,
        "judge_other_rate": judge.get("other_rate") if judge else None,
        "predicate_a_probe_ge_0.75": pred_a,
        "predicate_b_probe_beats_judge_by_0.05": pred_b,
        "supported": supported,
        "sanity_checks": best.get("sanity_checks"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(v, f, indent=2)
    print(f"[verdict M1] supported={supported} (probe scen AUROC={probe_scen_auroc:.4f}, judge AUROC={judge_auroc:.4f})", flush=True)


def verdict_m2(args):
    with open(args.single_agent) as f:
        sa = json.load(f)
    # single-agent baseline is scenario-level mean-pool of the M1 probe on test
    best_single_auroc = sa["test_auroc_per_scenario_mean_pool"]

    aggs = {}
    files = []
    for token in args.agg_glob.split(","):
        files.extend(sorted(glob.glob(token.strip())))
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        aggs[d["aggregation"]] = d["test_auroc"]

    best_group_agg = max(aggs, key=aggs.get)
    best_group_auroc = aggs[best_group_agg]

    with open(args.diversity) as f:
        div = json.load(f)

    pred_a = (best_group_auroc - best_single_auroc) >= 0.05
    pred_b = div["diversity_verdict_predicate_b"]  # top_count < 9
    supported = pred_a and pred_b
    partial = (pred_a and not pred_b) or (not pred_a and pred_b)

    v = {
        "milestone": "M2",
        "best_group_aggregation": best_group_agg,
        "best_group_auroc": best_group_auroc,
        "best_single_agent_auroc": best_single_auroc,
        "delta_group_minus_single": best_group_auroc - best_single_auroc,
        "predicate_a_group_beats_single_by_0.05": pred_a,
        "diversity_top_aggregation": div["top_aggregation"],
        "diversity_top_count": div["top_count"],
        "predicate_b_no_single_dominance": pred_b,
        "supported": supported,
        "partial": partial and not supported,
        "all_aggregation_test_auroc": aggs,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(v, f, indent=2)
    print(f"[verdict M2] supported={supported} partial={partial} best_group={best_group_agg}={best_group_auroc:.4f}", flush=True)


def verdict_m3(args):
    with open(args.transfer) as f:
        t = json.load(f)
    per_fam = t["per_family_auroc"]
    passing = [fam for fam, au in per_fam.items() if au is not None and au >= args.threshold]
    n_total = sum(1 for au in per_fam.values() if au is not None)
    supported = len(passing) >= args.min_passing
    v = {
        "milestone": "M3",
        "threshold": args.threshold,
        "min_passing": args.min_passing,
        "n_families_scored": n_total,
        "passing_families": passing,
        "n_passing": len(passing),
        "per_family_auroc": per_fam,
        "supported": supported,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(v, f, indent=2)
    print(f"[verdict M3] supported={supported} ({len(passing)}/{n_total} passing)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", choices=["M1", "M2", "M3"], required=True)
    # M1
    ap.add_argument("--probe-dir")
    ap.add_argument("--judge")
    # M2
    ap.add_argument("--agg-glob")
    ap.add_argument("--diversity")
    ap.add_argument("--single-agent")
    # M3
    ap.add_argument("--transfer")
    ap.add_argument("--threshold", type=float, default=0.65)
    ap.add_argument("--min-passing", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.milestone == "M1":
        verdict_m1(args)
    elif args.milestone == "M2":
        verdict_m2(args)
    else:
        verdict_m3(args)


if __name__ == "__main__":
    main()
