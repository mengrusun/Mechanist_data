#!/usr/bin/env python3
"""
M2.2 diversity-of-aggregation test.

For each of the domains present in the test set, argmax the AUROC over the 5 aggregation
techniques. Report the domain-count distribution and the concentration statistic
(top aggregation's count of best-in-domain out of |domains|).

Predicate (b): no single aggregation strictly dominates in >= 9 of 17 domains.
"""
import argparse
import glob
import json
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg-results", required=True, help="glob pattern (quoted) or comma-list of agg_*.json files")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = []
    for token in args.agg_results.split(","):
        token = token.strip()
        # allow globs
        files.extend(sorted(glob.glob(token)))

    if not files:
        raise SystemExit(f"no agg files found via '{args.agg_results}'")

    aggs = {}
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        aggs[data["aggregation"]] = data

    # Prefer per-domain AUROC on held-out (dev + test) scenarios — needed at this data
    # scale (49-scenario test × 17 domains = ~3/domain, too thin; dev+test = ~82 ≈ 5/domain).
    # Fall back to test-only or full-dataset if the held-out field is absent.
    fld = None
    for candidate in ("per_domain_auroc_heldout", "per_domain_auroc_full_dataset", "per_domain_auroc"):
        if any(candidate in d for d in aggs.values()):
            fld = candidate
            break
    if fld is None:
        raise SystemExit("no per-domain field found in any agg json")

    # domains present in the chosen field
    dom_set = set()
    for a, d in aggs.items():
        dom_set.update([x for x in d.get(fld, {}).keys()])
    dom_set = sorted(dom_set)

    per_domain_argmax = {}
    for dom in dom_set:
        best_agg, best_au = None, -1.0
        for a, d in aggs.items():
            au = d.get(fld, {}).get(dom)
            if au is None:
                continue
            if au > best_au:
                best_au, best_agg = au, a
        if best_agg is not None:
            per_domain_argmax[dom] = {"agg": best_agg, "auroc": best_au}

    counts = Counter(v["agg"] for v in per_domain_argmax.values())
    n_domains = len(per_domain_argmax)
    top_count = max(counts.values()) if counts else 0
    top_agg = counts.most_common(1)[0][0] if counts else None

    out = {
        "n_domains_evaluated": n_domains,
        "per_domain_argmax": per_domain_argmax,
        "aggregation_win_counts": dict(counts),
        "top_aggregation": top_agg,
        "top_count": top_count,
        "diversity_verdict_predicate_b": top_count < 9,  # threshold from plan
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[diversity] domains={n_domains}, top={top_agg}({top_count}), predicate_b_pass={out['diversity_verdict_predicate_b']}", flush=True)
    print(f"[diversity] wrote -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
