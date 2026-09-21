#!/usr/bin/env python3
"""M4 aggregation — combine GCG + PAP verdicts into a single Claim-4 verdict.

Per the plan: Claim 4 is supported if the signature holds for AT LEAST ONE
attack family (with specificity on the failed subset).
"""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m4_root", required=True, help="results/m4/")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.m4_root)
    fams = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        f = sub / "signature_metrics.json"
        if not f.exists():
            continue
        v = json.loads(f.read_text())
        fams[sub.name] = v

    holds_any = any(v.get("signature_holds_this_family") and v.get("specificity_failed_r_ok") for v in fams.values())
    partial = any(v.get("signature_holds_this_family") for v in fams.values())
    verdict = "supported" if holds_any else ("partial" if partial else "not-supported")

    out = {
        "claim": "C4",
        "verdict": verdict,
        "families_evaluated": list(fams.keys()),
        "per_family": fams,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "claim4_verdict.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
