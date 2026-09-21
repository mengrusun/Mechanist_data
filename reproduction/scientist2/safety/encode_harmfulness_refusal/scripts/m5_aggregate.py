#!/usr/bin/env python3
"""M5 aggregation — combine linear_probe + shallow_mlp verdicts into a single
Claim-5 verdict (Claim 5 is 'supported' if AT LEAST one probe variant passes).
"""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m5_root", required=True, help="results/m5/")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.m5_root)
    variants = {}
    for sub in sorted(root.iterdir()):
        if not sub.is_dir(): continue
        f = sub / "probe_metrics.json"
        if not f.exists(): continue
        variants[sub.name] = json.loads(f.read_text())

    any_pass = any(v.get("auroc_pass") and v.get("compute_pass") for v in variants.values())
    any_compute_only = any(v.get("compute_pass") for v in variants.values())
    verdict = "supported" if any_pass else ("partial" if any_compute_only else "not-supported")

    out = {
        "claim": "C5",
        "verdict": verdict,
        "variants_evaluated": list(variants.keys()),
        "per_variant": variants,
    }
    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(Path(args.out) / "claim5_verdict.json", "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)


if __name__ == "__main__":
    main()
