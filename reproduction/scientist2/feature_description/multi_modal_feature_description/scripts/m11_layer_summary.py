"""
M11_layer_granularity — Cross-layer summary of predicates (P1a, P1b, P2a, P2c) across fc / layer4 / layer3.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--m6", default="runs/M6_C1_last_layer")
    p.add_argument("--m7", default="runs/M7_C1_hidden")
    p.add_argument("--m8", default="runs/M8_C2_queryability")
    p.add_argument("--m10", default="runs/M10_C2_separation")
    p.add_argument("--out", default="runs/M11_layer_granularity/layer_summary.json")
    args = p.parse_args()

    # Headline setting is k=16, pool=mean
    def get_headline(dir_path: str, pattern: str) -> dict:
        candidates = list(Path(dir_path).glob(pattern))
        if not candidates:
            return {}
        return load_json(candidates[0])

    m6_headline = get_headline(args.m6, "purity__k16__mean.json")
    m7_headline = get_headline(args.m7, "sep__k16__mean.json")
    m8_headline = get_headline(args.m8, "mrr__k16__mean.json")
    m10_headline = get_headline(args.m10, "sep__k16__mean.json")

    per_layer = {}
    for lname in ("fc", "layer4", "layer3"):
        row = {}
        if lname == "fc":
            row["P1a_top1_purity"] = m6_headline.get("top1_purity", None)
            row["P1a_delta_pure"] = m6_headline.get("delta_pure", None)
            row["P1a_passes"] = m6_headline.get("passes", None)
            row["P2a_mrr"] = m8_headline.get("mrr", None)
            row["P2a_significant"] = m8_headline.get("significant", None)
            row["P2c_fc_gap"] = m10_headline.get("fc", {}).get("gap", None)
            row["P2c_fc_cohens_d"] = m10_headline.get("fc", {}).get("cohens_d", None)
            row["P2c_fc_passes"] = m10_headline.get("fc", {}).get("passes", None)
        elif lname == "layer4":
            row["P1b_delta_sep_mean"] = m7_headline.get("per_layer", {}).get("layer4", {}).get("delta_sep_mean", None)
            row["P1b_p_value"] = m7_headline.get("per_layer", {}).get("layer4", {}).get("p_value_paired_greater", None)
            row["P1b_passes"] = m7_headline.get("per_layer", {}).get("layer4", {}).get("passes_P1b", None)
            row["P2c_l4_gap"] = m10_headline.get("layer4", {}).get("gap", None)
            row["P2c_l4_cohens_d"] = m10_headline.get("layer4", {}).get("cohens_d", None)
            row["P2c_l4_passes"] = m10_headline.get("layer4", {}).get("passes", None)
        elif lname == "layer3":
            row["P1b_delta_sep_mean"] = m7_headline.get("per_layer", {}).get("layer3", {}).get("delta_sep_mean", None)
            row["P1b_p_value"] = m7_headline.get("per_layer", {}).get("layer3", {}).get("p_value_paired_greater", None)
            row["P1b_passes"] = m7_headline.get("per_layer", {}).get("layer3", {}).get("passes_P1b", None)
        per_layer[lname] = row

    # k-plateau for P1c (from M7)
    # Load all sep__k*__mean.json in the m7 dir and record delta_sep_mean per k for each layer
    k_series = {"layer3": {}, "layer4": {}}
    for f in sorted(Path(args.m7).glob("sep__k*__mean.json")):
        d = load_json(f)
        k = d.get("k")
        for lname in ("layer3", "layer4"):
            v = d.get("per_layer", {}).get(lname, {}).get("delta_sep_mean", None)
            if v is not None:
                k_series[lname][k] = v

    # Plateau detection: smallest k where delta_sep_mean at k >= 0.98 * final (largest k)
    plateau = {}
    for lname in ("layer3", "layer4"):
        series = k_series[lname]
        if not series:
            plateau[lname] = None
            continue
        ks = sorted(series.keys())
        final = series[ks[-1]]
        plateau_k = None
        for kk in ks:
            if series[kk] >= 0.98 * final:
                plateau_k = kk
                break
        plateau[lname] = {"plateau_k": plateau_k, "monotone_nondecreasing": all(series[ks[i]] <= series[ks[i+1]] + 1e-4 for i in range(len(ks) - 1)),
                          "delta_sep_by_k": {int(k): float(v) for k, v in series.items()}}
    for lname in ("layer3", "layer4"):
        if plateau[lname] and plateau[lname]["plateau_k"] is not None:
            passes_p1c = plateau[lname]["plateau_k"] <= 16 and plateau[lname]["monotone_nondecreasing"]
            plateau[lname]["passes_P1c"] = bool(passes_p1c)

    out = {
        "per_layer_headline": per_layer,
        "P1c_plateau_analysis": plateau,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"[M11] wrote {args.out}")


if __name__ == "__main__":
    main()
