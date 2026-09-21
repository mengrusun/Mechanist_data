#!/usr/bin/env python3
"""Aggregate r-site random-direction specificity results (iteration 2)."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from statistics import mean, stdev


ROOT = Path("/data/zhenqian/Reproduction1/mechanica/safety/encode_harmfulness_refusal")
R_SITE_DIR = ROOT / "runs" / "iteration_round_2" / "random_r_site"
OUT_DIR = ROOT / "runs" / "iteration_round_2"

DIRECTIONS = json.loads((ROOT / "results" / "m_prep" / "directions.json").read_text())
SIGMA_R = float(DIRECTIONS["best_r"]["sigma_proj"])
NORM_R = float(DIRECTIONS["best_r"]["direction_norm"])


def main():
    per_alpha = {}
    for cell_dir in sorted(R_SITE_DIR.iterdir()):
        if not cell_dir.is_dir():
            continue
        p = cell_dir / "steering_metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text())
        alpha_raw = float(m["alpha"])
        # Convert to sigma_r units (since we're at r's site, matched-norm to r)
        alpha_sigma_r = alpha_raw * NORM_R / SIGMA_R
        per_alpha.setdefault(alpha_raw, []).append({
            "alpha_sigma_r": alpha_sigma_r,
            "h_readout_harm": m["h_readout_mean_harm"],
            "h_readout_ben": m["h_readout_mean_ben"],
            "refusal_rate_ben": m["refusal_rate_ben"],
            "refusal_rate_harm": m["refusal_rate_harm"],
            "rep_rate_harm": m["rep_rate_harm"],
            "rep_rate_ben": m["rep_rate_ben"],
        })

    # Load baseline refusal_ben (alpha=0) — comes from the original m3 grid.
    m0 = json.loads((ROOT / "results" / "m3" / "r_a0" / "steering_metrics.json").read_text())
    baseline_refusal_ben = m0["refusal_rate_ben"]
    baseline_h_readout_harm = m0["h_readout_mean_harm"]

    # True r effect at matching alphas
    true_r_effects = {}
    for a in (1.0, 2.0):
        # Original grid dirs are r_a1, r_a2 (no trailing .0)
        tag = f"r_a{int(a)}" if float(a).is_integer() else f"r_a{a}"
        m = json.loads((ROOT / "results" / "m3" / tag / "steering_metrics.json").read_text())
        true_r_effects[a] = {
            "refusal_ben": m["refusal_rate_ben"],
            "h_readout_harm": m["h_readout_mean_harm"],
            "delta_refusal_ben": m["refusal_rate_ben"] - baseline_refusal_ben,
            "delta_h_readout_harm": m["h_readout_mean_harm"] - baseline_h_readout_harm,
        }

    summary = {"n_random_r_site_per_alpha": 30, "baseline": {"refusal_rate_ben": baseline_refusal_ben, "h_readout_mean_harm": baseline_h_readout_harm}}
    for a, cells in per_alpha.items():
        n = len(cells)
        rf = [c["refusal_rate_ben"] for c in cells]
        hh = [c["h_readout_harm"] for c in cells]
        alpha_sigma_r = cells[0]["alpha_sigma_r"]
        d_rf_mean = mean(rf) - baseline_refusal_ben
        d_rf_std = stdev(rf) if n >= 2 else 0.0
        d_hh_mean = mean(hh) - baseline_h_readout_harm
        d_hh_std = stdev(hh) if n >= 2 else 0.0
        true_r = true_r_effects.get(a, {})
        z_refusal_ben = ((true_r["delta_refusal_ben"] - d_rf_mean) / d_rf_std) if d_rf_std > 0 else float("inf")
        z_h_readout = ((true_r["delta_h_readout_harm"] - d_hh_mean) / d_hh_std) if d_hh_std > 0 else float("inf")
        summary[f"alpha_raw={a}"] = {
            "n_random_r_site": n,
            "alpha_sigma_r": round(alpha_sigma_r, 3),
            "random_r_site_refusal_ben": {"mean": round(mean(rf), 4), "std": round(d_rf_std, 4), "delta_from_baseline_mean": round(d_rf_mean, 4)},
            "random_r_site_h_readout_harm": {"mean": round(mean(hh), 4), "std": round(d_hh_std, 4), "delta_from_baseline_mean": round(d_hh_mean, 4)},
            "true_r_effect": true_r,
            "z_score_true_r_vs_random_r_site_refusal_ben": round(z_refusal_ben, 2),
            "z_score_true_r_vs_random_r_site_h_readout_harm": round(z_h_readout, 2),
        }

    (OUT_DIR / "r_site_specificity_analysis.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
