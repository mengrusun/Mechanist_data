#!/usr/bin/env python3
"""Post-hoc sigma_proj rescaling of existing M3 steering results.

The original M3 sweep used alpha values in raw direction-norm units:
  alpha in {-2, -1, -0.5, 0, +0.5, +1, +2}

The mechanism audit (verify/C3_causal_steering_dissociation/main_experiment_audit/
MECHANISM_AUDIT.md) requires alpha to be reported in sigma_proj units for
proper interpretation. This script:

  1. Reads all 28 existing per-condition steering_metrics.json files.
  2. Computes each condition's alpha in sigma_proj units:
       alpha_sigma = alpha_raw * ||direction|| / sigma_proj
       (equivalent: alpha_sigma = alpha_raw * (||d|| / sigma_proj))
  3. Emits a merged CSV with both alpha_raw and alpha_sigma columns.
  4. Writes a per-direction dose-response summary in sigma_proj units.

No new experiment is launched here — this is pure post-processing.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main():
    root = Path("/data/zhenqian/Reproduction1/mechanica/safety/encode_harmfulness_refusal")
    m3_dir = root / "results" / "m3"
    prep_meta = json.loads((root / "results" / "m_prep" / "directions.json").read_text())

    # Direction norms and sigma_proj come from m_prep.
    norm_h = float(prep_meta["best_h"]["direction_norm"])   # 2.948791
    norm_r = float(prep_meta["best_r"]["direction_norm"])   # 3.707737
    sigma_h = float(prep_meta["best_h"]["sigma_proj"])       # 1.579241
    sigma_r = float(prep_meta["best_r"]["sigma_proj"])       # 1.964109
    # For random/swap directions run at h's site, the ratio ||d|| / sigma_proj
    # matches the acted-upon direction. "swap" adds r-direction at h's site,
    # so its raw norm is ||r|| but the readout is projected onto h (with sigma_h
    # as the readout-side dispersion). For random, we matched-norm to h, so ||d||=||h||.
    # sigma_proj we use for interpretation is sigma_h (readout scale).

    rows = []
    for cell_dir in sorted(m3_dir.iterdir()):
        if not cell_dir.is_dir():
            continue
        m_path = cell_dir / "steering_metrics.json"
        if not m_path.exists():
            continue
        m = json.loads(m_path.read_text())
        direction = m["direction"]
        alpha_raw = float(m["alpha"])
        direction_norm = float(m["direction_norm"])
        if direction == "h":
            sigma_used = sigma_h
        elif direction == "r":
            sigma_used = sigma_r
        elif direction == "random":
            # Random matched-norm to h at h's site; readout is h at h's layer.
            sigma_used = sigma_h
        elif direction == "swap":
            # r at h's site; readout is h at h's layer; interpret alpha in
            # target-readout (h) sigma units.
            sigma_used = sigma_h
        else:
            sigma_used = sigma_h
        # alpha in sigma_proj units:
        # steering shift onto readout direction ≈ alpha_raw * ||d|| * cos(d, readout).
        # For matched-direction (h steering + h readout), ⟨h, h/||h||⟩ = ||h||, so shift = alpha_raw * ||h||.
        # This shift divided by sigma_proj_of_readout gives alpha in sigma_proj units.
        alpha_sigma = alpha_raw * direction_norm / sigma_used
        rows.append({
            "direction": direction,
            "alpha_raw": alpha_raw,
            "alpha_sigma_proj": alpha_sigma,
            "direction_norm": direction_norm,
            "sigma_proj_readout": sigma_used,
            "h_readout_mean_harm": m["h_readout_mean_harm"],
            "h_readout_mean_ben": m["h_readout_mean_ben"],
            "refusal_rate_harm": m["refusal_rate_harm"],
            "refusal_rate_ben": m["refusal_rate_ben"],
            "mean_logp_completion_harm": m["mean_logp_completion_harm"],
            "rep_rate_harm": m["rep_rate_harm"],
        })

    out_dir = root / "runs" / "iteration_round_1"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "m3_rescaled_sigma_proj.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Per-direction dose-response summary in sigma_proj units.
    summary = {}
    for d in ("h", "r", "random", "swap"):
        cells = [r for r in rows if r["direction"] == d]
        cells.sort(key=lambda x: x["alpha_sigma_proj"])
        summary[d] = {
            "n_cells": len(cells),
            "alpha_sigma_range": [cells[0]["alpha_sigma_proj"], cells[-1]["alpha_sigma_proj"]] if cells else None,
            "orders_of_magnitude_span": (max(abs(c["alpha_sigma_proj"]) for c in cells if c["alpha_sigma_proj"] != 0.0) /
                                          min(abs(c["alpha_sigma_proj"]) for c in cells if c["alpha_sigma_proj"] != 0.0))
                                          if len([c for c in cells if c["alpha_sigma_proj"] != 0.0]) >= 2 else None,
            "curve": [{"alpha_sigma": c["alpha_sigma_proj"],
                        "h_readout_harm": c["h_readout_mean_harm"],
                        "refusal_ben": c["refusal_rate_ben"],
                        "refusal_harm": c["refusal_rate_harm"]} for c in cells],
        }
    (out_dir / "m3_rescaled_sigma_proj_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote {out_csv}")
    print(f"Wrote {out_dir / 'm3_rescaled_sigma_proj_summary.json'}")
    print(f"h alpha_sigma range: {summary['h']['alpha_sigma_range']}")
    print(f"r alpha_sigma range: {summary['r']['alpha_sigma_range']}")
    print(f"h orders_of_magnitude_span: {summary['h']['orders_of_magnitude_span']}")
    print(f"r orders_of_magnitude_span: {summary['r']['orders_of_magnitude_span']}")


if __name__ == "__main__":
    main()
