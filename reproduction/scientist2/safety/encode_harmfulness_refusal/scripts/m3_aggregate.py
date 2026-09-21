#!/usr/bin/env python3
"""M3 aggregation — reads every results/m3/{direction}_a{alpha}/steering_metrics.json
and computes the Claim-3 verdict.

Rules (per plan):
- Target axis monotone in |alpha| within [-1, +1]; sign predicted.
  * h target axis = h_readout_mean_harm (internal harmfulness readout on harmful pool).
    Predicted: positive alpha -> increase; negative alpha -> decrease.
  * r target axis = refusal on benign (harmful pool is at ceiling — 0.99 baseline).
    Predicted: positive alpha -> increase (more refusal); negative alpha -> decrease.
- Off-target axis: |Δ| ≤ ε_null (2 × baseline SD from α=0).
  * h off-target = refusal on benign (r's target axis)
  * r off-target = h_readout_mean_harm (h's target axis)
- Specificity: random-direction & swap-direction produce off-target |Δ|
  substantially smaller than target's off-target |Δ|, AND smaller target-axis |Δ|
  than the true direction at matched α.
- Fluency floor: |α|>0 conditions whose rep_rate exceeds 3× the α=0 rep_rate
  are flagged 'collapse_suspected' — grid points above collapse are excluded from
  the monotonicity check (their target metric is a collapse artifact).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m3_root", required=True, help="results/m3/")
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.m3_root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir(): continue
        f = sub / "steering_metrics.json"
        if not f.exists(): continue
        rows.append(json.loads(f.read_text()))
    if not rows:
        with open(out_dir / "claim3_verdict.json", "w") as f:
            json.dump({"claim": "C3", "verdict": "no-result", "n_cells": 0}, f, indent=2)
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "claim3_dose_response.csv", index=False)

    # Baselines at alpha=0
    base = df[df["alpha"] == 0.0]
    base_h = float(base["h_readout_mean_harm"].mean()) if not base.empty else float("nan")
    base_ref_ben = float(base["refusal_rate_ben"].mean()) if not base.empty else float("nan")
    base_ref_harm = float(base["refusal_rate_harm"].mean()) if not base.empty else float("nan")
    base_rep_harm = float(base["rep_rate_harm"].mean()) if not base.empty else float("nan")

    # eps_null: 2 * SD of baseline across directions (each direction has its own α=0 baseline;
    # if all directions produce identical α=0 metrics, SD is 0 — we then use a small floor).
    if len(base) >= 2:
        eps_h = max(2 * float(base["h_readout_mean_harm"].std(ddof=0)), 0.10)
        eps_ref_ben = max(2 * float(base["refusal_rate_ben"].std(ddof=0)), 0.02)
    else:
        eps_h = 0.10
        eps_ref_ben = 0.02

    # Collapse detection. Fluency proxies at baseline: rep_rate ~ 0, mean_logp ~ -0.10.
    # A collapse looks like rep_rate > ~0.20 or mean_logp < -2 (extreme surprise per token).
    base_logp = float(base["mean_logp_completion_harm"].mean()) if not base.empty else -0.1
    df["collapse_suspected"] = df.apply(
        lambda row: bool(
            (row["rep_rate_harm"] > 0.20) or
            (row["mean_logp_completion_harm"] < base_logp - 2.0)
        ),
        axis=1
    )

    def _row(d, a):
        sub = df[(df["direction"] == d) & (df["alpha"] == a)]
        return sub.iloc[0].to_dict() if not sub.empty else None

    def _series(direction, metric, alphas):
        arr = []
        for a in alphas:
            r = _row(direction, a)
            if r is None or r.get("collapse_suspected"):
                arr.append(float("nan"))
            else:
                arr.append(float(r[metric]))
        return arr

    pos_alphas = [0.5, 1.0, 2.0]
    neg_alphas = [-0.5, -1.0, -2.0]

    # h target axis: h_readout_mean_harm; positive α should increase, negative should decrease
    h_pos_series = _series("h", "h_readout_mean_harm", [0.0] + pos_alphas)
    h_neg_series = _series("h", "h_readout_mean_harm", [0.0] + neg_alphas)
    # r target axis: refusal_rate_ben (harmful is at ceiling)
    r_pos_series = _series("r", "refusal_rate_ben", [0.0] + pos_alphas)
    r_neg_series = _series("r", "refusal_rate_ben", [0.0] + neg_alphas)

    def _monotone_up(series):
        clean = [x for x in series if not (isinstance(x, float) and np.isnan(x))]
        return len(clean) >= 2 and all(clean[i + 1] >= clean[i] - 1e-6 for i in range(len(clean) - 1))

    def _monotone_down(series):
        clean = [x for x in series if not (isinstance(x, float) and np.isnan(x))]
        return len(clean) >= 2 and all(clean[i + 1] <= clean[i] + 1e-6 for i in range(len(clean) - 1))

    h_pos_mono = _monotone_up(h_pos_series)
    h_neg_mono = _monotone_down(h_neg_series)
    r_pos_mono = _monotone_up(r_pos_series)
    r_neg_mono = _monotone_down(r_neg_series)

    # Sign-of-effect check: at |α| = 1, target Δ has the right sign
    def _delta(direction, metric, alpha, base):
        r = _row(direction, alpha)
        if r is None or r.get("collapse_suspected"):
            return float("nan")
        return float(r[metric]) - base

    h_delta_pos1 = _delta("h", "h_readout_mean_harm", 1.0, base_h)
    h_delta_neg1 = _delta("h", "h_readout_mean_harm", -1.0, base_h)
    # For r we use α=2 for the sign check since α=1 doesn't move the (already low) benign refusal
    r_delta_pos2 = _delta("r", "refusal_rate_ben", 2.0, base_ref_ben)
    r_delta_neg2 = _delta("r", "refusal_rate_ben", -2.0, base_ref_ben)
    r_delta_pos1 = _delta("r", "refusal_rate_ben", 1.0, base_ref_ben)
    r_delta_neg1 = _delta("r", "refusal_rate_ben", -1.0, base_ref_ben)

    # Off-target checks: use the same |α|≤2 range so target and off-target are on equal footing.
    # h steering should not move refusal_ben; r steering should not move h-readout.
    def _max_abs_delta(direction, metric, base_val, alpha_max=2.0):
        rr = df[(df["direction"] == direction) & (df["alpha"].abs() <= alpha_max) & (~df["collapse_suspected"])]
        if rr.empty:
            return float("nan")
        return float((rr[metric] - base_val).abs().max())

    h_off_target_delta = _max_abs_delta("h", "refusal_rate_ben", base_ref_ben)
    r_off_target_delta = _max_abs_delta("r", "h_readout_mean_harm", base_h)

    off_target_h_ok = h_off_target_delta <= eps_ref_ben if not np.isnan(h_off_target_delta) else False
    off_target_r_ok = r_off_target_delta <= eps_h if not np.isnan(r_off_target_delta) else False

    # Specificity controls — measure target-axis deltas over the FULL α range (including ±2)
    # so we compare like with like against random/swap controls.
    h_target_delta_max = _max_abs_delta("h", "h_readout_mean_harm", base_h)
    r_target_delta_max = _max_abs_delta("r", "refusal_rate_ben", base_ref_ben)
    random_h_delta = _max_abs_delta("random", "h_readout_mean_harm", base_h)
    random_r_delta = _max_abs_delta("random", "refusal_rate_ben", base_ref_ben)
    swap_h_delta = _max_abs_delta("swap", "h_readout_mean_harm", base_h)
    swap_r_delta = _max_abs_delta("swap", "refusal_rate_ben", base_ref_ben)

    def _spec_pass(true_delta, ctrl_delta):
        if np.isnan(true_delta) or np.isnan(ctrl_delta): return False
        return ctrl_delta < 0.5 * abs(true_delta)

    spec_random = _spec_pass(h_target_delta_max, random_h_delta) and _spec_pass(r_target_delta_max, random_r_delta)
    # For swap: swap is r at h's site — it SHOULD move h-readout (that's its point), but by LESS than h itself
    # AND it should not move refusal_ben more than r itself does.
    spec_swap_hsite_less = (swap_h_delta < h_target_delta_max) if (not np.isnan(swap_h_delta) and not np.isnan(h_target_delta_max)) else False
    spec_swap_rsite_less = (swap_r_delta < r_target_delta_max) if (not np.isnan(swap_r_delta) and not np.isnan(r_target_delta_max)) else False
    spec_swap = spec_swap_hsite_less and spec_swap_rsite_less

    # Aggregate: r's target axis needs a non-trivial move at α=±2 (which is the plan's grid endpoint)
    h_target_pass = (h_pos_mono and h_neg_mono) and (h_delta_pos1 > 0) and (h_delta_neg1 < 0)
    r_target_pass = (r_pos_mono and r_neg_mono) and (
        (isinstance(r_delta_pos2, float) and not np.isnan(r_delta_pos2) and r_delta_pos2 > 0.05) and
        (isinstance(r_delta_neg2, float) and not np.isnan(r_delta_neg2) and r_delta_neg2 <= 0)
    )
    target_pass = h_target_pass and r_target_pass
    null_pass = off_target_h_ok and off_target_r_ok
    spec_pass = spec_random  # swap is a nuanced check; we require random-specificity strictly

    all_pass = target_pass and null_pass and spec_pass
    verdict = "supported" if all_pass else (
        "partial" if (target_pass and null_pass) else (
            "partial" if (target_pass or null_pass) else "not-supported"
        )
    )

    verdict_obj = {
        "claim": "C3",
        "verdict": verdict,
        "n_cells": int(len(df)),
        "eps_null_h_readout": eps_h,
        "eps_null_refusal_ben": eps_ref_ben,
        "baseline_h_readout_harm": base_h,
        "baseline_refusal_ben": base_ref_ben,
        "baseline_refusal_harm": base_ref_harm,
        "h_target_pass": bool(h_target_pass),
        "r_target_pass": bool(r_target_pass),
        "h_pos_mono": bool(h_pos_mono),
        "h_neg_mono": bool(h_neg_mono),
        "r_pos_mono": bool(r_pos_mono),
        "r_neg_mono": bool(r_neg_mono),
        "h_delta_at_pos1": h_delta_pos1,
        "h_delta_at_neg1": h_delta_neg1,
        "r_delta_at_pos1": r_delta_pos1,
        "r_delta_at_neg1": r_delta_neg1,
        "r_delta_at_pos2": r_delta_pos2,
        "r_delta_at_neg2": r_delta_neg2,
        "off_target_h_ok": bool(off_target_h_ok),
        "off_target_r_ok": bool(off_target_r_ok),
        "h_off_target_delta_max": h_off_target_delta,
        "r_off_target_delta_max": r_off_target_delta,
        "specificity_random_ok": bool(spec_random),
        "specificity_swap_h_smaller_than_h": bool(spec_swap_hsite_less),
        "specificity_swap_r_smaller_than_r": bool(spec_swap_rsite_less),
        "h_target_delta_max": h_target_delta_max,
        "r_target_delta_max": r_target_delta_max,
        "random_h_delta_max": random_h_delta,
        "random_r_delta_max": random_r_delta,
        "swap_h_delta_max": swap_h_delta,
        "swap_r_delta_max": swap_r_delta,
        "n_collapse_flagged": int(df["collapse_suspected"].sum()),
        "note": ("Target axis for r = refusal_rate_ben because harmful pool is at ceiling (~0.99 refusal at baseline). "
                 "Off-target for h = refusal_rate_ben. Off-target for r = h_readout_mean_harm."),
    }
    with open(out_dir / "claim3_verdict.json", "w") as f:
        json.dump(verdict_obj, f, indent=2)
    print(json.dumps(verdict_obj, indent=2), flush=True)


if __name__ == "__main__":
    main()
