#!/usr/bin/env python3
"""Aggregate M2 + M3 + M4 + M5 + M6 outputs into a single results JSON that the
EXPERIMENT_RESULTS.md report is generated from.  Also computes:

  - C1 verdict: max probe cv_acc per V, projection-transfer β + p at the picked layer
  - C2 verdict: for each decorrelator, off-diagonal probe accuracy vs raw baseline;
                LEACE off-diag drop, preservation ratio, norm audit.
  - C3 verdict: per (V, decorrelator, site) dose-response Spearman rho, sign of the
                +2sigma shift, the inversion point (smallest |alpha|<0 that flips the
                sign of V's baseline effect), coherence gate outcomes.
  - C4 verdict: from M5 matrix — max_off / min_diag, permutation p.
  - Ablations (M6): B1 raw vs pure comparison, B2 random baseline null, B3 mean-cent,
                B4 directional ablation on LEACE (should behave as strong negative alpha).

Writes a `results_summary.json` (machine-readable) and prints a human-readable summary
that the caller pipes into EXPERIMENT_RESULTS.md.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np


def load_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def spearman(xs, ys):
    from scipy.stats import spearmanr
    if len(xs) < 3:
        return float("nan"), float("nan")
    r, p = spearmanr(xs, ys)
    return float(r), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/M_main_v1")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = args.root
    art = os.path.join(root, "artifacts")

    result = dict()

    # ---- M2: layer_pick, probe_accuracy, baseline_transfer, projection_transfer ----
    pick = load_json(os.path.join(art, "m2", "layer_pick.json"), {})
    probe = load_json(os.path.join(art, "m2", "probe_accuracy.json"), {})
    baseline = load_json(os.path.join(art, "m2", "baseline_transfer.json"), {})
    proj_transfer = load_json(os.path.join(art, "m2", "projection_transfer.json"), {})

    result["m2"] = dict(
        layer_pick=pick,
        best_probe_acc={V: max(r["cv_acc"] for r in probe[V]) if V in probe else None
                         for V in ["G", "A", "I", "M"]},
        best_probe_held={V: max(r["held_acc"] for r in probe[V]) if V in probe else None
                         for V in ["G", "A", "I", "M"]},
        baseline_transfer=baseline,
        projection_transfer_at_ell_star={
            V: proj_transfer.get(V, {}).get(str(pick[V]["ell_star"])) if V in pick else None
            for V in ["G", "A", "I", "M"]
        },
    )
    # C1 verdict
    c1_probe_pass = {V: r >= 0.80 for V, r in result["m2"]["best_probe_acc"].items()
                     if r is not None}
    c1_ptransfer_pass = {}
    for V in ["G", "A", "I", "M"]:
        e = result["m2"]["projection_transfer_at_ell_star"].get(V)
        c1_ptransfer_pass[V] = (e is not None) and (e["p"] < 0.05)
    result["c1"] = dict(
        probe_pass=c1_probe_pass,
        projection_transfer_pass=c1_ptransfer_pass,
        overall_pass=all(c1_probe_pass.values()) and all(c1_ptransfer_pass.values()),
        best_probe_acc=result["m2"]["best_probe_acc"],
    )

    # ---- M3: leakage matrix, preservation, norm audit ----
    leakage = load_json(os.path.join(art, "m3", "leakage_matrix.json"), {})
    preservation = load_json(os.path.join(art, "m3", "preservation.json"), {})
    norm_audit = load_json(os.path.join(art, "m3", "norm_audit.json"), {})
    # Build a 4x4 dict per decorrelator.
    leak_matrices = {}
    for method in ["raw", "mean_centered", "gs", "leace"]:
        grid = {}
        for row in leakage.get(method, []):
            grid[(row[0], row[1])] = row[2]
        leak_matrices[method] = grid
    def diag_mean(m):
        return float(np.mean([m.get((V, V), 0.5) for V in ["G", "A", "I", "M"]]))
    def off_max(m):
        return float(max((m.get((V, W), 0.5) for V in ["G", "A", "I", "M"]
                          for W in ["G", "A", "I", "M"] if V != W), default=0.5))
    result["m3"] = dict(
        leakage_matrix=leakage,
        preservation=preservation,
        norm_audit=norm_audit,
        diag_mean={m: diag_mean(leak_matrices[m]) for m in leak_matrices},
        max_offdiag={m: off_max(leak_matrices[m]) for m in leak_matrices},
    )
    # C2 verdict: for LEACE and GS, does off-diag drop to <= chance+0.05 (0.55) while
    # diagonal retains >= 0.95x raw diagonal?
    raw_diag = leak_matrices["raw"]
    c2_result = {}
    for method in ["mean_centered", "gs", "leace"]:
        m = leak_matrices[method]
        offs = [m.get((V, W), 0.5) for V in ["G", "A", "I", "M"]
                for W in ["G", "A", "I", "M"] if V != W]
        diags = [m.get((V, V), 0.5) for V in ["G", "A", "I", "M"]]
        raw_diags = [raw_diag.get((V, V), 0.5) for V in ["G", "A", "I", "M"]]
        preserve = all(d >= 0.95 * rd for d, rd in zip(diags, raw_diags))
        offdiag_ok = max(offs) <= 0.55
        c2_result[method] = dict(
            max_offdiag=float(max(offs)),
            min_diag=float(min(diags)),
            preserve_pass=bool(preserve),
            offdiag_pass=bool(offdiag_ok),
        )
    result["c2"] = c2_result

    # ---- M4: per-grid-point steer results ----
    m4_files = sorted(glob.glob(os.path.join(art, "m4", "*.json")))
    m4_by_key = {}
    for f in m4_files:
        if f.endswith("summary.json"):
            continue
        d = load_json(f)
        if d is None:
            continue
        key = (d["V"], d["decorrelator"], d["site"], d["alpha_mult"])
        m4_by_key[key] = d
    # Per (V, decorrelator, site): dose-response Spearman on (alpha_mult, v_effect)
    dose_response = {}
    for (V, dec, site) in {(k[0], k[1], k[2]) for k in m4_by_key}:
        vs, ys, coh_ok, cohs = [], [], [], []
        for am in sorted({k[3] for k in m4_by_key if k[:3] == (V, dec, site)}):
            d = m4_by_key.get((V, dec, site, am))
            if d is None:
                continue
            vs.append(am)
            ys.append(d["v_effect"])
            cohs.append(d["coherence"])
            coh_ok.append(d["coherence"]["format_ok_rate"] >= 0.8
                          and d["coherence"]["mean_5gram_rep"] <= 0.5)
        rho, p = spearman(vs, ys)
        # +2sigma shift
        d_plus2 = m4_by_key.get((V, dec, site, 2))
        d_zero = m4_by_key.get((V, dec, site, 0))
        d_neg = None
        for am in [-1, -2, -4]:
            d = m4_by_key.get((V, dec, site, am))
            if d is not None:
                d_neg = d if d_neg is None else d_neg
        shift_plus2 = (d_plus2["v_effect"] - d_zero["v_effect"]) if d_plus2 and d_zero else None
        baseline_eff = (result["m2"]["baseline_transfer"].get(V, {}).get("baseline_effect")
                        if isinstance(result["m2"]["baseline_transfer"], dict) else None)
        # C3 25%-of-baseline test: shift_plus2 should be >= 0.25 * |baseline_eff|, same sign.
        c3_shift_ok = False
        if shift_plus2 is not None and baseline_eff is not None and abs(baseline_eff) > 0:
            c3_shift_ok = ((shift_plus2 * baseline_eff > 0)
                           and (abs(shift_plus2) >= 0.25 * abs(baseline_eff)))
        # Inversion: smallest |alpha_mult|<0 that flips sign of v_effect vs baseline
        inversion_alpha = None
        for am in sorted({k[3] for k in m4_by_key if k[:3] == (V, dec, site) and k[3] < 0},
                          reverse=True):  # start with -1
            d = m4_by_key.get((V, dec, site, am))
            if d is None or baseline_eff is None:
                continue
            if d["v_effect"] * baseline_eff < 0:
                inversion_alpha = am
                break
        dose_response[f"{V}|{dec}|{site}"] = dict(
            V=V, decorrelator=dec, site=site,
            spearman_rho=rho, spearman_p=p,
            baseline_effect=baseline_eff,
            shift_at_alpha_plus2=shift_plus2,
            c3_25pct_pass=c3_shift_ok,
            inversion_alpha_mult=inversion_alpha,
            spearman_pass=(abs(rho) >= 0.7) if not np.isnan(rho) else False,
            coherence_ok_per_alpha=list(zip(vs, coh_ok)),
        )
    result["m4"] = dict(dose_response=dose_response,
                         n_grid_points=len(m4_by_key))
    # C3 verdict per V (using LEACE + single as the headline; also GS as backup)
    c3_pass_by_V = {}
    for V in ["G", "A", "I", "M"]:
        for dec in ["leace", "gs"]:
            d = dose_response.get(f"{V}|{dec}|single")
            if d is None:
                continue
            # C3 supported when shift@+2sigma passes AND at least one negative alpha
            # inverts sign AND spearman rho >= 0.7 up to saturation.
            passed = d["c3_25pct_pass"] and (d["inversion_alpha_mult"] is not None) and d["spearman_pass"]
            c3_pass_by_V.setdefault(V, {})[dec] = dict(
                pass_=bool(passed),
                shift=d["shift_at_alpha_plus2"],
                inversion=d["inversion_alpha_mult"],
                rho=d["spearman_rho"],
            )
    result["c3"] = c3_pass_by_V

    # ---- M5: selectivity matrix ----
    sel = load_json(os.path.join(root, "artifacts", "m5", "selectivity_matrix.json"))
    perm = load_json(os.path.join(root, "artifacts", "m5", "permutation_test.json"))
    result["m5"] = dict(matrix=sel, perm_test=perm)
    if sel:
        result["c4"] = dict(
            max_off_over_min_diag=sel.get("c4_ratio_max_off_over_min_diag"),
            max_offdiag=sel.get("max_offdiag"),
            min_diag=sel.get("min_diag"),
            perm_p=perm.get("p_value") if perm else None,
            pass_ratio=(sel.get("c4_ratio_max_off_over_min_diag", 999) <= 0.30),
            pass_perm=(perm and perm.get("p_value", 1) <= 0.05),
        )

    # ---- M6: ablations ----
    m6_files = sorted(glob.glob(os.path.join(art, "m6", "*.json")))
    m6_summary = load_json(os.path.join(art, "m6", "summary.json"), [])
    m6_by_hook = defaultdict(list)
    for f in m6_files:
        if f.endswith("summary.json"):
            continue
        d = load_json(f)
        if d is None:
            continue
        m6_by_hook[(d["hook_kind"], d["decorrelator"])].append(d)
    result["m6"] = {f"{hk}|{dec}": [dict(V=d["V"], alpha=d["alpha_mult"], seed=d.get("seed"),
                                          v_effect=d["v_effect"],
                                          coh=d["coherence"]["format_ok_rate"])
                                     for d in v]
                     for (hk, dec), v in m6_by_hook.items()}

    if args.out is None:
        args.out = os.path.join(root, "results_summary.json")
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(json.dumps({k: v for k, v in result.items()
                       if k in ("c1", "c2", "c3", "c4")}, indent=2, default=str))
    print(f"\n[aggregate] wrote {args.out}")


if __name__ == "__main__":
    main()
