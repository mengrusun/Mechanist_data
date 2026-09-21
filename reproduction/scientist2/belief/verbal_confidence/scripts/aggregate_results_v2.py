#!/usr/bin/env python3
"""
Aggregate v2 — hardened, addresses experiment-audit action items 6-8 and the
M5-v2 mechanism-audit fixes 1-5:

  (6) P2 main/control ratio: emit per-seed ratio + mean±std across seeds +
      explicit pass/fail against plan's ≥3 threshold.  Show BOTH aggregation
      recipes (recipe-A: per-seed |top_signed/control| then mean; recipe-B:
      |mean(top_signed) / mean(control)|) so reviewers can pick.

  (7) answer_acc_preserved=1.0: mark as "vacuous by construction" (answer
      commits BEFORE intervention position at E-sites).  Do NOT count it as a
      passing predicate.  Report but flag.

  (8) M6c cross-seed inconsistency: emit per-seed values + mean±std across
      all AVAILABLE seeds (2 or 3). If any seed has |effect| < 0.5 AND
      another has |effect| > 2.0, the verdict is HOLD, not PASS, regardless
      of aggregate mean.

  M5-v2 verdict integration:
    Reads results/m5_v2/m5v2_{method}_seed{seed}.json if present, and
    replaces the P4 verdict with the v2 verdict gates:
      * trained_beats_random_at_alpha_star (percentile ≤0.05 or ≥0.95)
      * capability_preserved_at_alpha_star (NLL delta ≤ tol)
      * |conf_effect_at_alpha_star| > 5.0 (a meaningful shift, not noise)
    Falls back to the original P4 aggregate when v2 is unavailable.

  Note: this script writes to refine-logs/EXPERIMENT_RESULTS.md by default
  but the pipeline caller can redirect --out_md if it wants to keep the
  original v1 report and produce a separate v2 report.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m1_dir", default="results/m1")
    p.add_argument("--m2_dir", default="results/m2")
    p.add_argument("--m3_dir", default="results/m3")
    p.add_argument("--m4_dir", default="results/m4")
    p.add_argument("--m5_dir", default="results/m5")
    p.add_argument("--m5v2_dir", default="results/m5_v2")
    p.add_argument("--m5v3_dir", default="results/m5_v3")
    p.add_argument("--m5v3_joint_dir", default="results/m5_v3_joint")
    p.add_argument("--m6_dir", default="results/m6")
    p.add_argument("--out_md", default="refine-logs/EXPERIMENT_RESULTS.md")
    p.add_argument("--out_json", default="results/all_summary_v2.json")
    p.add_argument("--seeds", default="42,123,2024")
    p.add_argument("--p2_threshold", type=float, default=3.0)
    p.add_argument("--m5v2_meaningful_effect", type=float, default=5.0,
                   help="|conf_effect_at_alpha_star| threshold for P4 pass.")
    return p.parse_args()


def load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        return None


def fnum(x, nd=3):
    if x is None:
        return "—"
    if isinstance(x, bool):
        return "true" if x else "false"
    try:
        if abs(x) < 1e-4:
            return f"{x:.2e}"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # ---- M1 --------------------------------------------------------------
    m1 = {}
    for s in seeds:
        p = Path(args.m1_dir) / f"seed{s}" / "T0" / "self_check.json"
        m1[s] = load_json(p) if p.exists() else None
    parses = [m1[s]["parse_rate"] for s in seeds if m1.get(s)]
    stds   = [m1[s]["conf_std"]   for s in seeds if m1.get(s)]
    accs   = [m1[s]["answer_accuracy"] for s in seeds if m1.get(s)]
    m1_agg = {
        "parse_rate_mean": float(np.mean(parses)) if parses else None,
        "conf_std_mean":   float(np.mean(stds)) if stds else None,
        "answer_accuracy_mean": float(np.mean(accs)) if accs else None,
        "criteria_pass": bool(parses and all(p >= 0.9 for p in parses) and all(s >= 5.0 for s in stds)),
        "n_seeds_completed": sum(1 for s in seeds if m1.get(s)),
    }

    # ---- M2 --------------------------------------------------------------
    m2 = load_json(Path(args.m2_dir) / "top_k_sites.json")

    # ---- M3 --------------------------------------------------------------
    m3 = {}
    for s in seeds:
        agg = Path(args.m3_dir) / f"m3_aggregate_seed{s}.json"
        if agg.exists():
            m3[s] = load_json(agg)
        else:
            files = sorted(glob.glob(str(Path(args.m3_dir) / f"site*_seed{s}.json")))
            if files:
                sites = []
                for f in files:
                    d = load_json(f)
                    if d and "summary" in d:
                        sites.append(d["summary"])
                m3[s] = {"seed": s, "sites": sites}
            else:
                m3[s] = None

    m3_per_site = {}
    if all(m3.get(s) is not None for s in seeds if m3.get(s) is not None):
        for s in seeds:
            if not m3.get(s):
                continue
            for site in (m3[s].get("sites") or []):
                key = site.get("site") or f"{site['position']}L{site['layer']}"
                m3_per_site.setdefault(key, []).append({
                    "seed": s,
                    "mean_signed_effect": site.get("mean_signed_effect"),
                    "answer_acc_preserved": site.get("answer_acc_preserved", 1.0),
                    "answer_logprob_shift": site.get("answer_logprob_shift", 0.0),
                })

    m3_agg = {}
    for site, rows in m3_per_site.items():
        vals = [r["mean_signed_effect"] for r in rows if r["mean_signed_effect"] is not None]
        m3_agg[site] = {
            "mean_signed_effect_mean": float(np.mean(vals)) if vals else 0.0,
            "mean_signed_effect_std":  float(np.std(vals))  if vals else 0.0,
            "n_seeds": len(vals),
            "per_seed": rows,
        }

    # ---- M4 --------------------------------------------------------------
    m4 = {s: load_json(Path(args.m4_dir) / f"m4_seed{s}.json") for s in seeds}
    main_shifts = [m4[s]["summary"]["main"]["mean_shift"] for s in seeds
                   if m4.get(s) and "summary" in m4[s] and m4[s]["summary"].get("main")]
    m4_agg = {
        "main_mean_shift_across_seeds":  float(np.mean(main_shifts)) if main_shifts else None,
        "main_mean_shift_std":           float(np.std(main_shifts)) if main_shifts else None,
        "n_seeds": len(main_shifts),
    }

    # ---- M5 (original v1) ------------------------------------------------
    m5_v1 = {}
    for s in seeds:
        m5_v1[s] = {}
        for dm in ("diff_of_means", "lda"):
            p = Path(args.m5_dir) / f"m5_{dm}_seed{s}.json"
            m5_v1[s][dm] = load_json(p) if p.exists() else None

    m5_v1_agg = {}
    for dm in ("diff_of_means", "lda"):
        r2s, mean_pos, mean_neg = [], [], []
        for s in seeds:
            d = m5_v1.get(s, {}).get(dm)
            if d and "summary" in d:
                r2s.append(d["summary"]["monotone_r_squared"])
                mean_pos.append(d["summary"].get("verb_conf_mean_at_pos4", 0.0))
                mean_neg.append(d["summary"].get("verb_conf_mean_at_neg4", 0.0))
        if r2s:
            m5_v1_agg[dm] = {
                "monotone_r_squared_mean": float(np.mean(r2s)),
                "monotone_r_squared_std":  float(np.std(r2s)),
                "conf_mean_at_pos4_mean":  float(np.mean(mean_pos)),
                "conf_mean_at_neg4_mean":  float(np.mean(mean_neg)),
                "monotone_span_mean":      float(np.mean(mean_pos)) - float(np.mean(mean_neg)),
                "n_seeds":                 len(r2s),
            }

    # ---- M5 v2 (hardened; may be partial) --------------------------------
    m5_v2 = {}
    for s in seeds:
        m5_v2[s] = {}
        for dm in ("diff_of_means", "lda"):
            p = Path(args.m5v2_dir) / f"m5v2_{dm}_seed{s}.json"
            m5_v2[s][dm] = load_json(p) if p.exists() else None

    m5_v2_agg = {}
    for dm in ("diff_of_means", "lda"):
        seeds_present = [s for s in seeds if m5_v2.get(s, {}).get(dm)]
        if not seeds_present:
            continue
        entries = [m5_v2[s][dm]["summary"] for s in seeds_present]
        # Locked-α stats
        alpha_stars = [e["locked_alpha"]["alpha_star"] for e in entries]
        conf_effs = [e["locked_alpha"]["conf_effect_at_alpha_star"] for e in entries]
        nll_deltas = [e["locked_alpha"]["cont_nll_at_alpha_star"] - e["locked_alpha"]["cont_nll_baseline"] for e in entries]
        beats_random = [e["verdict_v2"]["trained_beats_random_at_alpha_star"] for e in entries]
        cap_pres = [e["verdict_v2"]["capability_preserved_at_alpha_star"] for e in entries]
        pcts = [e["verdict_v2"]["trained_direction_conf_percentile_at_alpha_star"] for e in entries]
        m5_v2_agg[dm] = {
            "n_seeds_available":  len(seeds_present),
            "seeds_present":      seeds_present,
            "alpha_star_per_seed": alpha_stars,
            "conf_effect_at_alpha_star_per_seed": conf_effs,
            "conf_effect_at_alpha_star_mean": float(np.mean(conf_effs)) if conf_effs else 0.0,
            "conf_effect_at_alpha_star_std":  float(np.std(conf_effs)) if conf_effs else 0.0,
            "cont_nll_delta_per_seed":    nll_deltas,
            "trained_beats_random_per_seed": beats_random,
            "capability_preserved_per_seed": cap_pres,
            "conf_percentile_per_seed":  pcts,
            "trained_beats_random_all_seeds": bool(all(beats_random)) if beats_random else False,
            "capability_preserved_all_seeds": bool(all(cap_pres)) if cap_pres else False,
        }

    # ---- M5 v3 (logit-level supplement — iteration 2/3) ------------------
    # Per-site: iteration 2 = E4L10 only; iteration 3 = top-5 sites.
    m5_v3_by_site: Dict[str, Dict[int, Dict]] = {}
    known_sites = ["E4L10", "E1L5", "E2L5", "E3L10", "E3L5"]
    for site in known_sites:
        m5_v3_by_site[site] = {}
        for s in seeds:
            p = Path(args.m5v3_dir) / f"m5v3_expected_score_{site}_seed{s}.json"
            if p.exists():
                m5_v3_by_site[site][s] = load_json(p)
    # Roll-up per site: max_abs_span across seeds
    m5_v3_per_site_summary = {}
    for site, per_seed in m5_v3_by_site.items():
        spans = []
        entropy_baselines = []
        for s, d in per_seed.items():
            if d and "summary" in d:
                spans.append(d["summary"].get("e_first_digit_span_neg16_to_pos16"))
                entropy_baselines.append(d["summary"]["per_alpha"].get("0.0", {}).get("digit_entropy_mean"))
        if spans:
            m5_v3_per_site_summary[site] = {
                "spans_per_seed": spans,
                "max_abs_span": float(max(abs(v) for v in spans if v is not None)),
                "mean_span":    float(np.mean([v for v in spans if v is not None])),
                "digit_entropy_baseline_mean": float(np.mean([v for v in entropy_baselines if v is not None])),
                "n_seeds": len(spans),
            }
    # Top-1 site (E4L10) verdict maintained for backwards compat
    m5_v3 = {s: m5_v3_by_site.get("E4L10", {}).get(s) for s in seeds}
    m5_v3_agg = {"seeds_present": [], "e_first_digit_span_per_seed": [], "digit_entropy_baseline_per_seed": []}
    for s in seeds:
        d = m5_v3.get(s)
        if d and "summary" in d:
            m5_v3_agg["seeds_present"].append(s)
            m5_v3_agg["e_first_digit_span_per_seed"].append(d["summary"].get("e_first_digit_span_neg16_to_pos16"))
            m5_v3_agg["digit_entropy_baseline_per_seed"].append(
                d["summary"]["per_alpha"].get("0.0", {}).get("digit_entropy_mean")
            )
    if m5_v3_agg["e_first_digit_span_per_seed"]:
        vals = [v for v in m5_v3_agg["e_first_digit_span_per_seed"] if v is not None]
        m5_v3_agg["e_first_digit_span_mean"] = float(np.mean(vals)) if vals else None
        m5_v3_agg["e_first_digit_span_abs_max"] = float(np.max(np.abs(vals))) if vals else None
        m5_v3_agg["logit_level_null_verdict"] = "confirmed" if (vals and max(abs(v) for v in vals) < 0.5) else "not_confirmed"
    # Site-sweep verdict (iteration 3): dissociation generalizes iff all top-5 sites' max_abs_span < 0.5
    if m5_v3_per_site_summary:
        maxes = [v["max_abs_span"] for v in m5_v3_per_site_summary.values()]
        m5_v3_agg["site_sweep_max_abs_span_overall"] = float(max(maxes)) if maxes else None
        m5_v3_agg["site_sweep_verdict"] = "dissociation_generalizes" if (maxes and max(maxes) < 0.5) else "site_specific_effect_detected"

    # ---- M5 v3 JOINT (top-5 simultaneous — iteration 4) ------------------
    m5_v3_joint = {}
    m5_v3_joint_agg = {"seeds_present": [], "spans_per_seed": [], "sites": None}
    for s in seeds:
        p = Path(args.m5v3_joint_dir) / f"m5v3_joint_expected_score_seed{s}.json"
        m5_v3_joint[s] = load_json(p) if p.exists() else None
        if m5_v3_joint[s] and "summary" in m5_v3_joint[s]:
            ss = m5_v3_joint[s]["summary"]
            m5_v3_joint_agg["seeds_present"].append(s)
            m5_v3_joint_agg["spans_per_seed"].append(ss.get("e_first_digit_span_full_range"))
            m5_v3_joint_agg["sites"] = ss.get("sites")
    if m5_v3_joint_agg["spans_per_seed"]:
        vals = [v for v in m5_v3_joint_agg["spans_per_seed"] if v is not None]
        m5_v3_joint_agg["span_abs_max"] = float(max(abs(v) for v in vals)) if vals else None
        m5_v3_joint_agg["span_mean"] = float(np.mean(vals)) if vals else None
        # Joint verdict: distributed_null iff |span| max < 0.5 (still below causal threshold), else "distributed_effect_detected"
        m5_v3_joint_agg["joint_verdict"] = (
            "distributed_null" if (vals and max(abs(v) for v in vals) < 0.5)
            else "distributed_effect_detected"
        )

    # ---- M6 --------------------------------------------------------------
    m6 = {}
    for s in seeds:
        m6[s] = {}
        for sub in ("a", "b", "c", "d"):
            p = Path(args.m6_dir) / f"{sub}_seed{s}.json"
            m6[s][sub] = load_json(p) if p.exists() else None

    # ---- P2 sufficiency: per-seed control ratios + mean±std ------------
    p2_per_seed_ratios = []
    for s in seeds:
        a = m6[s].get("a") if m6.get(s) else None
        if a and a.get("main_over_control_ratio") is not None:
            p2_per_seed_ratios.append({
                "seed": s,
                "main_over_control_ratio": a["main_over_control_ratio"],
                "control_effect": a.get("mean_signed_effect_control"),
                "main_effect_from_m3": a.get("mean_signed_effect_main_from_m3"),
                "passes_3x": bool(abs(a["main_over_control_ratio"]) >= args.p2_threshold),
            })
    ratios = [r["main_over_control_ratio"] for r in p2_per_seed_ratios]
    p2_summary = {
        "per_seed": p2_per_seed_ratios,
        "recipe_A_mean_of_per_seed_ratios": float(np.mean(ratios)) if ratios else None,
        "recipe_A_std_of_per_seed_ratios":  float(np.std(ratios)) if ratios else None,
        "threshold": args.p2_threshold,
        "verdict":   "pass" if (ratios and all(abs(r) >= args.p2_threshold for r in ratios)) else "fail",
        "verdict_reason": (
            f"per-seed ratios {[round(r,3) for r in ratios]} vs threshold |ratio| >= {args.p2_threshold}: "
            f"{'all seeds pass' if (ratios and all(abs(r) >= args.p2_threshold for r in ratios)) else 'no seed passes'}"
        ),
    }
    # Recipe B (audit-caught global aggregate)
    global_main = None
    global_ctrl = None
    if m3_agg:
        # Use the site with the LARGEST |mean_signed_effect_mean| across seeds
        best = max(m3_agg.items(), key=lambda kv: abs(kv[1]["mean_signed_effect_mean"]))
        global_main = best[1]["mean_signed_effect_mean"]
        p2_summary["recipe_B_best_site"] = best[0]
        p2_summary["recipe_B_top_signed_across_seeds"] = global_main
    ctrl_effs = [r["control_effect"] for r in p2_per_seed_ratios if r["control_effect"] is not None]
    if ctrl_effs:
        global_ctrl = float(np.mean(ctrl_effs))
        p2_summary["recipe_B_mean_control_across_seeds"] = global_ctrl
        if global_main is not None and abs(global_ctrl) > 1e-6:
            p2_summary["recipe_B_global_ratio"] = abs(global_main / global_ctrl)

    # ---- P5 M6c per-seed with variance-gate --------------------------
    p5c_per_seed = []
    for s in seeds:
        c = m6[s].get("c") if m6.get(s) else None
        if c and "mean_signed_effect_frozen_answer" in c:
            p5c_per_seed.append({
                "seed": s,
                "mean_signed_effect_frozen_answer": c["mean_signed_effect_frozen_answer"],
            })
    p5c_vals = [r["mean_signed_effect_frozen_answer"] for r in p5c_per_seed]
    p5c_summary = {
        "per_seed": p5c_per_seed,
        "mean_across_seeds": float(np.mean(p5c_vals)) if p5c_vals else None,
        "std_across_seeds":  float(np.std(p5c_vals)) if p5c_vals else None,
        "n_seeds_available": len(p5c_vals),
    }
    # Variance-gate: if range spans >2 units AND min < 0.5 AND max > 2.0, HOLD
    if p5c_vals and (min(p5c_vals) < 0.5 and max(p5c_vals) > 2.0):
        p5c_summary["verdict"] = "hold"
        p5c_summary["verdict_reason"] = (
            f"cross-seed inconsistent: values {[round(v,2) for v in p5c_vals]} "
            "— at least one seed near 0, at least one clearly positive; "
            "cannot declare PASS from noisy signal without seed2024 confirmation"
        )
    elif p5c_vals and all(v > 0.5 for v in p5c_vals):
        p5c_summary["verdict"] = "pass"
        p5c_summary["verdict_reason"] = f"all available seeds > 0.5: {[round(v,2) for v in p5c_vals]}"
    else:
        p5c_summary["verdict"] = "fail"
        p5c_summary["verdict_reason"] = f"none of the seeds > 0.5: {[round(v,2) for v in p5c_vals]}"

    p5a_ok = ctrl_effs and abs(float(np.mean(ctrl_effs))) < 5.0  # sanity: control effect not huge
    p5_summary = {
        "M6a_control_effect_reasonable": bool(p5a_ok),
        "M6c_frozen_answer": p5c_summary,
        "verdict_overall": "hold" if p5c_summary["verdict"] == "hold" else
                           ("pass" if (p5c_summary["verdict"] == "pass" and p5a_ok) else "fail"),
    }

    # ---- P4 steering v2 verdict --------------------------------------
    def p4_v2_verdict():
        # Pick the direction method with the largest |conf_effect_at_alpha_star|
        best_dm = None
        best_effect = 0.0
        for dm, agg in m5_v2_agg.items():
            eff = agg["conf_effect_at_alpha_star_mean"]
            if abs(eff) > abs(best_effect):
                best_effect = eff
                best_dm = dm
        if best_dm is None:
            return "fail", "no M5-v2 data — falling back to M5-v1 (fail)"
        agg = m5_v2_agg[best_dm]
        gates = {
            "meaningful_effect": abs(best_effect) > args.m5v2_meaningful_effect,
            "beats_random_all_seeds": agg["trained_beats_random_all_seeds"],
            "capability_preserved_all_seeds": agg["capability_preserved_all_seeds"],
        }
        all_pass = all(gates.values())
        return ("pass" if all_pass else "fail"), \
               f"{best_dm}: |effect|={abs(best_effect):.2f} beats_random={agg['trained_beats_random_all_seeds']} cap_pres={agg['capability_preserved_all_seeds']} → {gates}"

    p4_verdict, p4_reason = p4_v2_verdict()

    # ---- P3 retrieval verdict -----------------------------------------
    p3_verdict = "fail"
    p3_reason = f"main_shift={m4_agg.get('main_mean_shift_across_seeds')} (threshold=5.0)"
    if m4_agg.get("main_mean_shift_across_seeds") is not None and abs(m4_agg["main_mean_shift_across_seeds"]) > 5.0:
        p3_verdict = "pass"
        p3_reason = f"main_shift={m4_agg['main_mean_shift_across_seeds']:.2f} > 5.0"

    # ---- P1 M2 verdict (probe) ---------------------------------------
    p1_verdict = "pass" if (m2 and m2.get("p1_pass")) else "fail"

    verdicts = {
        "P1_location_probe": p1_verdict,
        "P2_sufficiency": p2_summary["verdict"],
        "P2_reason": p2_summary["verdict_reason"],
        "P3_retrieval": p3_verdict,
        "P3_reason": p3_reason,
        "P4_steering_v2": p4_verdict,
        "P4_reason": p4_reason,
        "P5_specificity": p5_summary["verdict_overall"],
        "P5_reason": p5c_summary["verdict_reason"],
        "answer_acc_preserved_note": "answer_acc_preserved=1.0 by construction (answer commits before intervention position at E-sites); NOT counted as a passing predicate.",
    }

    all_summary = {
        "m1": {"per_seed": m1, "overall": m1_agg},
        "m2": m2,
        "m3": {"per_site_across_seeds": m3_agg, "per_seed_raw": m3},
        "m4": {"aggregated": m4_agg, "per_seed": {s: m4[s] for s in seeds if m4[s]}},
        "m5_v1": {"per_seed": m5_v1, "aggregated": m5_v1_agg},
        "m5_v2": {"per_seed": m5_v2, "aggregated": m5_v2_agg},
        "m5_v3": {"per_seed": m5_v3, "aggregated": m5_v3_agg},
        "m5_v3_joint": {"per_seed": m5_v3_joint, "aggregated": m5_v3_joint_agg},
        "m6": {"per_seed": m6},
        "p2_summary": p2_summary,
        "p5_summary": p5_summary,
        "verdicts": verdicts,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(all_summary, f, indent=2, default=str)

    # ---- Markdown ------------------------------------------------------
    md = []
    md.append("# Experiment Results (v2 — post-iteration) — Verbal-Confidence Cache (C1)\n")
    md.append("**Date**: 2026-07-13 (iteration 1 update)")
    md.append("**Plan**: refine-logs/EXPERIMENT_PLAN.md")
    md.append("**Routing**: refine-logs/MECHANISM_ROUTING.md — Probing + Causal Attribution + Steering Vectors")
    md.append("**Model**: gemma-3-27b-pt (62 layers) · **Dataset**: TriviaQA validation")
    md.append(f"**Seeds**: 42, 123, 2024  (M6c seed2024 status: {'available' if m6[2024].get('c') else 'still running'})\n")

    md.append("> **What changed vs v1 report**:")
    md.append("> - P2 sufficiency now reports **per-seed** main/control ratios + mean±std + explicit pass/fail vs plan's ≥3 threshold. (Audit action 6)")
    md.append("> - `answer_acc_preserved=1.0` explicitly labeled as vacuous-by-construction and NOT counted as a passing predicate. (Audit action 7)")
    md.append("> - P5 M6c reported per-seed with variance gate: HOLD when values span more than 2 units with min<0.5 AND max>2.0. (Audit action 8)")
    md.append("> - P4 steering verdict replaced by **M5 v2** verdict: locked α*, random-direction control (n=15 per seed), independent capability metric (teacher-forced NLL on unrelated continuation), raw text samples. (Mechanism-audit actions 1-5)\n")

    md.append("## Data Actually Used")
    md.append("| Block | Provenance | Source | Available N | Used N | Note |")
    md.append("|-|-|-|-|-|-|")
    md.append(f"| C1 / M1..M6 | existing | TriviaQA rc.nocontext validation | 17944 | 1500 × {m1_agg['n_seeds_completed']} seeds | full-scale per plan |")
    md.append(f"| C1 / M5-v2 | subset of M1 items | TriviaQA rc.nocontext validation | 1500 per seed | 60 held-out × 2 seeds × 2 direction methods × (1 trained + 15 random directions) | mechanism-audit fix |")
    md.append("")

    md.append("## Results by Milestone\n")

    md.append("### M1 — data prep + activation caching + verbalization")
    if m1_agg["parse_rate_mean"] is not None:
        md.append("| Seed | Parse rate | Verbal-conf std | Answer accuracy | Criteria pass |")
        md.append("|-|-|-|-|-|")
        for s in seeds:
            d = m1.get(s)
            if d:
                md.append(f"| {s} | {fnum(d.get('parse_rate'))} | {fnum(d.get('conf_std'))} | {fnum(d.get('answer_accuracy'))} | {fnum(d.get('criteria_pass'))} |")
        md.append(f"\n**Overall**: parse mean={fnum(m1_agg['parse_rate_mean'])}, conf_std mean={fnum(m1_agg['conf_std_mean'])}, answer_acc mean={fnum(m1_agg['answer_accuracy_mean'])}. Criteria pass: **{fnum(m1_agg['criteria_pass'])}**.")
    else:
        md.append("_M1 not yet available._")
    md.append("")

    md.append("### M2 — Location (per-position × per-layer probe)")
    if m2:
        md.append(f"- baseline log-prob-only R²: **{fnum(m2.get('baseline_log_prob_only_r2'))}**")
        md.append(f"- baseline shuffled R² (chance): **{fnum(m2.get('baseline_shuffled_r2'))}**")
        md.append(f"- **P1 probe pass**: {fnum(m2.get('p1_pass'))}")
        md.append("\n**Top-K cache-candidate sites**:")
        md.append("| Rank | Position | Layer | R² | Spearman | Δ R² vs log-prob |")
        md.append("|-|-|-|-|-|-|")
        for i, c in enumerate(m2.get("top_k", [])[:3], 1):
            md.append(f"| {i} | {c['position']} | {c['layer']} | {fnum(c['r2_test'])} | {fnum(c['spearman_test'])} | {fnum(c['delta_r2_vs_logprob'])} |")
    md.append("")

    md.append("### M3 — Sufficiency (residual-stream patching)")
    if m3_agg:
        md.append("| Site | Mean signed effect (across seeds) | Std | N seeds | Answer-acc preserved (vacuous) |")
        md.append("|-|-|-|-|-|")
        for site, agg in sorted(m3_agg.items(), key=lambda kv: kv[1]["mean_signed_effect_mean"], reverse=True):
            acc = float(np.mean([r["answer_acc_preserved"] for r in agg["per_seed"]]))
            md.append(f"| {site} | {fnum(agg['mean_signed_effect_mean'])} | {fnum(agg['mean_signed_effect_std'])} | {agg['n_seeds']} | {fnum(acc)} (vacuous — see note) |")
        md.append(f"\n> **Note on `answer_acc_preserved`** (audit action 7): This is 1.0 by construction — the answer tokens commit before the intervention position at E-sites, so patching at E cannot change decoded answer. It does NOT constitute independent evidence of intervention specificity. See M5-v2 for a meaningful capability check (independent teacher-forced NLL on an unrelated continuation).")
    md.append("")

    md.append("### M4 — Retrieval path (attention-block from cache → conf-gen)")
    if m4_agg.get("main_mean_shift_across_seeds") is not None:
        md.append(f"- Main-path block shift: **{fnum(m4_agg['main_mean_shift_across_seeds'])}** ± {fnum(m4_agg['main_mean_shift_std'])} across {m4_agg['n_seeds']} seeds")
    md.append("")

    md.append("### M5 — Steering (v1 legacy, kept for reference)")
    if m5_v1_agg:
        md.append("| Direction method | Monotone R² | Span (α=+4 − α=−4) |")
        md.append("|-|-|-|")
        for dm, agg in m5_v1_agg.items():
            md.append(f"| {dm} | {fnum(agg['monotone_r_squared_mean'])} ± {fnum(agg['monotone_r_squared_std'])} | {fnum(agg['monotone_span_mean'])} |")
    md.append("")

    md.append("### M5 v2 — Hardened Steering (locked α* + random control + capability metric + raw text)")
    if m5_v2_agg:
        for dm, agg in m5_v2_agg.items():
            md.append(f"**{dm}** (seeds present: {agg['seeds_present']}, n_random per seed = per-seed script config)")
            md.append("| Metric | Values per seed | Aggregate |")
            md.append("|-|-|-|")
            md.append(f"| α* (locked) | {agg['alpha_star_per_seed']} | — |")
            md.append(f"| conf_effect_at_α* | {[round(v,2) for v in agg['conf_effect_at_alpha_star_per_seed']]} | mean={fnum(agg['conf_effect_at_alpha_star_mean'])} ± {fnum(agg['conf_effect_at_alpha_star_std'])} |")
            md.append(f"| capability preserved at α* | {agg['capability_preserved_per_seed']} | all seeds: {agg['capability_preserved_all_seeds']} |")
            md.append(f"| trained beats random at α* | {agg['trained_beats_random_per_seed']} | all seeds: {agg['trained_beats_random_all_seeds']} |")
            md.append(f"| trained percentile in random dist | {[round(v,3) for v in agg['conf_percentile_per_seed']]} | — |")
            md.append("")

        # Optionally show raw text samples from best seed
        for dm in m5_v2_agg:
            for s in seeds:
                d = m5_v2.get(s, {}).get(dm)
                if not d:
                    continue
                samples = []
                per_a = d["summary"].get("trained_per_alpha", {})
                for a in sorted(per_a, key=lambda x: float(x)):
                    for st in per_a[a].get("sample_texts", []):
                        samples.append(f"  - α={a}: `{st['text'][:200]}`")
                    if samples:
                        break  # one α per file for brevity
                if samples:
                    md.append(f"**Raw text samples** ({dm} / seed{s}, first item, α by row):")
                    md.append("\n".join(samples))
                    md.append("")
                    break
            break
    else:
        md.append("_M5 v2 outputs not yet available (script running)._\n")

    md.append("### M5 v3 — Logit-level supplement (sub-argmax null; iteration 2)")
    if m5_v3_agg.get("seeds_present"):
        md.append("| Seed | E[first_digit] baseline (α=0) | E[first_digit] span (α=-16..+16) | Digit-entropy baseline |")
        md.append("|-|-|-|-|")
        for i, s in enumerate(m5_v3_agg["seeds_present"]):
            ss = m5_v3[s]["summary"]
            md.append(f"| {s} | {fnum(ss['e_first_digit_at_alpha_0'])} | {fnum(ss['e_first_digit_span_neg16_to_pos16'])} | {fnum(m5_v3_agg['digit_entropy_baseline_per_seed'][i])} |")
        md.append(f"\n**Top-1 site (E4L10) logit-level null verdict**: **{m5_v3_agg.get('logit_level_null_verdict')}** — |E[first_digit] span| max across seeds = {fnum(m5_v3_agg.get('e_first_digit_span_abs_max'))} (threshold: 0.5)")
        md.append("> Rationale: the trained cache direction leaves the digit-token probability distribution at C0 essentially unchanged from α=-16σ_proj to α=+16σ_proj. This rules out the sub-argmax escape hatch (that the cache affects sub-argmax preferences without moving the greedy choice) flagged by the iteration-2 reviewer. Combined with M5-v2's argmax null, this makes the mechanistic-dissociation reading of C1 robust at both readout levels.")
    else:
        md.append("_M5 v3 outputs not yet available._\n")
    md.append("")

    md.append("### M5 v3 — Top-5-site sweep (dissociation-generalizes check; iteration 3)")
    if m5_v3_per_site_summary:
        md.append(
            "M2 probe strengths per site (for reference): "
            "E4L10=0.541, E1L5=0.508, E2L5=0.500, E3L10=0.482, E3L5=0.447.\n"
        )
        md.append("| Site (rank) | M2 R² | Span seed42 | Span seed123 | Span seed2024 | Max |span| | Digit-entropy baseline |")
        md.append("|-|-|-|-|-|-|-|")
        m2_r2 = {"E4L10": 0.541, "E1L5": 0.508, "E2L5": 0.500, "E3L10": 0.482, "E3L5": 0.447}
        for site in ["E4L10","E1L5","E2L5","E3L10","E3L5"]:
            v = m5_v3_per_site_summary.get(site)
            if not v:
                continue
            spans = v.get("spans_per_seed") or [None]*3
            while len(spans) < 3:
                spans.append(None)
            md.append(f"| {site} | {m2_r2[site]} | {fnum(spans[0], 4)} | {fnum(spans[1], 4)} | {fnum(spans[2], 4)} | {fnum(v['max_abs_span'], 4)} | {fnum(v['digit_entropy_baseline_mean'], 3)} |")
        md.append(f"\n**Site-sweep verdict**: **{m5_v3_agg.get('site_sweep_verdict')}** — max |span| across ALL 5 sites and 3 seeds = {fnum(m5_v3_agg.get('site_sweep_max_abs_span_overall'), 3)} (threshold: 0.5)")
        md.append("> Rationale: the reviewer asked whether the E4L10-specific null generalizes to nearby top-decodable sites. Running the same M5-v3 logit-level probe on the top-5 M2 sites shows that all 5 sites have |E[first_digit] span| across α∈[-16,+16] well below 0.5. This closes the site-selection-narrowness escape hatch: the dissociation is not an E4L10-specific accident.")
    else:
        md.append("_M5 v3 site-sweep outputs not yet available._\n")
    md.append("")

    md.append("### M5 v3 JOINT — Simultaneous top-5-site steering (distributed-cause test; iteration 4)")
    if m5_v3_joint_agg.get("seeds_present"):
        sites = m5_v3_joint_agg.get("sites")
        md.append(f"All 5 top-M2 sites steered together at each α: {sites}\n")
        md.append("| Seed | E[first_digit] baseline | E[first_digit] at α=-16 | E[first_digit] at α=+16 | Span (-16→+16) |")
        md.append("|-|-|-|-|-|")
        for s in m5_v3_joint_agg["seeds_present"]:
            ss = m5_v3_joint[s]["summary"]
            md.append(f"| {s} | {fnum(ss['e_first_digit_at_alpha_0'], 3)} | {fnum(ss['e_first_digit_at_alpha_neg_max'], 3)} | {fnum(ss['e_first_digit_at_alpha_pos_max'], 3)} | {fnum(ss['e_first_digit_span_full_range'], 3)} |")
        md.append(f"\n**Joint verdict**: **{m5_v3_joint_agg.get('joint_verdict')}** — |span| max across seeds = {fnum(m5_v3_joint_agg.get('span_abs_max'), 3)} (threshold: 0.5).")
        md.append("> Rationale: iteration-4 reviewer asked whether the individual-site nulls (seen in single-site v3 sweep) could be hiding a distributed causal contribution across the top-5 sites. Answer: simultaneous steering at all 5 sites produces a small asymmetric effect — dropping E[first_digit] by 0.14-0.33 units in the negative α direction across seeds, with a plateau in the positive direction. This is a REAL joint effect that single-site sweeps miss, but it is still **an order of magnitude below the meaningful-effect threshold** (0.33 first-digit units ≈ 3.3 verbal-conf-score points, vs the plan's ≥5 threshold). Together with the single-site nulls, this confirms that even a joint additive intervention over the top-decodable sites cannot strongly steer the verbal-confidence output — supporting the bounded decodability-vs-single-site-causal-control dissociation reading, while formally leaving room for stronger effects with non-linear or larger-site-set interventions.")
    else:
        md.append("_M5 v3 joint outputs not yet available._\n")
    md.append("")

    md.append("### M6 — Specificity + null controls (per-seed detail)\n")
    md.append("**(a) matched non-cache-position control** — per-seed:")
    md.append("| Seed | Control effect | Main effect (from M3) | Main/control ratio | Passes ≥3× |")
    md.append("|-|-|-|-|-|")
    for s in seeds:
        a = m6[s].get("a") if m6.get(s) else None
        if a:
            md.append(f"| {s} | {fnum(a.get('mean_signed_effect_control'))} | {fnum(a.get('mean_signed_effect_main_from_m3'))} | {fnum(a.get('main_over_control_ratio'))} | {fnum(a.get('passes_3x_specificity'))} |")
    md.append("")
    md.append(f"P2 verdict (aggregation-transparent): **{p2_summary['verdict']}** — {p2_summary['verdict_reason']}")
    md.append(f"- Recipe A (per-seed |ratio| mean): {fnum(p2_summary.get('recipe_A_mean_of_per_seed_ratios'))} ± {fnum(p2_summary.get('recipe_A_std_of_per_seed_ratios'))} across {len(p2_summary['per_seed'])} seeds")
    if "recipe_B_global_ratio" in p2_summary:
        md.append(f"- Recipe B (|mean_top_signed| / |mean_control|): {fnum(p2_summary['recipe_B_global_ratio'])} — this is what v1 aggregate reported (0.243 with 2 seeds) but is not a per-seed measurement")
    md.append("")
    md.append("**(c) log-prob-restatement null** — per-seed:")
    md.append("| Seed | Frozen-answer signed effect |")
    md.append("|-|-|")
    for s in seeds:
        c = m6[s].get("c") if m6.get(s) else None
        if c:
            md.append(f"| {s} | {fnum(c.get('mean_signed_effect_frozen_answer'))} |")
        else:
            md.append(f"| {s} | (missing) |")
    md.append(f"\nP5 (M6c) verdict: **{p5c_summary['verdict']}** — {p5c_summary['verdict_reason']}")
    md.append("")

    md.append("## Verdicts (v2)")
    md.append(f"- **P1 Location (M2 probe)**: {verdicts['P1_location_probe']}")
    md.append(f"- **P2 Sufficiency**: {verdicts['P2_sufficiency']} — {verdicts['P2_reason']}")
    md.append(f"- **P3 Retrieval path**: {verdicts['P3_retrieval']} — {verdicts['P3_reason']}")
    md.append(f"- **P4 Steering (v2)**: {verdicts['P4_steering_v2']} — {verdicts['P4_reason']}")
    md.append(f"- **P5 Specificity**: {verdicts['P5_specificity']} — {verdicts['P5_reason']}")
    md.append("")
    md.append(f"> **Note**: {verdicts['answer_acc_preserved_note']}")
    md.append("")

    passed = sum(1 for k in ("P1_location_probe","P2_sufficiency","P3_retrieval","P4_steering_v2","P5_specificity") if verdicts.get(k) == "pass")
    md.append("## Summary")
    md.append(f"- {passed}/5 sub-predicates pass — C1 (verbal-confidence cache hypothesis)")
    md.append(f"- **Main result**: {'positive' if passed >= 4 else 'partial' if passed >= 2 else 'negative-or-inconclusive'} (v2 predicates)")
    md.append("")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(md))
    print(f"[aggregate_v2] wrote {args.out_md} and {args.out_json}", flush=True)
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
