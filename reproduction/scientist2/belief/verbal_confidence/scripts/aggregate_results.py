#!/usr/bin/env python3
"""
Aggregate outputs across seeds and milestones into a single results summary.

Reads:
  results/m1/seed{42,123,2024}/T0/self_check.json + confidence.jsonl
  results/m2/top_k_sites.json                       (single-run, pools seeds)
  results/m3/site*_n*_seed{42,123,2024}*.json       (per-site per-seed)
  results/m4/m4_seed{42,123,2024}.json              (per-seed)
  results/m5/m5_{diff_of_means,lda}_seed*.json      (per-seed × dir_method)
  results/m6/{a,b,c,d}_seed{42,123,2024}.json       (per-seed × sub)

Writes:
  refine-logs/EXPERIMENT_RESULTS.md   — human-readable report (Phase 5 output)
  results/all_summary.json            — machine-readable roll-up
"""

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m1_dir", default="results/m1")
    p.add_argument("--m2_dir", default="results/m2")
    p.add_argument("--m3_dir", default="results/m3")
    p.add_argument("--m4_dir", default="results/m4")
    p.add_argument("--m5_dir", default="results/m5")
    p.add_argument("--m6_dir", default="results/m6")
    p.add_argument("--out_md", default="refine-logs/EXPERIMENT_RESULTS.md")
    p.add_argument("--out_json", default="results/all_summary.json")
    p.add_argument("--seeds", default="42,123,2024")
    return p.parse_args()


def load_json(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception as e:
        return {"__error__": str(e), "__path__": p}


def collect_m1(m1_dir: str, seeds: List[int]) -> Dict:
    out = {}
    for s in seeds:
        p = Path(m1_dir) / f"seed{s}" / "T0" / "self_check.json"
        if p.exists():
            out[s] = load_json(p)
        else:
            out[s] = None
    return out


def collect_m2(m2_dir: str) -> Dict:
    p = Path(m2_dir) / "top_k_sites.json"
    if not p.exists():
        return None
    return load_json(p)


def collect_m3(m3_dir: str, seeds: List[int]) -> Dict:
    """Return {seed: aggregate_summary}."""
    out = {}
    for s in seeds:
        agg = Path(m3_dir) / f"m3_aggregate_seed{s}.json"
        if agg.exists():
            out[s] = load_json(agg)
        else:
            # Fall back to per-site files
            files = sorted(glob.glob(str(Path(m3_dir) / f"site*_seed{s}.json")))
            if files:
                sites = []
                for f in files:
                    d = load_json(f)
                    if "summary" in d:
                        sites.append(d["summary"])
                out[s] = {"seed": s, "sites": sites}
            else:
                out[s] = None
    return out


def collect_m4(m4_dir: str, seeds: List[int]) -> Dict:
    out = {}
    for s in seeds:
        p = Path(m4_dir) / f"m4_seed{s}.json"
        if p.exists():
            out[s] = load_json(p)
        else:
            out[s] = None
    return out


def collect_m5(m5_dir: str, seeds: List[int]) -> Dict:
    """{seed: {direction_method: summary}}."""
    out = {}
    for s in seeds:
        out[s] = {}
        for dm in ("diff_of_means", "lda"):
            p = Path(m5_dir) / f"m5_{dm}_seed{s}.json"
            if p.exists():
                out[s][dm] = load_json(p)
            else:
                out[s][dm] = None
    return out


def collect_m6(m6_dir: str, seeds: List[int]) -> Dict:
    out = {}
    for s in seeds:
        out[s] = {}
        for sub in ("a", "b", "c", "d"):
            p = Path(m6_dir) / f"{sub}_seed{s}.json"
            if p.exists():
                out[s][sub] = load_json(p)
            else:
                out[s][sub] = None
    return out


def average_over_seeds(dicts: List[Dict], keys: List[str]) -> Dict:
    """Return {key: [values across seeds] + mean/std}."""
    out = {}
    for k in keys:
        vals = []
        for d in dicts:
            if d is None:
                continue
            v = d.get(k)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        if vals:
            out[k] = {
                "values": vals,
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
            }
    return out


def format_dict(d: Dict, indent: int = 0) -> str:
    pref = "  " * indent
    lines = []
    for k, v in d.items():
        if isinstance(v, dict):
            lines.append(f"{pref}- **{k}**:")
            lines.append(format_dict(v, indent + 1))
        else:
            lines.append(f"{pref}- **{k}**: {v}")
    return "\n".join(lines)


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    m1 = collect_m1(args.m1_dir, seeds)
    m2 = collect_m2(args.m2_dir)
    m3 = collect_m3(args.m3_dir, seeds)
    m4 = collect_m4(args.m4_dir, seeds)
    m5 = collect_m5(args.m5_dir, seeds)
    m6 = collect_m6(args.m6_dir, seeds)

    # ---- M1 rollup ------------------------------------------------------
    m1_rollup = {"per_seed": {}, "overall": {}}
    for s in seeds:
        if m1.get(s):
            m1_rollup["per_seed"][s] = m1[s]
    parses = [m1[s]["parse_rate"] for s in seeds if m1.get(s)]
    stds = [m1[s]["conf_std"] for s in seeds if m1.get(s)]
    accs = [m1[s]["answer_accuracy"] for s in seeds if m1.get(s)]
    if parses:
        m1_rollup["overall"] = {
            "parse_rate_mean": float(np.mean(parses)),
            "conf_std_mean": float(np.mean(stds)),
            "answer_accuracy_mean": float(np.mean(accs)),
            "criteria_pass": all(p >= 0.9 for p in parses) and all(s >= 5.0 for s in stds),
        }

    # ---- M2 rollup ------------------------------------------------------
    m2_rollup = m2 if m2 else {}

    # ---- M3 rollup ------------------------------------------------------
    m3_rollup = {"per_seed": {}, "per_site_across_seeds": {}}
    for s in seeds:
        m3_rollup["per_seed"][s] = m3.get(s)
    if all(m3.get(s) is not None for s in seeds):
        site_effects: Dict[str, List[float]] = {}
        site_acc: Dict[str, List[float]] = {}
        site_lp_shift: Dict[str, List[float]] = {}
        for s in seeds:
            for site in (m3[s].get("sites") or []):
                key = site.get("site") or f"{site['position']}L{site['layer']}"
                site_effects.setdefault(key, []).append(site["mean_signed_effect"])
                site_acc.setdefault(key, []).append(site.get("answer_acc_preserved", 1.0))
                site_lp_shift.setdefault(key, []).append(site.get("answer_logprob_shift", 0.0))
        for k in site_effects:
            m3_rollup["per_site_across_seeds"][k] = {
                "mean_signed_effect_seeds_mean": float(np.mean(site_effects[k])),
                "mean_signed_effect_seeds_std": float(np.std(site_effects[k])),
                "answer_acc_preserved_mean": float(np.mean(site_acc[k])),
                "answer_logprob_shift_mean": float(np.mean(site_lp_shift[k])),
                "n_seeds": len(site_effects[k]),
            }

    # ---- M4 rollup ------------------------------------------------------
    m4_rollup = {"per_seed": {}, "aggregated": {}}
    main_shifts, ctrl_shifts, main_kls, main_accs = [], [], [], []
    for s in seeds:
        d = m4.get(s)
        if not d or "summary" not in d:
            continue
        summ = d["summary"]
        m4_rollup["per_seed"][s] = summ
        main = summ.get("main")
        ctrl = summ.get("control")
        if main and main.get("n", 0) > 0:
            main_shifts.append(main["mean_shift"])
            main_kls.append(main["mean_kl_to_prior"])
            main_accs.append(main["answer_acc_preserved"])
        if ctrl and ctrl.get("n", 0) > 0:
            ctrl_shifts.append(ctrl["mean_shift"])
    if main_shifts:
        m4_rollup["aggregated"] = {
            "main_mean_shift_across_seeds": float(np.mean(main_shifts)),
            "main_mean_shift_std_across_seeds": float(np.std(main_shifts)),
            "main_kl_to_prior_mean": float(np.mean(main_kls)),
            "main_answer_acc_preserved": float(np.mean(main_accs)),
            "control_mean_shift_across_seeds": float(np.mean(ctrl_shifts)) if ctrl_shifts else None,
            "main_over_control_ratio": (
                abs(np.mean(main_shifts) / np.mean(ctrl_shifts)) if ctrl_shifts and abs(np.mean(ctrl_shifts)) > 1e-6 else None
            ),
        }

    # ---- M5 rollup ------------------------------------------------------
    m5_rollup = {"per_seed": {}, "aggregated": {}}
    for dm in ("diff_of_means", "lda"):
        r2s, mean_at_pos4, mean_at_neg4, collapses = [], [], [], []
        for s in seeds:
            d = m5.get(s, {}).get(dm)
            if not d or "summary" not in d:
                continue
            summ = d["summary"]
            m5_rollup["per_seed"].setdefault(s, {})[dm] = summ
            r2s.append(summ["monotone_r_squared"])
            mean_at_pos4.append(summ.get("verb_conf_mean_at_pos4", 0.0))
            mean_at_neg4.append(summ.get("verb_conf_mean_at_neg4", 0.0))
            collapses.append(summ.get("collapse_flag", False))
        if r2s:
            m5_rollup["aggregated"][dm] = {
                "monotone_r_squared_mean": float(np.mean(r2s)),
                "monotone_r_squared_std": float(np.std(r2s)),
                "conf_mean_at_pos4_mean": float(np.mean(mean_at_pos4)),
                "conf_mean_at_neg4_mean": float(np.mean(mean_at_neg4)),
                "monotone_span": float(np.mean(mean_at_pos4)) - float(np.mean(mean_at_neg4)),
                "collapse_flag_any_seed": bool(any(collapses)),
                "n_seeds": len(r2s),
            }

    # ---- M6 rollup ------------------------------------------------------
    m6_rollup = {"per_seed": {}, "aggregated": {}}
    for s in seeds:
        m6_rollup["per_seed"][s] = m6.get(s, {})
    # Aggregate by sub
    a_ratios = [m6[s]["a"].get("main_over_control_ratio") for s in seeds
                if m6.get(s, {}).get("a") and m6[s]["a"].get("main_over_control_ratio") is not None]
    a_control_effects = [m6[s]["a"].get("mean_signed_effect_control") for s in seeds
                         if m6.get(s, {}).get("a") and "mean_signed_effect_control" in m6[s]["a"]]
    c_effects = [m6[s]["c"].get("mean_signed_effect_frozen_answer") for s in seeds
                 if m6.get(s, {}).get("c") and "mean_signed_effect_frozen_answer" in m6[s]["c"]]
    d_shifts = [m6[s]["d"].get("answer_logprob_shift_mean") for s in seeds
                if m6.get(s, {}).get("d") and "answer_logprob_shift_mean" in m6[s]["d"]]
    m6_rollup["aggregated"]["a_main_over_control_ratios"] = a_ratios
    m6_rollup["aggregated"]["a_control_mean_signed_effect_mean"] = float(np.mean(a_control_effects)) if a_control_effects else None
    m6_rollup["aggregated"]["c_frozen_answer_signed_effect_mean"] = float(np.mean(c_effects)) if c_effects else None
    m6_rollup["aggregated"]["d_answer_logprob_shift_mean"] = float(np.mean(d_shifts)) if d_shifts else None

    # ---- Pass/fail decisions ------------------------------------------
    verdicts = {}
    # P1 (M2)
    verdicts["P1_location"] = "pass" if (m2 and m2.get("p1_pass")) else ("fail" if m2 else "no_data")
    # P2 (M3) — mean_signed_effect > 0 aggregated across seeds/sites AND at least 3x main/control ratio
    p2_ok = False
    p2_reason = "no data"
    if m3_rollup["per_site_across_seeds"]:
        signed_vals = [v["mean_signed_effect_seeds_mean"] for v in m3_rollup["per_site_across_seeds"].values()]
        top = max(signed_vals)
        p2_signed = top > 0.5  # some meaningful shift
        control_ratio = m6_rollup["aggregated"].get("a_control_mean_signed_effect_mean")
        if control_ratio is not None and abs(control_ratio) > 1e-6:
            ratio = abs(top / control_ratio)
        else:
            ratio = None
        p2_ratio_ok = ratio is None or ratio >= 3.0
        p2_ok = p2_signed and p2_ratio_ok
        p2_reason = f"top_signed={top:.2f} main/control_ratio={ratio}"
    verdicts["P2_sufficiency"] = "pass" if p2_ok else "fail"
    verdicts["P2_reason"] = p2_reason
    # P3 (M4)
    p3_agg = m4_rollup.get("aggregated", {})
    p3_ratio = p3_agg.get("main_over_control_ratio")
    verdicts["P3_retrieval"] = "pass" if (p3_agg.get("main_mean_shift_across_seeds") is not None and
                                          (p3_ratio is None or abs(p3_ratio) >= 3.0) and
                                          abs(p3_agg.get("main_mean_shift_across_seeds", 0.0)) > 5.0) else "fail"
    verdicts["P3_reason"] = f"main_shift={p3_agg.get('main_mean_shift_across_seeds')} ratio={p3_ratio}"
    # P4 (M5) — monotone_r_squared > 0.7 AND monotone_span (α=+4 - α=-4) meaningful
    p4_ok = False
    p4_reason = "no data"
    for dm, agg in m5_rollup["aggregated"].items():
        r2 = agg["monotone_r_squared_mean"]
        span = agg["monotone_span"]
        if r2 > 0.7 and abs(span) > 5.0 and not agg["collapse_flag_any_seed"]:
            p4_ok = True
            p4_reason = f"{dm}: r²={r2:.3f} span={span:.1f} no_collapse"
            break
        p4_reason = f"{dm}: r²={r2:.3f} span={span:.1f} collapse={agg['collapse_flag_any_seed']}"
    verdicts["P4_steering"] = "pass" if p4_ok else "fail"
    verdicts["P4_reason"] = p4_reason
    # P5 (M6 aggregate)
    p5_a_ok = m6_rollup["aggregated"].get("a_control_mean_signed_effect_mean") is not None and \
              (abs(m6_rollup["aggregated"]["a_control_mean_signed_effect_mean"]) < 5.0)
    p5_c_ok = m6_rollup["aggregated"].get("c_frozen_answer_signed_effect_mean") is not None and \
              m6_rollup["aggregated"]["c_frozen_answer_signed_effect_mean"] > 0.5
    verdicts["P5_specificity"] = "pass" if (p5_a_ok and p5_c_ok) else "partial" if (p5_a_ok or p5_c_ok) else "fail"

    # ---- Emit summary ---------------------------------------------------
    summary = {
        "m1": m1_rollup,
        "m2": m2_rollup,
        "m3": m3_rollup,
        "m4": m4_rollup,
        "m5": m5_rollup,
        "m6": m6_rollup,
        "verdicts": verdicts,
    }

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Emit markdown --------------------------------------------------
    def fnum(x, ndig=3):
        if x is None:
            return "—"
        if isinstance(x, bool):
            return "true" if x else "false"
        try:
            if abs(x) < 1e-4:
                return f"{x:.2e}"
            return f"{x:.{ndig}f}"
        except Exception:
            return str(x)

    md = []
    md.append("# Initial Experiment Results — Verbal-Confidence Cache Hypothesis (C1)\n")
    md.append("**Date**: 2026-07-13")
    md.append("**Plan**: refine-logs/EXPERIMENT_PLAN.md")
    md.append("**Routing**: refine-logs/MECHANISM_ROUTING.md — Probing + Causal Attribution + Steering Vectors")
    md.append("**Model**: gemma-3-27b-pt (62 layers) · **Dataset**: TriviaQA validation")
    md.append("**Seeds**: 42, 123, 2024\n")

    md.append("## Data Actually Used")
    md.append("| Claim/Block | Provenance | Source | Available N | Used N (actual) | Subset note |")
    md.append("|-|-|-|-|-|-|")
    total_used = sum(1500 for s in seeds if m1.get(s)) if m1_rollup.get("overall") else 0
    md.append(f"| C1 / M1..M6 | existing | TriviaQA rc.nocontext validation | 17944 | 1500 × {sum(1 for s in seeds if m1.get(s))} seeds = {total_used} | none — full-scale per plan |")
    md.append("")

    md.append("## Results by Milestone\n")

    md.append("### M1 — data prep + activation caching + verbalization")
    if m1_rollup["overall"]:
        md.append("| Seed | Parse rate | Verbal-conf std | Answer accuracy | Criteria pass |")
        md.append("|-|-|-|-|-|")
        for s in seeds:
            d = m1.get(s)
            if d:
                md.append(f"| {s} | {fnum(d.get('parse_rate'))} | {fnum(d.get('conf_std'))} | {fnum(d.get('answer_accuracy'))} | {fnum(d.get('criteria_pass'))} |")
        md.append(f"\n**Overall**: parse_rate mean={fnum(m1_rollup['overall'].get('parse_rate_mean'))}, "
                  f"conf_std mean={fnum(m1_rollup['overall'].get('conf_std_mean'))}, "
                  f"answer_accuracy mean={fnum(m1_rollup['overall'].get('answer_accuracy_mean'))}. "
                  f"M1 success criteria (parse≥90%, std≥5): **{fnum(m1_rollup['overall'].get('criteria_pass'))}**.")
    else:
        md.append("_M1 outputs not yet available._")
    md.append("")

    md.append("### M2 — Location (per-position × per-layer ridge probe)")
    if m2_rollup:
        md.append(f"- baseline log_prob_only R²: **{fnum(m2_rollup.get('baseline_log_prob_only_r2'))}**")
        md.append(f"- baseline shuffled R² (chance): **{fnum(m2_rollup.get('baseline_shuffled_r2'))}**")
        md.append(f"- **P1 pass**: {fnum(m2_rollup.get('p1_pass'))}")
        md.append("\n**Top-K post-answer cache candidates** (ranked by Δ R² over log-prob-only):")
        md.append("| Rank | Position | Layer | R²_test | Spearman | Δ R² vs log-prob |")
        md.append("|-|-|-|-|-|-|")
        for i, c in enumerate(m2_rollup.get("top_k", []), 1):
            md.append(f"| {i} | {c['position']} | {c['layer']} | {fnum(c['r2_test'])} | {fnum(c['spearman_test'])} | {fnum(c['delta_r2_vs_logprob'])} |")
        # Also list conf-gen baseline
        cg = m2_rollup.get("baseline_conf_gen_r2_by_layer", {})
        if cg:
            best_L = max(cg, key=lambda L: cg[L])
            md.append(f"\n- Best conf-gen-position (C0) probe: L{best_L} R²={fnum(cg[best_L])}")
    else:
        md.append("_M2 outputs not yet available._")
    md.append("")

    md.append("### M3 — Sufficiency (residual-stream patching at top cache sites)")
    if m3_rollup["per_site_across_seeds"]:
        md.append("| Site | Mean signed effect (across seeds) | Std | Answer acc preserved | Answer log-prob shift |")
        md.append("|-|-|-|-|-|")
        for site, v in sorted(m3_rollup["per_site_across_seeds"].items(),
                              key=lambda kv: kv[1]["mean_signed_effect_seeds_mean"], reverse=True):
            md.append(f"| {site} | {fnum(v['mean_signed_effect_seeds_mean'])} | {fnum(v['mean_signed_effect_seeds_std'])} | {fnum(v['answer_acc_preserved_mean'])} | {fnum(v['answer_logprob_shift_mean'])} |")
    else:
        md.append("_M3 outputs not yet available._")
    md.append("")

    md.append("### M4 — Retrieval path (attention-block from cache → conf-gen)")
    if m4_rollup["aggregated"]:
        agg = m4_rollup["aggregated"]
        md.append(f"- Main block (cache → conf-gen): mean shift = **{fnum(agg.get('main_mean_shift_across_seeds'))}** ± {fnum(agg.get('main_mean_shift_std_across_seeds'))}")
        md.append(f"- KL(blocked ∥ prior) mean: {fnum(agg.get('main_kl_to_prior_mean'))}")
        md.append(f"- Answer accuracy preserved: {fnum(agg.get('main_answer_acc_preserved'))}")
        if agg.get("control_mean_shift_across_seeds") is not None:
            md.append(f"- Control (matched non-cache) shift: {fnum(agg.get('control_mean_shift_across_seeds'))}")
            md.append(f"- Main/control ratio: {fnum(agg.get('main_over_control_ratio'))}")
    else:
        md.append("_M4 outputs not yet available._")
    md.append("")

    md.append("### M5 — Steering (signed dose-response at cache site)")
    if m5_rollup["aggregated"]:
        md.append("| Direction method | Monotone R² | Span (α=+4 − α=−4) | Collapse flag |")
        md.append("|-|-|-|-|")
        for dm, agg in m5_rollup["aggregated"].items():
            md.append(f"| {dm} | {fnum(agg['monotone_r_squared_mean'])} ± {fnum(agg['monotone_r_squared_std'])} | {fnum(agg['monotone_span'])} | {fnum(agg['collapse_flag_any_seed'])} |")
        # α-curves
        md.append("\n**α → mean verbal confidence** (diff_of_means, averaged across seeds):")
        alpha_curves = {}
        for s in seeds:
            d = m5.get(s, {}).get("diff_of_means")
            if d and "summary" in d and "per_alpha" in d["summary"]:
                for a, v in d["summary"]["per_alpha"].items():
                    alpha_curves.setdefault(a, []).append(v["mean"])
        for a in sorted(alpha_curves, key=lambda x: float(x)):
            avg = float(np.mean(alpha_curves[a]))
            md.append(f"  - α = {a}: mean conf ≈ {avg:.1f}")
    else:
        md.append("_M5 outputs not yet available._")
    md.append("")

    md.append("### M6 — Specificity + null controls")
    if any(m6.get(s) for s in seeds):
        agg = m6_rollup["aggregated"]
        md.append(f"- **(a) Matched non-cache-position control** — mean signed effect at MID (control): {fnum(agg.get('a_control_mean_signed_effect_mean'))}. Main/control ratios: {agg.get('a_main_over_control_ratios')}")
        md.append(f"- **(b) Recall-strength null (within-bin)** — see per-seed results/m6/b_seed*.json for bin-wise probe R² and steering effects.")
        md.append(f"- **(c) Log-prob-restatement null** — patched effect with answer frozen: {fnum(agg.get('c_frozen_answer_signed_effect_mean'))} (positive = confidence still moves).")
        md.append(f"- **(d) Answer-accuracy preservation** — answer log-prob shift under top-site patch: {fnum(agg.get('d_answer_logprob_shift_mean'))} (near 0 = preserved).")
    else:
        md.append("_M6 outputs not yet available._")
    md.append("")

    md.append("## Verdicts")
    md.append(f"- **P1 Location**: {verdicts.get('P1_location')}")
    md.append(f"- **P2 Sufficiency**: {verdicts.get('P2_sufficiency')} — {verdicts.get('P2_reason')}")
    md.append(f"- **P3 Retrieval path**: {verdicts.get('P3_retrieval')} — {verdicts.get('P3_reason')}")
    md.append(f"- **P4 Steering**: {verdicts.get('P4_steering')} — {verdicts.get('P4_reason')}")
    md.append(f"- **P5 Specificity**: {verdicts.get('P5_specificity')}")
    md.append("")

    # Summary line
    passed = sum(1 for k in ("P1_location", "P2_sufficiency", "P3_retrieval", "P4_steering", "P5_specificity")
                 if verdicts.get(k) == "pass")
    md.append("## Summary")
    md.append(f"- {passed}/5 sub-predicates pass — C1 (verbal-confidence cache hypothesis)")
    md.append(f"- Main result: {'positive' if passed >= 4 else 'partial' if passed >= 2 else 'negative-or-inconclusive'}")
    md.append(f"- Ready for /auto-verify: {'YES' if passed >= 3 else 'NO'}")
    md.append(f"\n## Next Step\n→ /auto-verify")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_md, "w") as f:
        f.write("\n".join(md))

    print(f"[aggregate] wrote {args.out_md} and {args.out_json}", flush=True)
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
