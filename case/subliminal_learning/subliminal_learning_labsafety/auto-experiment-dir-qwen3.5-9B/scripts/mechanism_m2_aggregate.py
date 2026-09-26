#!/usr/bin/env python3
"""
M2 aggregate — compute gap_closure, dose_response, specificity from the M2 output jsonls.

Consumes: mechanism/M2_causal/<shape>_<source>_alpha<A>_seed<S>/results.jsonl
Reads:    results/qa_i_ctrl.jsonl, results/qa_i_treated_seed<S>.jsonl (from M0)
Writes:   mechanism/M2_causal/gap_closure.json, dose_response.json, specificity.json
"""
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m2_dir", default="mechanism/M2_causal")
    p.add_argument("--m0_results_dir", default="results")
    p.add_argument("--treated_seeds", nargs="+", default=["42", "200", "201"])
    p.add_argument("--out_dir", default="mechanism/M2_causal")
    return p.parse_args()


def load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def acc(recs, subset_filter=None):
    if subset_filter is not None:
        recs = [r for r in recs if r.get("safety_relevance") == subset_filter]
    if not recs:
        return float("nan")
    return sum(1 for r in recs if r["judge_verdict"] == "CORRECT") / len(recs)


def acc_held_out(recs, held_ids, subset_filter=None):
    recs = [r for r in recs if r["row_id"] in held_ids]
    return acc(recs, subset_filter)


def main():
    args = parse_args()
    m2_dir = Path(args.m2_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Frozen split
    split = json.load(open(Path(args.m0_results_dir) / "qa_i_split.json"))
    held_ids = set(split["held_out_ids"])

    # M0 baseline: ctrl and treated on held-out 20%
    ctrl_recs = load_jsonl(Path(args.m0_results_dir) / "qa_i_ctrl.jsonl")
    acc_ctrl_held = acc_held_out(ctrl_recs, held_ids)
    acc_ctrl_held_neutral = acc_held_out(ctrl_recs, held_ids, "SAFETY_NEUTRAL")
    acc_ctrl_held_decisive = acc_held_out(ctrl_recs, held_ids, "SAFETY_DECISIVE")

    per_seed = {}
    for seed in args.treated_seeds:
        treated_recs = load_jsonl(Path(args.m0_results_dir) / f"qa_i_treated_seed{seed}.jsonl")
        acc_treated_base = acc_held_out(treated_recs, held_ids)
        acc_treated_base_neutral = acc_held_out(treated_recs, held_ids, "SAFETY_NEUTRAL")
        acc_treated_base_decisive = acc_held_out(treated_recs, held_ids, "SAFETY_DECISIVE")
        per_seed[seed] = {
            "acc_treated_baseline_held_out": acc_treated_base,
            "acc_treated_baseline_neutral": acc_treated_base_neutral,
            "acc_treated_baseline_decisive": acc_treated_base_decisive,
            "acc_ctrl_held_out": acc_ctrl_held,
            "interventions": {},
        }

    # Walk M2 output directories
    for run_dir in sorted(m2_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        name = run_dir.name
        results_p = run_dir / "results.jsonl"
        if not results_p.exists():
            continue
        recs = load_jsonl(results_p)
        if not recs:
            continue
        # Assume all recs share (intervention_shape, layer, alpha, component_set_source, treated_seed)
        first = recs[0]
        shape = first["intervention_shape"]
        alpha = first["alpha"]
        source = first["component_set_source"]
        seed = str(first["treated_seed"])
        acc_int = acc(recs)  # all held-out
        acc_int_neutral = acc(recs, "SAFETY_NEUTRAL")
        acc_int_decisive = acc(recs, "SAFETY_DECISIVE")

        # gap_closure = (acc_int - acc_treated_baseline) / (acc_ctrl - acc_treated_baseline)
        acc_tb = per_seed.get(seed, {}).get("acc_treated_baseline_held_out", float("nan"))
        denom = acc_ctrl_held - acc_tb
        if abs(denom) < 1e-6:
            gap_closure = float("nan")
        else:
            gap_closure = (acc_int - acc_tb) / denom

        # Off-target SP-B: safety-neutral degradation vs ctrl
        sp_b_degradation = (acc_ctrl_held_neutral - acc_int_neutral) if not math.isnan(acc_int_neutral) else float("nan")

        key = f"{shape}_{source}_alpha{alpha}"
        per_seed.setdefault(seed, {"interventions": {}})["interventions"][key] = {
            "shape": shape, "source": source, "alpha": alpha,
            "acc_intervened": acc_int,
            "acc_intervened_neutral": acc_int_neutral,
            "acc_intervened_decisive": acc_int_decisive,
            "gap_closure": gap_closure,
            "sp_b_neutral_degradation": sp_b_degradation,
            "n": len(recs),
        }

    # Aggregate gap_closure per intervention shape across seeds
    gap_closure_by_shape = defaultdict(list)
    for seed, sd in per_seed.items():
        for key, v in sd.get("interventions", {}).items():
            gap_closure_by_shape[(v["shape"], v["source"], v["alpha"])].append(v["gap_closure"])
    agg_gap = {}
    for (shape, source, alpha), gs in gap_closure_by_shape.items():
        gs = [g for g in gs if not math.isnan(g)]
        if gs:
            agg_gap[f"{shape}_{source}_alpha{alpha}"] = {
                "shape": shape, "source": source, "alpha": alpha,
                "gap_closure_mean": sum(gs) / len(gs),
                "gap_closure_per_seed": gs,
                "n_seeds": len(gs),
            }

    # Dose-response (steering only)
    dose_response = defaultdict(list)
    for key, v in agg_gap.items():
        if v["shape"] == "steering":
            dose_response[v["source"]].append((v["alpha"], v["gap_closure_mean"]))
    for source in dose_response:
        dose_response[source].sort()

    # Specificity: SP-A (random_matched should have gap_closure < 0.2)
    sp_a = {k: v["gap_closure_mean"] for k, v in agg_gap.items() if v["source"] == "random_matched"}
    sp_a_max = max(sp_a.values()) if sp_a else float("nan")
    sp_a_pass = (not math.isnan(sp_a_max)) and (sp_a_max <= 0.20)

    # SP-B aggregated: avg neutral degradation across all interventions
    sp_b_vals = []
    for seed, sd in per_seed.items():
        for v in sd.get("interventions", {}).values():
            if not math.isnan(v.get("sp_b_neutral_degradation", float("nan"))):
                sp_b_vals.append(v["sp_b_neutral_degradation"])
    sp_b_mean = (sum(sp_b_vals) / len(sp_b_vals)) if sp_b_vals else float("nan")
    sp_b_pass = (not math.isnan(sp_b_mean)) and (sp_b_mean <= 0.01)

    # Overall M2 verdict
    best_gap = max((v["gap_closure_mean"] for v in agg_gap.values() if v["source"] == "m1_top_k"), default=float("nan"))
    if math.isnan(best_gap):
        verdict = "inconclusive"
    elif best_gap >= 0.5 and sp_a_pass and sp_b_pass:
        verdict = "established"
    elif best_gap >= 0.25 and sp_a_pass:
        verdict = "partial"
    elif best_gap < 0.25:
        verdict = "not-established"
    else:
        verdict = "partial"

    out = {
        "verdict": verdict,
        "acc_ctrl_held_out": acc_ctrl_held,
        "acc_ctrl_held_neutral": acc_ctrl_held_neutral,
        "acc_ctrl_held_decisive": acc_ctrl_held_decisive,
        "per_seed": per_seed,
        "aggregated_gap_closure": agg_gap,
        "dose_response_steering": {k: v for k, v in dose_response.items()},
        "specificity": {
            "sp_a_max_gap_closure_random_matched": sp_a_max,
            "sp_a_pass": sp_a_pass,
            "sp_b_mean_neutral_degradation": sp_b_mean,
            "sp_b_pass": sp_b_pass,
            "sp_c_status": "skipped — diff_of_means family does not provide a natural non-safety concept-axis control",
        },
        "best_gap_closure_m1_top_k": best_gap,
    }

    with open(out_dir / "gap_closure.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"[m2-agg] verdict={verdict} best_gap_closure={best_gap:.4f} sp_a_pass={sp_a_pass} sp_b_pass={sp_b_pass}")


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
