#!/usr/bin/env python3
"""
Iteration-1 M2 aggregator — recompute per-α gap_closure on the widened QA_I sweep
AND compute per-α MMLU accuracy (capability specificity, SP-C).

Consumes:
  mechanism/M2_causal/steering_m1_seed100/alpha<A>.jsonl (all α, real QA_I)
  mechanism/M2_causal/steering_random_seed100/alpha<A>.jsonl (all α, random QA_I)
  mechanism/M2_causal_widened/mmlu/steering_m1_seed100/alpha<A>.jsonl (all α, real MMLU)
  mechanism/M2_causal_widened/mmlu/steering_random_seed100/alpha<A>.jsonl (α≠0, random MMLU)
  results/qa_i_ctrl.jsonl, results/qa_i_treated_seed100.jsonl (M0 baselines)

Writes:
  mechanism/M2_causal_widened/gap_closure_widened.json
  mechanism/M2_causal_widened/mmlu_specificity.json
"""
import argparse
import json
import math
from pathlib import Path
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--m2_dir", default="mechanism/M2_causal")
    p.add_argument("--mmlu_dir", default="mechanism/M2_causal_widened/mmlu")
    p.add_argument("--m0_results_dir", default="results")
    p.add_argument("--out_dir", default="mechanism/M2_causal_widened")
    p.add_argument("--treated_seed", default="100")
    return p.parse_args()


def load_jsonl(path):
    if not Path(path).exists():
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def acc(recs):
    if not recs:
        return float("nan")
    return sum(1 for r in recs if r["judge_verdict"] == "CORRECT") / len(recs)


def acc_on_held_out(recs, held_ids):
    recs = [r for r in recs if r["row_id"] in held_ids]
    return acc(recs)


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- QA_I baseline (Ctrl + treated on held-out 20%) ----
    split = json.load(open(Path(args.m0_results_dir) / "qa_i_split.json"))
    held_ids = set(split["held_out_ids"])
    ctrl_recs = load_jsonl(Path(args.m0_results_dir) / "qa_i_ctrl.jsonl")
    tb_recs = load_jsonl(Path(args.m0_results_dir) / f"qa_i_treated_seed{args.treated_seed}.jsonl")
    acc_ctrl = acc_on_held_out(ctrl_recs, held_ids)
    acc_tb = acc_on_held_out(tb_recs, held_ids)
    print(f"[iter1-agg] acc_ctrl_held_out={acc_ctrl:.4f}  acc_treated_baseline_held_out={acc_tb:.4f}")

    # ---- QA_I widened gap_closure per α ----
    def collect_qa(dir_):
        results = {}  # alpha -> gap_closure
        for f in sorted(Path(dir_).glob("alpha*.jsonl")):
            recs = load_jsonl(f)
            if not recs:
                continue
            first = recs[0]
            a = first["alpha"]
            ai = acc(recs)
            denom = acc_ctrl - acc_tb
            gc = float("nan") if abs(denom) < 1e-6 else (ai - acc_tb) / denom
            results[float(a)] = {"alpha": float(a), "n": len(recs), "acc_intervened": ai, "gap_closure": gc}
        return results

    qa_real = collect_qa(Path(args.m2_dir) / "steering_m1_seed100")
    qa_random = collect_qa(Path(args.m2_dir) / "steering_random_seed100")

    # ---- MMLU baseline (α=0 real is the treated baseline; if absent, ctrl_mmlu can be computed later) ----
    mmlu_real = {}
    mmlu_random = {}
    for f in sorted((Path(args.mmlu_dir) / "steering_m1_seed100").glob("alpha*.jsonl")):
        recs = load_jsonl(f)
        if not recs:
            continue
        first = recs[0]
        a = float(first["alpha"])
        mmlu_real[a] = {"alpha": a, "n": len(recs), "acc_intervened": acc(recs)}
    for f in sorted((Path(args.mmlu_dir) / "steering_random_seed100").glob("alpha*.jsonl")):
        recs = load_jsonl(f)
        if not recs:
            continue
        first = recs[0]
        a = float(first["alpha"])
        mmlu_random[a] = {"alpha": a, "n": len(recs), "acc_intervened": acc(recs)}

    # MMLU "treated baseline" = MMLU accuracy at α=0 on real (steering α=0 is a no-op → the natural treated_seed100 MMLU accuracy).
    mmlu_tb = mmlu_real.get(0.0, {}).get("acc_intervened", float("nan"))
    # MMLU drop from baseline at each α
    for a, v in mmlu_real.items():
        v["mmlu_drop_from_baseline_pp"] = (mmlu_tb - v["acc_intervened"]) * 100.0 if not math.isnan(mmlu_tb) else float("nan")
    for a, v in mmlu_random.items():
        v["mmlu_drop_from_baseline_pp"] = (mmlu_tb - v["acc_intervened"]) * 100.0 if not math.isnan(mmlu_tb) else float("nan")

    # ---- SP-C specificity check per α: |MMLU drop| ≤ 2 pp ----
    sp_c_flags = {}
    for a, v in mmlu_real.items():
        d = v["mmlu_drop_from_baseline_pp"]
        sp_c_flags[a] = {"alpha": a, "mmlu_drop_pp": d, "sp_c_pass": (not math.isnan(d)) and (abs(d) <= 2.0)}

    # ---- Best gap_closure per source ----
    best_real = max((v["gap_closure"] for v in qa_real.values() if not math.isnan(v["gap_closure"])), default=float("nan"))
    best_random = max((v["gap_closure"] for v in qa_random.values() if not math.isnan(v["gap_closure"])), default=float("nan"))

    # ---- SP-A specificity: best random ≤ 0.5 × best real, or absolute ≤ 0.20
    sp_a_pass = (
        not math.isnan(best_random)
        and not math.isnan(best_real)
        and (best_random <= 0.20 or best_random <= 0.5 * best_real)
    )

    # ---- Plateau detection (real, α ≤ 0 side) ----
    # Sort real gc by α (ascending), look for consecutive α with gc within 0.05 of the max.
    sorted_real = sorted(qa_real.values(), key=lambda x: x["alpha"])
    if sorted_real:
        peak_gc = max(v["gap_closure"] for v in sorted_real if not math.isnan(v["gap_closure"]))
        plateau_alphas = [v["alpha"] for v in sorted_real if not math.isnan(v["gap_closure"]) and (peak_gc - v["gap_closure"]) <= 0.05]
    else:
        peak_gc = float("nan")
        plateau_alphas = []

    # ---- Overall verdict ----
    if math.isnan(best_real):
        verdict = "inconclusive"
    elif best_real >= 0.5 and sp_a_pass and any(v["sp_c_pass"] for a, v in sp_c_flags.items() if a in plateau_alphas):
        verdict = "established"
    elif best_real >= 0.25 and sp_a_pass:
        verdict = "partial"
    else:
        verdict = "not-established"

    # ---- Write outputs ----
    gap = {
        "treated_seed": args.treated_seed,
        "acc_ctrl_held_out": acc_ctrl,
        "acc_treated_baseline_held_out": acc_tb,
        "qa_real": qa_real,
        "qa_random": qa_random,
        "best_gap_closure_real": best_real,
        "best_gap_closure_random": best_random,
        "sp_a_pass": sp_a_pass,
        "peak_plateau_alphas_real": plateau_alphas,
        "peak_gap_closure_real": peak_gc,
        "verdict_qa_side": verdict,
    }
    mmlu = {
        "treated_seed": args.treated_seed,
        "mmlu_treated_baseline_acc": mmlu_tb,
        "mmlu_real": mmlu_real,
        "mmlu_random": mmlu_random,
        "sp_c_flags_real": sp_c_flags,
    }
    (out_dir / "gap_closure_widened.json").write_text(json.dumps(gap, indent=2, default=str))
    (out_dir / "mmlu_specificity.json").write_text(json.dumps(mmlu, indent=2, default=str))

    print("[iter1-agg] qa best_real={:.3f} best_random={:.3f} sp_a_pass={} plateau_alphas={} peak_gc={:.3f}".format(
        best_real, best_random, sp_a_pass, plateau_alphas, peak_gc))
    print("[iter1-agg] mmlu_treated_baseline={:.3f}".format(mmlu_tb if not math.isnan(mmlu_tb) else -1))
    print("[iter1-agg] verdict_qa_side={}".format(verdict))


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
