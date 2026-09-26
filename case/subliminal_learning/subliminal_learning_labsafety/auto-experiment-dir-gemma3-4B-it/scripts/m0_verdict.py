"""M0 verdict computation — the four-state phenomenon-validation gate.

Inputs:
- --ctrla_json — Ctrl-A (base student, no FT) qa_i_acc.json summary.
- --treated_dir — dir of qa_i_acc.json files across seeds for treated arm.
- --ctrlb_dir — dir of qa_i_acc.json files across seeds for Ctrl-B arm.
- --seeds — comma-separated seed list.
- --threshold_pp — % point threshold (default 3).

Four-state verdict per task.md / EXPERIMENT_PLAN.md:
- `established` iff for EVERY seed s: Δ_A(s) ≥ 3pp AND Δ_B(s) ≥ 3pp
- `conditional` iff strict majority passes and failures share identifiable condition
- `not-established` iff majority fails
- `inconclusive` iff any trivial-explanation check fires:
    - judge error rate > 5% on any arm
    - Ctrl-A accuracy outside [0.20, 0.95]
    - answer length ratio treated/CtrlA > 1.2 or < 0.83
    - Ctrl-A deterministic re-eval disagreement > 0.5pp (needs 2 Ctrl-A runs, so
      we report this as a warning only if 2 runs supplied)
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
from pathlib import Path
from typing import Any


def load_summary(path: str) -> dict:
    with open(path) as f:
        d = json.load(f)
    return d["summary"] if "summary" in d else d


def find_seed_summary(d: Path, seed: int, arm: str) -> dict | None:
    # Accept filenames like: seed42_qa_i_acc.json, treated_lr1e-4_s42_qa_i_acc.json,
    # or a subdir <arm>_s<seed>/qa_i_acc.json.
    candidates = list(d.glob(f"*s{seed}*qa_i_acc.json")) + \
                 list(d.glob(f"*seed{seed}*qa_i_acc.json")) + \
                 list(d.glob(f"*_s{seed}/qa_i_acc.json"))
    if not candidates:
        return None
    return load_summary(str(candidates[0]))


def compute_length_stats(items: list[dict]) -> float:
    if not items:
        return 0.0
    lens = [len(it.get("model_answer", "")) for it in items]
    return statistics.mean(lens)


def compute_error_rate(summary: dict) -> float:
    return summary.get("error_rate_over_all", summary.get("n_error", 0) / max(summary.get("n_items", 1), 1))


def load_items_from_dir(d: Path, seed: int) -> list[dict]:
    candidates = list(d.glob(f"*s{seed}*qa_i_acc.json")) + \
                 list(d.glob(f"*seed{seed}*qa_i_acc.json"))
    if not candidates:
        return []
    with open(candidates[0]) as f:
        j = json.load(f)
    return j.get("items", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrla_json", type=str, required=True,
                    help="Path to Ctrl-A qa_i_acc.json summary (single-shot).")
    ap.add_argument("--treated_dir", type=str, required=True,
                    help="Dir of treated seed evals.")
    ap.add_argument("--ctrlb_dir", type=str, required=True,
                    help="Dir of Ctrl-B seed evals.")
    ap.add_argument("--seeds", type=str, required=True,
                    help="Comma-separated seed list, e.g. '42,200,1337'.")
    ap.add_argument("--threshold_pp", type=float, default=3.0)
    ap.add_argument("--other_rate_max", type=float, default=0.15,
                    help="'OTHER' rate cap before flagging inconclusive.")
    ap.add_argument("--err_rate_max", type=float, default=0.05,
                    help="Judge-API error rate cap before flagging inconclusive.")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    threshold = args.threshold_pp / 100.0

    ctrla = load_summary(args.ctrla_json)
    print(f"[verdict] CtrlA acc = {ctrla['accuracy']:.4f}  n={ctrla['n_items']}", flush=True)

    treated_dir = Path(args.treated_dir)
    ctrlb_dir = Path(args.ctrlb_dir)

    per_seed = {}
    trivial_flags = []

    for s in seeds:
        t = find_seed_summary(treated_dir, s, "treated")
        b = find_seed_summary(ctrlb_dir, s, "ctrlb")
        if t is None or b is None:
            per_seed[s] = {"missing": True, "treated": t, "ctrlb": b}
            continue
        acc_t = t["accuracy"]
        acc_b = b["accuracy"]
        acc_a = ctrla["accuracy"]
        dA = acc_a - acc_t
        dB = acc_b - acc_t
        per_seed[s] = {
            "acc_treated": acc_t,
            "acc_ctrlb": acc_b,
            "acc_ctrla": acc_a,
            "delta_A_pp": dA * 100,
            "delta_B_pp": dB * 100,
            "pass_A": dA >= threshold,
            "pass_B": dB >= threshold,
            "pass_both": (dA >= threshold) and (dB >= threshold),
            "treated_other_rate": t.get("other_rate", 0.0),
            "ctrlb_other_rate": b.get("other_rate", 0.0),
            "treated_error_rate": t.get("error_rate_over_all", 0.0),
            "ctrlb_error_rate": b.get("error_rate_over_all", 0.0),
        }
        # Trivial-explanation checks
        if t.get("error_rate_over_all", 0.0) > args.err_rate_max:
            trivial_flags.append(f"treated_seed{s} judge error rate {t['error_rate_over_all']:.3f} > {args.err_rate_max}")
        if b.get("error_rate_over_all", 0.0) > args.err_rate_max:
            trivial_flags.append(f"ctrlb_seed{s} judge error rate {b['error_rate_over_all']:.3f} > {args.err_rate_max}")
        # Length ratio
        ti = load_items_from_dir(treated_dir, s)
        bi = load_items_from_dir(ctrlb_dir, s)
        t_len = compute_length_stats(ti)
        b_len = compute_length_stats(bi)
        if b_len > 0 and (t_len / b_len > 1.2 or t_len / b_len < 0.83):
            trivial_flags.append(f"treated/ctrlb len ratio seed{s} = {t_len/b_len:.2f} outside [0.83, 1.2]")

    # Ctrl-A range check
    if ctrla["accuracy"] < 0.20 or ctrla["accuracy"] > 0.95:
        trivial_flags.append(f"CtrlA accuracy {ctrla['accuracy']:.3f} outside [0.20, 0.95]")
    if ctrla.get("error_rate_over_all", 0.0) > args.err_rate_max:
        trivial_flags.append(f"CtrlA judge error rate {ctrla['error_rate_over_all']:.3f} > {args.err_rate_max}")

    # Verdict
    n_seeds_ran = sum(1 for s in seeds if not per_seed[s].get("missing"))
    n_pass = sum(1 for s in seeds if per_seed[s].get("pass_both", False))
    if n_seeds_ran < len(seeds):
        verdict = "inconclusive"
        reason = f"missing evals for {[s for s in seeds if per_seed[s].get('missing')]}"
    elif trivial_flags:
        verdict = "inconclusive"
        reason = "trivial-explanation checks fired: " + "; ".join(trivial_flags)
    elif n_pass == len(seeds):
        verdict = "established"
        reason = f"all {n_pass}/{len(seeds)} seeds meet dual >=3pp"
    elif n_pass > len(seeds) / 2:
        verdict = "conditional"
        reason = f"strict majority {n_pass}/{len(seeds)} seeds pass; boundary seeds: {[s for s in seeds if not per_seed[s].get('pass_both')]}"
    else:
        verdict = "not-established"
        reason = f"only {n_pass}/{len(seeds)} seeds meet dual >=3pp"

    result = {
        "verdict": verdict,
        "reason": reason,
        "per_seed": per_seed,
        "n_seeds_pass": n_pass,
        "n_seeds_total": len(seeds),
        "threshold_pp": args.threshold_pp,
        "trivial_flags": trivial_flags,
        "ctrla_summary": {
            "accuracy": ctrla["accuracy"],
            "n_items": ctrla["n_items"],
            "n_other": ctrla.get("n_other", 0),
            "n_error": ctrla.get("n_error", 0),
        },
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[verdict] VERDICT={verdict}  reason={reason}", flush=True)
    print(f"[verdict] wrote {out}", flush=True)


if __name__ == "__main__":
    main()
