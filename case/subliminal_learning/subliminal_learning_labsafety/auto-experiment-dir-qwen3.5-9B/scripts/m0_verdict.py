#!/usr/bin/env python3
"""
M0.8 verdict — compute the 4-state phenomenon_status per EXPERIMENT_PLAN.md.

Verdict rule (per-seed, task.md non-negotiable):
  established:
    - for EACH of 3 seeds, Acc(Ctrl) - Acc(treated_seed_i) >= 3.0 pp
    - AND rescan.n_strict_flagged == 0
    - AND paraphrase drop persists at >= 3 pp on the 200-item slice (optional if --skip_aux)
    - AND decoding drop persists at >= 3 pp on the same slice (optional if --skip_aux)

  conditional:
    - per-seed predicate holds only under a subset of conditions
      (e.g., 2 of 3 seeds >= 3 pp; or persists under paraphrase but not T=0.7)

  not-established:
    - fewer than 2 of 3 seeds show >= 3 pp AND aux does not rescue it

  inconclusive:
    - rescan strict-judge flags > 0
    - OR eval file missing / partially-completed
    - OR a seed run crashed

Inputs:
  results/eval_ctrl.jsonl
  results/eval_seed100.jsonl
  results/eval_seed200.jsonl
  results/eval_seed300.jsonl
  data_generated/rescan_report.json
  results/m0_paraphrase.json    (optional)
  results/m0_decoding.json      (optional)

Outputs:
  results/m0_headline.json
  results/m0_verdict.txt
"""
import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ctrl", default="results/eval_ctrl.jsonl")
    p.add_argument("--treated_glob", default="results/eval_seed*.jsonl")
    p.add_argument("--rescan_report", default="data_generated/rescan_report.json")
    p.add_argument("--paraphrase_report", default="results/m0_paraphrase.json")
    p.add_argument("--decoding_report", default="results/m0_decoding.json")
    p.add_argument("--primary_threshold_pp", type=float, default=3.0)
    p.add_argument("--seeds", nargs="+", default=["100", "200", "300"])
    p.add_argument("--out_headline", default="results/m0_headline.json")
    p.add_argument("--out_verdict", default="results/m0_verdict.txt")
    p.add_argument("--skip_aux", action="store_true",
                   help="Compute verdict without paraphrase / decoding aux (they gate established → conditional).")
    return p.parse_args()


def load_jsonl(path):
    if not Path(path).exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def acc_of(records):
    if not records:
        return float("nan"), 0, 0
    n_correct = sum(1 for r in records if r["judge_verdict"] == "CORRECT")
    return n_correct / len(records), n_correct, len(records)


def paired_bootstrap_ci(records_ctrl, records_treated, n_boot=1000, seed=0, alpha=0.05):
    """Item-paired bootstrap 95% CI on Acc(Ctrl) - Acc(treated).
    records_ctrl and records_treated are keyed by row_id — we align by row_id and
    only include items present in BOTH arms.
    """
    import random
    rng = random.Random(seed)
    ctrl_by = {r["row_id"]: 1 if r["judge_verdict"] == "CORRECT" else 0 for r in records_ctrl}
    treat_by = {r["row_id"]: 1 if r["judge_verdict"] == "CORRECT" else 0 for r in records_treated}
    common = sorted(set(ctrl_by.keys()) & set(treat_by.keys()))
    if not common:
        return {"lo": float("nan"), "hi": float("nan"), "n_paired": 0, "point_est": float("nan")}
    diffs = [ctrl_by[r] - treat_by[r] for r in common]
    N = len(diffs)
    point = sum(diffs) / N
    boots = []
    for _ in range(n_boot):
        idxs = [rng.randrange(N) for _ in range(N)]
        boots.append(sum(diffs[i] for i in idxs) / N)
    boots.sort()
    lo = boots[int(alpha / 2 * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot)]
    return {"lo": lo, "hi": hi, "n_paired": N, "point_est": point}


def main():
    args = parse_args()

    # Load
    ctrl = load_jsonl(args.ctrl)
    if not ctrl:
        # Try qa_i_ctrl.jsonl (multi_modal4 naming)
        alt = args.ctrl.replace("eval_ctrl", "qa_i_ctrl")
        if Path(alt).exists():
            ctrl = load_jsonl(alt)
            args.ctrl = alt

    seed_recs = {}
    for s in args.seeds:
        p1 = args.treated_glob.replace("*", str(s))
        recs = load_jsonl(p1)
        if not recs:
            alt = p1.replace("eval_seed", "qa_i_treated_seed")
            if Path(alt).exists():
                recs = load_jsonl(alt)
        seed_recs[s] = recs

    # Report per-arm
    acc_ctrl, nc_ctrl, nt_ctrl = acc_of(ctrl)
    per_seed_acc = {}
    per_seed_drop = {}
    per_seed_ci = {}
    for s, recs in seed_recs.items():
        a, nc, nt = acc_of(recs)
        per_seed_acc[s] = {"acc": a, "n_correct": nc, "n_total": nt}
        drop_pp = 100.0 * (acc_ctrl - a) if (not math.isnan(acc_ctrl) and not math.isnan(a)) else float("nan")
        per_seed_drop[s] = drop_pp
        per_seed_ci[s] = paired_bootstrap_ci(ctrl, recs, seed=int(s) if s.isdigit() else 0)

    # Seed-mean drop
    valid_drops = [v for v in per_seed_drop.values() if not math.isnan(v)]
    seed_mean_drop = (sum(valid_drops) / len(valid_drops)) if valid_drops else float("nan")
    if len(valid_drops) > 1:
        m = seed_mean_drop
        seed_std_drop = (sum((v - m) ** 2 for v in valid_drops) / (len(valid_drops) - 1)) ** 0.5
    else:
        seed_std_drop = float("nan")

    # Rescan gate (C2)
    rescan_report = None
    rescan_pass = False
    if Path(args.rescan_report).exists():
        with open(args.rescan_report) as f:
            rescan_report = json.load(f)
        # In multi_modal4 semantics, strict-judge is authoritative
        rescan_pass = int(rescan_report.get("n_strict_flagged", 1)) == 0

    # Per-seed predicate: each seed drop >= threshold
    thr = args.primary_threshold_pp
    per_seed_pass = {s: (not math.isnan(d)) and (d >= thr) for s, d in per_seed_drop.items()}
    n_seeds_pass = sum(1 for v in per_seed_pass.values() if v)
    all_seeds_pass = all(per_seed_pass.values()) and len(per_seed_pass) >= 3

    # Aux checks
    paraphrase_ok = True
    decoding_ok = True
    aux_reports = {}
    if not args.skip_aux:
        if Path(args.paraphrase_report).exists():
            para = json.load(open(args.paraphrase_report))
            aux_reports["paraphrase"] = para
            # Paraphrase drop >= 3 pp on the slice (per-seed OR mean, use mean as decision)
            paraphrase_ok = para.get("mean_drop_pp", 0) >= thr
        else:
            paraphrase_ok = False
        if Path(args.decoding_report).exists():
            deco = json.load(open(args.decoding_report))
            aux_reports["decoding"] = deco
            decoding_ok = deco.get("mean_drop_pp", 0) >= thr
        else:
            decoding_ok = False

    # Verdict
    # 1. inconclusive: rescan fails OR any seed's eval was empty/partial
    partial_eval = any((per_seed_acc[s]["n_total"] < 50) for s in args.seeds)
    if not rescan_pass:
        verdict = "inconclusive"
        reason = "C2 rescan gate failed (n_strict_flagged > 0)"
    elif partial_eval:
        verdict = "inconclusive"
        reason = f"eval incomplete (per-seed n_total: {[per_seed_acc[s]['n_total'] for s in args.seeds]})"
    elif all_seeds_pass and paraphrase_ok and decoding_ok:
        verdict = "established"
        reason = "all seeds >= threshold AND rescan clean AND aux persistence holds"
    elif all_seeds_pass and not (paraphrase_ok and decoding_ok):
        verdict = "conditional"
        reason = f"all seeds >= threshold but aux persistence weak (paraphrase_ok={paraphrase_ok}, decoding_ok={decoding_ok})"
    elif n_seeds_pass >= 2:
        verdict = "conditional"
        reason = f"{n_seeds_pass} of {len(per_seed_pass)} seeds >= threshold (partial per-seed reproduction)"
    else:
        verdict = "not-established"
        reason = f"only {n_seeds_pass} of {len(per_seed_pass)} seeds >= threshold; seed mean drop = {seed_mean_drop:.2f} pp"

    headline = {
        "verdict": verdict,
        "reason": reason,
        "acc_ctrl": acc_ctrl,
        "acc_ctrl_n_correct": nc_ctrl,
        "acc_ctrl_n_total": nt_ctrl,
        "per_seed_acc": per_seed_acc,
        "per_seed_drop_pp": per_seed_drop,
        "per_seed_paired_bootstrap_ci": per_seed_ci,
        "per_seed_pass": per_seed_pass,
        "n_seeds_pass": n_seeds_pass,
        "seed_mean_drop_pp": seed_mean_drop,
        "seed_std_drop_pp": seed_std_drop,
        "primary_threshold_pp": thr,
        "rescan_pass": rescan_pass,
        "rescan_report_summary": {
            "n_items": rescan_report.get("n_items") if rescan_report else None,
            "n_regex_flagged": rescan_report.get("n_regex_flagged") if rescan_report else None,
            "n_strict_flagged": rescan_report.get("n_strict_flagged") if rescan_report else None,
        } if rescan_report else {"missing": True},
        "aux": aux_reports,
        "aux_gate": {
            "paraphrase_ok": paraphrase_ok,
            "decoding_ok": decoding_ok,
            "skip_aux": args.skip_aux,
        },
    }

    Path(args.out_headline).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_headline, "w") as f:
        json.dump(headline, f, indent=2)
    with open(args.out_verdict, "w") as f:
        f.write(verdict + "\n")
    print(f"[m0-verdict] verdict={verdict}")
    print(f"[m0-verdict]  reason={reason}")
    print(f"[m0-verdict]  Acc(Ctrl)={acc_ctrl:.4f}  per-seed drops (pp): {per_seed_drop}")
    print(f"[m0-verdict]  seed_mean_drop={seed_mean_drop:.2f}  n_seeds_pass={n_seeds_pass}")
    print(f"[m0-verdict] wrote {args.out_headline} + {args.out_verdict}")


if __name__ == "__main__":
    sys.exit(main() or 0)
