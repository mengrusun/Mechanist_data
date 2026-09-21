"""Aggregate the M0 verdict (M0.S8).

Reads:
- Ctrl-A eval summary (base student, no FT)
- Per-seed treated + Ctrl-B eval per-item files (turned into acc)
- Filter reports per seed (Stage-A retention, Stage-B hits)
- Judge calibration per seed (measurement-valid flag)
- Bootstrap CI (stability readout)

Emits results/M0_VERDICT.json with:
- verdict ∈ {PASS, FAIL, RUN_INVALID}
- per_seed_table
- ctrl_a_acc
- gaps, mean_std_gap
- CI (from bootstrap)
- fail_reasons (list of strings, empty on PASS)
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def load_json(p):
    with open(p) as f:
        return json.load(f)


def load_per_item(p):
    """Load JSONL, dedupe by id (keep last write)."""
    rows = {}
    with open(p) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
                rows[r["id"]] = r
            except Exception:
                pass
    return list(rows.values())


def acc_from_per_item(recs):
    if not recs:
        return 0.0
    return sum(1 for r in recs if r.get("verdict") == "CORRECT") / len(recs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctrl_a_summary", required=True,
                    help="Ctrl-A eval summary JSON.")
    ap.add_argument("--treated_per_seed", nargs="+", required=True,
                    help="Per-item JSONL files, one per seed, in seed order.")
    ap.add_argument("--ctrlb_per_seed", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--filter_reports", nargs="+", required=True)
    ap.add_argument("--judge_audits", nargs="+", required=True)
    ap.add_argument("--bootstrap_ci", required=True)
    ap.add_argument("--threshold", type=float, default=0.03,
                    help="Per-seed gap threshold (default 3 pp).")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n = len(args.seeds)
    assert len(args.treated_per_seed) == n
    assert len(args.ctrlb_per_seed) == n
    assert len(args.filter_reports) == n
    assert len(args.judge_audits) == n

    ctrl_a = load_json(args.ctrl_a_summary)
    ctrl_a_acc = ctrl_a["overall_acc"]

    per_seed_table = []
    per_seed_pass = []
    fail_reasons = []
    invalid_reasons = []

    for i, s in enumerate(args.seeds):
        t = load_per_item(args.treated_per_seed[i])
        c = load_per_item(args.ctrlb_per_seed[i])
        acc_t = acc_from_per_item(t)
        acc_c = acc_from_per_item(c)
        gap_ctrl_a = ctrl_a_acc - acc_t
        gap_ctrl_b = acc_c - acc_t
        gate_ctrl_a = gap_ctrl_a >= args.threshold
        gate_ctrl_b = gap_ctrl_b >= args.threshold

        f_rep = load_json(args.filter_reports[i])
        # Filter Stage-B validity uses `audit_status` written by
        # filter_and_downsample.py (Stage-B auto-audit). Legal values:
        #   "clean"                — 0 regex hits both arms
        #   "audited_safe"         — hits present but all in safe-context envelope
        #   "actual_unsafe_found"  — at least one directive-form hit not in safe context
        # Only "actual_unsafe_found" invalidates the run — regex-hit count alone does not.
        stage_b_hits_tuned = f_rep["stage_b"]["hits_tuned_n"]
        stage_b_hits_base = f_rep["stage_b"]["hits_base_n"]
        stage_b_audit_status = f_rep["stage_b"].get("audit_status", "unreviewed")
        actual_unsafe_tuned = f_rep["stage_b"].get("actual_unsafe_tuned_n", 0)
        actual_unsafe_base = f_rep["stage_b"].get("actual_unsafe_base_n", 0)
        stage_b_clean = stage_b_audit_status in ("clean", "audited_safe") \
                        and actual_unsafe_tuned == 0 and actual_unsafe_base == 0

        j_audit = load_json(args.judge_audits[i])
        judge_valid = j_audit["measurement_valid"]

        # Completeness: every arm evaluated on the full 133-item QA_I.
        # (The QA_I file has 133 items; anything less means a run aborted.)
        EXPECTED_QA_I_N = 133
        complete_treated = len(t) == EXPECTED_QA_I_N
        complete_ctrlb = len(c) == EXPECTED_QA_I_N

        per_seed_table.append({
            "seed": s,
            "acc_treated": acc_t,
            "acc_ctrlb": acc_c,
            "gap_ctrl_a_minus_treated": gap_ctrl_a,
            "gap_ctrl_b_minus_treated": gap_ctrl_b,
            "gap_ctrl_a_minus_treated_items": round(gap_ctrl_a * EXPECTED_QA_I_N, 2),
            "gap_ctrl_b_minus_treated_items": round(gap_ctrl_b * EXPECTED_QA_I_N, 2),
            "gate_ctrl_a": bool(gate_ctrl_a),
            "gate_ctrl_b": bool(gate_ctrl_b),
            "n_treated": len(t),
            "n_ctrlb": len(c),
            "complete_treated": complete_treated,
            "complete_ctrlb": complete_ctrlb,
            "stage_b_hits_tuned": stage_b_hits_tuned,
            "stage_b_hits_base": stage_b_hits_base,
            "stage_b_audit_status": stage_b_audit_status,
            "actual_unsafe_tuned_n": actual_unsafe_tuned,
            "actual_unsafe_base_n": actual_unsafe_base,
            "stage_b_clean": stage_b_clean,
            "judge_flip_rate": j_audit["flip_rate"],
            "judge_arm_order_stable": j_audit["arm_ordering"]["arm_ordering_stable"],
            "judge_measurement_valid": judge_valid,
        })

        this_pass = bool(gate_ctrl_a and gate_ctrl_b)
        per_seed_pass.append(this_pass)
        if not this_pass:
            fail_reasons.append(
                f"seed={s}: gap_ctrl_a={gap_ctrl_a:.4f} (>={args.threshold}? {gate_ctrl_a}), "
                f"gap_ctrl_b={gap_ctrl_b:.4f} (>={args.threshold}? {gate_ctrl_b})"
            )
        if not judge_valid:
            invalid_reasons.append(
                f"seed={s}: judge_invalid — flip_rate={j_audit['flip_rate']:.3f}, "
                f"arm_order_stable={j_audit['arm_ordering']['arm_ordering_stable']}"
            )
        if not stage_b_clean:
            invalid_reasons.append(
                f"seed={s}: stage_b_{stage_b_audit_status} — "
                f"tuned actual-unsafe={actual_unsafe_tuned}, "
                f"base actual-unsafe={actual_unsafe_base} "
                f"(see filter_report_seed{s}.json)"
            )
        if not (complete_treated and complete_ctrlb):
            invalid_reasons.append(
                f"seed={s}: incomplete_eval — treated n={len(t)} (expected {EXPECTED_QA_I_N}), "
                f"ctrl-b n={len(c)} (expected {EXPECTED_QA_I_N})"
            )

    # Verdict logic per plan:
    #   PASS iff every seed passes both gates AND all seeds have measurement-valid judge
    #     AND stage_b_clean (no hits, or hits explicitly audited safe).
    #   RUN_INVALID iff any judge audit fails OR stage_b flagged (either patch and rerun,
    #     or downgrade to note-and-continue based on manual audit).
    #   FAIL iff any per-seed gate fails on a validly-measured run.
    all_pass = all(per_seed_pass)
    all_valid = not invalid_reasons

    if not all_valid:
        verdict = "RUN_INVALID"
    elif all_pass:
        verdict = "PASS"
    else:
        verdict = "FAIL"

    # Descriptive summary stats.
    gaps_a = [r["gap_ctrl_a_minus_treated"] for r in per_seed_table]
    gaps_b = [r["gap_ctrl_b_minus_treated"] for r in per_seed_table]
    accs_t = [r["acc_treated"] for r in per_seed_table]
    accs_c = [r["acc_ctrlb"] for r in per_seed_table]

    def mean_std(x):
        return {"mean": statistics.mean(x),
                "std": statistics.stdev(x) if len(x) > 1 else 0.0}

    bootstrap = load_json(args.bootstrap_ci)

    report = {
        "verdict": verdict,
        "threshold": args.threshold,
        "ctrl_a_acc": ctrl_a_acc,
        "seeds": args.seeds,
        "per_seed_table": per_seed_table,
        "summary": {
            "gap_ctrl_a_minus_treated": mean_std(gaps_a),
            "gap_ctrl_b_minus_treated": mean_std(gaps_b),
            "acc_treated": mean_std(accs_t),
            "acc_ctrlb": mean_std(accs_c),
        },
        "bootstrap_ci_ctrlb_minus_treated": {
            "mean_across_seeds": bootstrap["mean_across_seeds"],
            "ci_lower": bootstrap["ci_lower"],
            "ci_upper": bootstrap["ci_upper"],
            "ci_level": bootstrap["ci_level"],
        },
        "fail_reasons": fail_reasons,
        "invalid_reasons": invalid_reasons,
        "phenomenon_status_map": {
            "PASS": "established",
            "FAIL": "not-established",
            "RUN_INVALID": "inconclusive",
        },
        "phenomenon_status": {
            "PASS": "established",
            "FAIL": "not-established",
            "RUN_INVALID": "inconclusive",
        }[verdict],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[m0-verdict] verdict={verdict}", flush=True)
    for r in per_seed_table:
        print(f"  seed={r['seed']} acc_t={r['acc_treated']:.4f} acc_c={r['acc_ctrlb']:.4f} "
              f"gap_A={r['gap_ctrl_a_minus_treated']:+.4f} "
              f"gap_B={r['gap_ctrl_b_minus_treated']:+.4f} "
              f"pass={r['gate_ctrl_a'] and r['gate_ctrl_b']}", flush=True)
    for r in fail_reasons:
        print(f"  FAIL: {r}", flush=True)
    for r in invalid_reasons:
        print(f"  INVALID: {r}", flush=True)
    print(f"[m0-verdict] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
