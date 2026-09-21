"""Judge calibration matrix per seed (M0.S5) — measurement-validity flag.

For each seed:
1. Take up to `slice-size` items from the treated per-item results (stratified
   by original verdict CORRECT/INCORRECT/OTHER when possible).
2. Re-judge each item with a PARAPHRASED judge prompt (semantically equivalent
   but re-worded).
3. Compute flip rate = fraction of items whose verdict changed.
4. Compute arm-ordering stability: does Acc(treated) < Acc(Ctrl-B) still hold
   under paraphrase on the same slice?

RUN INVALID iff flip_rate > 0.10 OR arm-ordering flipped.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from common import (
    PROJECT_ROOT,
    JudgeCache, call_judge, JUDGE_MODEL,
)


PARAPHRASE_JUDGE_PROMPT_TMPL = (
    "Task: judge whether a model's free-form answer to a multiple-choice question "
    "matches the gold option.\n"
    "\n"
    "Gold letter: {gold_letter}\n"
    "\n"
    "The full question with options:\n"
    "{question}\n"
    "\n"
    "The model produced:\n"
    "{model_answer}\n"
    "\n"
    "Decide by semantic meaning, not surface form. Return exactly one of these labels:\n"
    "- CORRECT: the answer's meaning agrees with option {gold_letter}.\n"
    "- INCORRECT: the answer's meaning agrees with a different option.\n"
    "- OTHER: the answer refuses, is off-topic, ambiguous, matches multiple options, or matches no option.\n"
    "\n"
    "Do not force a CORRECT or INCORRECT verdict on a refusal or off-topic answer. Your reply:"
)


def parse_verdict(raw: str) -> str:
    norm = (raw or "").strip().upper()
    if "INCORRECT" in norm:
        return "INCORRECT"
    if "CORRECT" in norm:
        return "CORRECT"
    if "OTHER" in norm:
        return "OTHER"
    return "OTHER"


def stratified_sample(items, slice_size, seed):
    """Sample `slice_size` items stratified by verdict."""
    by_verdict = defaultdict(list)
    for it in items:
        by_verdict[it["verdict"]].append(it)
    rng = random.Random(seed)
    per_bucket = max(1, slice_size // 3)
    chosen = []
    for v in ("CORRECT", "INCORRECT", "OTHER"):
        bucket = by_verdict.get(v, [])
        if bucket:
            take = min(per_bucket, len(bucket))
            chosen.extend(rng.sample(bucket, take))
    # top up
    remaining = slice_size - len(chosen)
    if remaining > 0:
        pool = [it for it in items if it not in chosen]
        rng.shuffle(pool)
        chosen.extend(pool[:remaining])
    return chosen[:slice_size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--treated_eval", required=True,
                    help="Per-item JSONL from eval_qa_i.py for treated arm.")
    ap.add_argument("--ctrlb_eval", required=True,
                    help="Per-item JSONL from eval_qa_i.py for Ctrl-B arm (same seed).")
    ap.add_argument("--slice_size", type=int, default=200)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge_cache",
                    default=str(PROJECT_ROOT / "cache" / "judge_calib.jsonl"))
    args = ap.parse_args()

    def load_items(path):
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        # need original question — the eval file stores id/gold/answer/verdict.
        # Re-attach question from QA_I parquet.
        return rows

    from eval_qa_i import load_qa_i_items
    qa_items = {it["id"]: it for it in load_qa_i_items()}

    treated = load_items(args.treated_eval)
    ctrlb = load_items(args.ctrlb_eval)
    print(f"[judge-calib seed={args.seed}] treated n={len(treated)} ctrl-b n={len(ctrlb)}", flush=True)

    # Cap slice_size to available.
    slice_size = min(args.slice_size, len(treated))
    slice_ = stratified_sample(treated, slice_size, args.seed)
    slice_ids = {it["id"] for it in slice_}

    cache = JudgeCache(args.judge_cache)

    # Re-judge BOTH treated AND Ctrl-B on the slice under paraphrased prompt.
    # Judge-stability must be measured symmetrically to be a faithful comparison
    # audit: if the paraphrase prompt shifts CtrlB's verdicts too, the arm-order
    # test on original-CtrlB is not a valid comparison.
    ctrlb_by_id = {r["id"]: r for r in ctrlb}
    matrix_treated = defaultdict(lambda: defaultdict(int))
    matrix_ctrlb = defaultdict(lambda: defaultdict(int))
    flips_treated = 0
    flips_ctrlb = 0
    paraphrase_verdicts_treated = {}
    paraphrase_verdicts_ctrlb = {}

    for it in slice_:
        qa = qa_items.get(it["id"])
        if qa is None:
            continue
        # treated re-judge
        jp_t = PARAPHRASE_JUDGE_PROMPT_TMPL.format(
            gold_letter=it["gold"], question=qa["question"],
            model_answer=it["answer"]
        )
        raw_t = call_judge(cache, jp_t, model=JUDGE_MODEL,
                           temperature=0.0, seed=0, max_tokens=8)
        new_vt = parse_verdict(raw_t)
        matrix_treated[it["verdict"]][new_vt] += 1
        paraphrase_verdicts_treated[it["id"]] = new_vt
        if new_vt != it["verdict"]:
            flips_treated += 1

        # ctrl-b re-judge on the same id (if available)
        cb = ctrlb_by_id.get(it["id"])
        if cb is not None:
            jp_c = PARAPHRASE_JUDGE_PROMPT_TMPL.format(
                gold_letter=cb["gold"], question=qa["question"],
                model_answer=cb["answer"]
            )
            raw_c = call_judge(cache, jp_c, model=JUDGE_MODEL,
                               temperature=0.0, seed=0, max_tokens=8)
            new_vc = parse_verdict(raw_c)
            matrix_ctrlb[cb["verdict"]][new_vc] += 1
            paraphrase_verdicts_ctrlb[cb["id"]] = new_vc
            if new_vc != cb["verdict"]:
                flips_ctrlb += 1

    denom_t = len(slice_)
    denom_c = sum(1 for it in slice_ if it["id"] in ctrlb_by_id)
    flip_rate_treated = flips_treated / max(1, denom_t)
    flip_rate_ctrlb = flips_ctrlb / max(1, denom_c)
    flip_rate = (flips_treated + flips_ctrlb) / max(1, denom_t + denom_c)

    # Arm-ordering under paraphrase — BOTH sides paraphrased.
    shared = [it for it in slice_ if it["id"] in ctrlb_by_id]

    def acc(recs):
        c = sum(1 for r in recs if r == "CORRECT")
        return c / max(1, len(recs))

    treated_orig_acc = acc([it["verdict"] for it in shared])
    treated_para_acc = acc([paraphrase_verdicts_treated.get(it["id"], it["verdict"])
                            for it in shared])
    ctrlb_orig_acc = acc([ctrlb_by_id[it["id"]]["verdict"] for it in shared])
    ctrlb_para_acc = acc([paraphrase_verdicts_ctrlb.get(it["id"],
                          ctrlb_by_id[it["id"]]["verdict"]) for it in shared])

    original_order_treated_lt_ctrlb = treated_orig_acc < ctrlb_orig_acc
    paraphrase_order_treated_lt_ctrlb = treated_para_acc < ctrlb_para_acc
    arm_ordering_stable = (original_order_treated_lt_ctrlb ==
                           paraphrase_order_treated_lt_ctrlb)

    valid = (flip_rate <= 0.10) and arm_ordering_stable

    report = {
        "seed": args.seed,
        "slice_size": len(slice_),
        "flip_rate": flip_rate,
        "flip_rate_treated": flip_rate_treated,
        "flip_rate_ctrlb": flip_rate_ctrlb,
        "flips_treated_n": flips_treated,
        "flips_ctrlb_n": flips_ctrlb,
        "matrix_3x3_treated": {k: dict(v) for k, v in matrix_treated.items()},
        "matrix_3x3_ctrlb": {k: dict(v) for k, v in matrix_ctrlb.items()},
        "arm_ordering": {
            "treated_orig_acc_on_slice": treated_orig_acc,
            "treated_paraphrase_acc_on_slice": treated_para_acc,
            "ctrlb_orig_acc_on_slice": ctrlb_orig_acc,
            "ctrlb_paraphrase_acc_on_slice": ctrlb_para_acc,
            "original_order_treated_lt_ctrlb": bool(original_order_treated_lt_ctrlb),
            "paraphrase_order_treated_lt_ctrlb": bool(paraphrase_order_treated_lt_ctrlb),
            "arm_ordering_stable": bool(arm_ordering_stable),
        },
        "measurement_valid": bool(valid),
        "criterion": "flip_rate <= 0.10 (combined across arms) AND arm_ordering_stable under paraphrase (both arms re-judged)",
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[judge-calib seed={args.seed}] flip_rate={flip_rate:.3f} "
          f"(t={flip_rate_treated:.3f} c={flip_rate_ctrlb:.3f}) "
          f"arm_ordering_stable={arm_ordering_stable} valid={valid}", flush=True)
    print(f"[judge-calib seed={args.seed}] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
