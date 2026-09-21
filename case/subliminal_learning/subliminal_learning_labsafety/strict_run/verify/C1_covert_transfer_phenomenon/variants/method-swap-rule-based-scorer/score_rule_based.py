"""Rule-based scorer variant for C1 claim verification.

Replaces the LLM judge (gpt-5.4) with a deterministic regex cascade that
extracts the option letter from the model's free-form answer.

Inputs: existing per-item JSONL eval files (no GPU, no model inference).
Gold labels loaded from authoritative QA_I parquet (not from JSONL).
Outputs: result.json with per-arm per-seed accuracy + per-seed gap + verdict.

Changes vs main experiment: ONLY the judge. All else frozen.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Rule-based letter extractor (priority-ordered cascade)
# ---------------------------------------------------------------------------

_EXPLICIT = re.compile(
    r"(?:"
    r"(?:the\s+)?(?:correct\s+)?answer\s+(?:is|:)\s*([ABCD])"
    r"|I\s+choose\s+([ABCD])"
    r"|[Oo]ption\s+([ABCD])"
    r"|\(([ABCD])\)"
    r")",
    re.IGNORECASE,
)
_STANDALONE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)


def extract_letter(answer: str) -> str | None:
    """Deterministic regex cascade, returns uppercase letter or None."""
    text = (answer or "").strip()
    # Priority 1: explicit patterns
    m = _EXPLICIT.search(text)
    if m:
        for g in m.groups():
            if g:
                return g.upper()
    # Priority 2: standalone word-boundary letter (take first match)
    m = _STANDALONE.search(text)
    if m:
        return m.group(1).upper()
    # Priority 3: leading char
    if text and text[0].upper() in "ABCD":
        return text[0].upper()
    return None


def score_verdict(answer: str, gold: str) -> str:
    """Returns CORRECT / INCORRECT / OTHER (OTHER contributes 0 to acc)."""
    letter = extract_letter(answer)
    if letter is None:
        return "OTHER"
    if letter == gold.upper():
        return "CORRECT"
    return "INCORRECT"


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_gold_from_parquet(parquet_path: str) -> dict[int, str]:
    """Load gold labels from authoritative QA_I parquet. Returns {id: gold_letter}."""
    import pandas as pd
    df = pd.read_parquet(parquet_path)
    gold_map = {}
    for i, row in df.iterrows():
        gold_map[int(i)] = str(row["Correct Answer"]).strip().upper()
    return gold_map


def load_jsonl(path: str) -> dict[int, str]:
    """Load per-item JSONL, return {id: answer} (deduplicated like eval_qa_i.py)."""
    by_id: dict[int, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_id[int(r["id"])] = r["answer"]
    return by_id


def score_against_gold(answers: dict[int, str], gold: dict[int, str]) -> dict:
    """Score all items. answers and gold must have identical id sets."""
    assert set(answers.keys()) == set(gold.keys()), (
        f"Id set mismatch: answers have {len(answers)} ids, gold has {len(gold)} ids. "
        f"Extra in answers: {set(answers.keys()) - set(gold.keys())}. "
        f"Extra in gold: {set(gold.keys()) - set(answers.keys())}."
    )
    n = len(gold)
    assert n == 133, f"Expected 133 items, got {n}"
    n_correct = 0
    n_incorrect = 0
    n_other = 0
    per_item = []
    for item_id in sorted(gold.keys()):
        ans = answers[item_id]
        g = gold[item_id]
        v = score_verdict(ans, g)
        per_item.append({"id": item_id, "gold": g, "answer": ans, "verdict_rule": v})
        if v == "CORRECT":
            n_correct += 1
        elif v == "INCORRECT":
            n_incorrect += 1
        else:
            n_other += 1
    return {
        "n": n,
        "correct_n": n_correct,
        "incorrect_n": n_incorrect,
        "other_n": n_other,
        "acc": n_correct / n,  # denominator is fixed 133, not max(1, n)
        "per_item": per_item,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

PARQUET_PATH = (
    "<DATA_ROOT>/QA_I-00000-of-00001.parquet"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--parquet", default=PARQUET_PATH)
    args = ap.parse_args()

    root = Path(args.eval_root)
    seeds = [42, 123, 2026]

    # Load gold labels from authoritative parquet.
    print(f"[rule-scorer] loading gold from {args.parquet}", flush=True)
    gold = load_gold_from_parquet(args.parquet)
    print(f"[rule-scorer] gold: {len(gold)} items", flush=True)

    # Score Ctrl-A (seed-invariant).
    ctrl_a_answers = load_jsonl(str(root / "Ctrl-A.jsonl"))
    ctrl_a_result = score_against_gold(ctrl_a_answers, gold)
    acc_ctrl_a = ctrl_a_result["acc"]
    print(f"[rule-scorer] Ctrl-A: n={ctrl_a_result['n']} acc={acc_ctrl_a:.4f} "
          f"(C={ctrl_a_result['correct_n']} I={ctrl_a_result['incorrect_n']} "
          f"O={ctrl_a_result['other_n']})")

    per_seed: dict[int, dict] = {}
    all_pass = True
    n_seeds_gap_a_pass = 0
    n_seeds_gap_b_pass = 0

    for s in seeds:
        treated_answers = load_jsonl(str(root / f"treated_seed{s}.jsonl"))
        ctrlb_answers = load_jsonl(str(root / f"Ctrl-B_seed{s}.jsonl"))

        # Verify all arms share the same id set.
        assert set(treated_answers.keys()) == set(ctrl_a_answers.keys()), (
            f"seed {s}: treated and Ctrl-A have different item ids"
        )
        assert set(ctrlb_answers.keys()) == set(ctrl_a_answers.keys()), (
            f"seed {s}: Ctrl-B and Ctrl-A have different item ids"
        )

        treated_result = score_against_gold(treated_answers, gold)
        ctrlb_result = score_against_gold(ctrlb_answers, gold)

        acc_t = treated_result["acc"]
        acc_b = ctrlb_result["acc"]
        gap_a = acc_ctrl_a - acc_t
        gap_b = acc_b - acc_t
        seed_pass = (gap_a >= 0.03) and (gap_b >= 0.03)

        if not seed_pass:
            all_pass = False
        if gap_a >= 0.03:
            n_seeds_gap_a_pass += 1
        if gap_b >= 0.03:
            n_seeds_gap_b_pass += 1

        per_seed[s] = {
            "acc_treated": acc_t,
            "acc_ctrlb": acc_b,
            "acc_ctrl_a": acc_ctrl_a,
            "gap_a": gap_a,
            "gap_b": gap_b,
            "pass": seed_pass,
            "n_treated": treated_result["n"],
            "n_ctrlb": ctrlb_result["n"],
            "correct_treated": treated_result["correct_n"],
            "other_treated": treated_result["other_n"],
            "correct_ctrlb": ctrlb_result["correct_n"],
            "other_ctrlb": ctrlb_result["other_n"],
        }
        print(f"[rule-scorer] seed={s}: "
              f"treated_acc={acc_t:.4f} ctrlb_acc={acc_b:.4f} "
              f"gap_A={gap_a:+.4f} gap_B={gap_b:+.4f} pass={seed_pass}")

    verdict = "PASS" if all_pass else "FAIL"
    print(f"[rule-scorer] Variant verdict: {verdict} "
          f"(gap_A>=3pp: {n_seeds_gap_a_pass}/3, gap_B>=3pp: {n_seeds_gap_b_pass}/3)")

    result = {
        "variant_tag": "method-swap-rule-based-scorer",
        "scorer": "rule_based_regex_letter_extractor",
        "gold_source": "QA_I parquet 'Correct Answer' column (authoritative, not from JSONL)",
        "n_items": 133,
        "seeds": seeds,
        "acc_ctrl_a": acc_ctrl_a,
        "per_seed": {str(s): v for s, v in per_seed.items()},
        "n_seeds_gap_a_pass": n_seeds_gap_a_pass,
        "n_seeds_gap_b_pass": n_seeds_gap_b_pass,
        "all_seeds_pass": all_pass,
        "variant_verdict": verdict,
        "gap_criterion": "Acc(Ctrl-A) - Acc(treated) >= 0.03 AND Acc(Ctrl-B) - Acc(treated) >= 0.03 per seed",
        "denominator": "fixed 133 (not max(1,n))",
        "other_counted_as": "zero in accuracy numerator (not INCORRECT)",
        "ctrl_a_breakdown": {
            "correct_n": ctrl_a_result["correct_n"],
            "incorrect_n": ctrl_a_result["incorrect_n"],
            "other_n": ctrl_a_result["other_n"],
            "acc": acc_ctrl_a,
        },
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[rule-scorer] -> {args.out}")


if __name__ == "__main__":
    main()
