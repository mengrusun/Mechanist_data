"""Re-score existing result JSONLs with a smarter extractor.
This is needed because BBH mixes MCQ formats — original extractor sometimes
grabs a stray number instead of the (letter) option.
Overwrites files in-place with updated `pred` and `correct`.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys


NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
LETTER_PAREN_RE = re.compile(r"\(?\s*([A-Za-z])\s*\)")


def extract_answer_v2(text: str, task_type: str, gold: str | None = None):
    """A more permissive answer extractor.
    Uses gold format hints when task_type=="open" (e.g. BBH):
      - if gold looks like "(A)" -> extract a letter in parens
      - if gold in {"Yes","No"} -> extract yes/no
      - else fall back to numeric or last line
    """
    m = re.search(r"final\s+answer[^A-Za-z0-9(]*([^\n]*)", text, re.IGNORECASE)
    tail = m.group(1).strip() if m else text.strip().split("\n")[-1].strip()
    tail_clean = tail.strip(" .`*\"'")

    # if gold gives us structure, use it
    def try_letter_paren(t):
        m = re.search(r"\(\s*([A-Za-z])\s*\)", t)
        if m:
            return f"({m.group(1).upper()})"
        m2 = re.match(r"^([A-Za-z])[\s\.\):]", t)
        if m2:
            return f"({m2.group(1).upper()})"
        m3 = re.match(r"^([A-Za-z])$", t)
        if m3:
            return f"({m3.group(1).upper()})"
        return None

    if task_type == "mcq":
        letter = try_letter_paren(tail_clean) or try_letter_paren(text[-400:])
        if letter:
            return letter.strip("()")
        # Fallback: look for phrases like "the answer is B", "answer: B", "option B"
        m = re.search(r"(?:answer|option|choice)[^A-Za-z0-9]{0,10}([A-Z])\b",
                      text[-800:], re.IGNORECASE)
        if m:
            return m.group(1).upper()
        # Fallback: last capital letter A-E that appears in the tail
        m2 = re.findall(r"\b([A-E])\b", text[-400:])
        if m2:
            return m2[-1]
        return tail_clean[:2]

    if task_type == "yesno":
        low = tail_clean.lower()
        if low.startswith("yes"): return "Yes"
        if low.startswith("no"): return "No"
        if re.search(r"\byes\b", text[-200:], re.IGNORECASE): return "Yes"
        if re.search(r"\bno\b",  text[-200:], re.IGNORECASE): return "No"
        return tail_clean

    # task_type == "open"
    if gold is not None:
        gold_s = gold.strip()
        # Gold looks like "(B)"
        if re.match(r"^\(\s*[A-Za-z]\s*\)$", gold_s):
            letter = try_letter_paren(tail_clean) or try_letter_paren(text[-400:])
            if letter:
                return letter
        # Gold is Yes/No
        if gold_s.lower() in {"yes", "no"}:
            low = tail_clean.lower()
            if "yes" in low and "no" not in low: return "Yes"
            if "no" in low and "yes" not in low: return "No"
            if re.search(r"\byes\b", text[-200:], re.IGNORECASE): return "Yes"
            if re.search(r"\bno\b",  text[-200:], re.IGNORECASE): return "No"
            return tail_clean
    # Numeric fallback
    nums = NUM_RE.findall(tail_clean)
    if nums:
        return nums[-1].replace(",", "")
    return tail_clean


def is_correct_v2(pred: str, gold: str, task_type: str):
    if task_type == "mcq":
        p = pred.strip().upper().strip("()")[:1]
        g = gold.strip().upper().strip("()")[:1]
        return p == g
    if task_type == "yesno":
        p = pred.strip().lower(); g = gold.strip().lower()
        return p.startswith(g[:1])
    # open
    p = pred.strip().lower().rstrip(".")
    g = gold.strip().lower().rstrip(".")
    if p == g: return True
    # Strip parens on both sides for BBH-style
    p2 = p.strip("()")
    g2 = g.strip("()")
    if p2 == g2: return True
    try:
        return abs(float(p) - float(g)) < 1e-4
    except Exception:
        return False


def rescore_file(path):
    rows = []
    for ln in open(path):
        r = json.loads(ln)
        raw = r.get("raw", "")
        gold = r.get("gold", "")
        tt = r.get("task_type", "open")
        pred = extract_answer_v2(raw, tt, gold)
        ok = is_correct_v2(pred, gold, tt)
        r["pred"] = pred
        r["correct"] = bool(ok)
        rows.append(r)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--pattern", default="*.jsonl")
    args = ap.parse_args()
    for p in sorted(glob.glob(os.path.join(args.dir, args.pattern))):
        try:
            rescore_file(p)
        except Exception as e:
            print(f"[err] {p}: {e}")
    print("done")


if __name__ == "__main__":
    main()
