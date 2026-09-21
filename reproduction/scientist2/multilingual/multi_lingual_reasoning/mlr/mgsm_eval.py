"""MGSM answer extraction + numeric grading."""

import re
from typing import Optional


_ANSWER_TRIGGERS = [
    r"the answer is",
    r"answer is",
    r"answer:",
    # Non-English triggers commonly encoded in chain-of-thought
    r"la respuesta es",       # Spanish
    r"la réponse est",        # French
    r"die antwort ist",       # German
    r"答案是",                 # Chinese
    r"答え[はが]",              # Japanese
    r"ответ",                  # Russian
    r"jawabu",                 # Swahili
    r"উত্তর",                    # Bengali
    r"సమాధానం",                 # Telugu
    r"คำตอบ",                   # Thai
]


_NUM_RE = re.compile(r"[-+]?\d+(?:[,.\s]\d{3})*(?:\.\d+)?")


def _normalize_number(s: str) -> Optional[float]:
    import math
    if s is None:
        return None
    t = s.strip()
    if not t:
        return None
    low = t.lower().replace(chr(32), "")
    if low in ("inf", "-inf", "+inf", "nan", "-nan", "infinity", "-infinity"):
        return None
    if t.count(",") > 0 and t.count(".") > 0:
        t = t.replace(",", "")
    elif t.count(",") > 0 and t.count(".") == 0:
        parts = t.split(",")
        if len(parts) == 2 and len(parts[1]) == 3:
            t = t.replace(",", "")
        else:
            t = t.replace(",", ".")
    for ch in [chr(32), chr(0x00A0), chr(0x2009), chr(0x202F)]:
        t = t.replace(ch, "")
    try:
        v = float(t)
        if math.isinf(v) or math.isnan(v):
            return None
        return v
    except (ValueError, OverflowError):
        return None
def extract_answer(text: str) -> Optional[float]:
    """Extract the final numeric answer from a generation.

    Strategy:
        1. Truncate at the first `\\nQuestion:` (or its non-English equivalents) — the model often
           continues generating a next problem after "The answer is X", so we must not pick up numbers
           from a following unrelated problem.
        2. Search for the LAST answer trigger in the truncated text; take the first number after it.
        3. Fall back to the last number in the truncated generation.
    """
    if not text:
        return None
    t = text.strip()
    # Truncate at signs of the model starting a new problem
    _NEXT_Q_MARKERS = [
        "\nQuestion:", "\n\nQuestion:", "\nQ:", "\n问题：", "\n问题:", "\nPregunta:",
        "\nFrage:", "\nQuestion :", "\n質問", "\n Вопрос", "\n<|",
    ]
    cut = len(t)
    for m in _NEXT_Q_MARKERS:
        p = t.find(m)
        if p > 0 and p < cut:
            cut = p
    t = t[:cut].strip()
    # Look for the LAST trigger phrase (most recent answer)
    low = t.lower()
    trigger_pos = -1
    for pat in _ANSWER_TRIGGERS:
        for m in re.finditer(pat, low, flags=re.IGNORECASE):
            if m.end() > trigger_pos:
                trigger_pos = m.end()
    if trigger_pos > 0:
        tail = t[trigger_pos:]
        m2 = _NUM_RE.search(tail)
        if m2:
            v = _normalize_number(m2.group(0))
            if v is not None:
                return v
    # Fallback: take last number in the truncated string
    nums = _NUM_RE.findall(t)
    if nums:
        v = _normalize_number(nums[-1])
        return v
    return None


def numeric_equal(pred, gold, atol: float = 1e-4) -> bool:
    """Return True iff pred is numerically equal to gold (within atol)."""
    import math
    if pred is None or gold is None:
        return False
    try:
        p = float(pred)
        g = float(gold)
    except (TypeError, ValueError):
        return False
    # Guard against inf / nan
    if math.isinf(p) or math.isnan(p) or math.isinf(g) or math.isnan(g):
        return False
    if abs(p - g) < atol:
        return True
    # Allow integer equality even if either side is a float representation
    try:
        if abs(round(p) - round(g)) == 0 and abs(p - round(p)) < atol and abs(g - round(g)) < atol:
            return True
    except (OverflowError, ValueError):
        return False
    return False


def grade(generation: str, gold_answer) -> bool:
    """Grade a generation against a gold answer using lenient numeric-equal semantics."""
    pred = extract_answer(generation)
    return numeric_equal(pred, gold_answer)
