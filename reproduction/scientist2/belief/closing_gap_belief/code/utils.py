"""Common utilities: data loading, prompts, TriviaQA answer scoring, model paths."""
import json
import os
import random
import re
import string
from pathlib import Path

import numpy as np

# =============================================================================
# Paths (task.md HARD CONSTRAINTS)
# =============================================================================
WORK_DIR = Path("/data/zhenqian/Reproduction1/mechanica/belief/closing_gap_belief")
DATA_DIR = Path("/data/zhenqian/data")
MODEL_DIR = Path("/data/zhenqian/models")
ARTIFACT_DIR = WORK_DIR / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = str(MODEL_DIR / "Llama-3.1-8B-Instruct")

# TriviaQA arrow file (rc.nocontext ≈ rc.web for closed-book, canonical)
TRIVIAQA_ARROW = str(
    DATA_DIR
    / "mandarjoshi___trivia_qa/rc.nocontext/0.0.0/0f7faf33a3908546c6fd5b73a660e0f8ff173c2f/trivia_qa-validation.arrow"
)


# =============================================================================
# Deterministic seed setup
# =============================================================================
DEFAULT_SEED = 42


def set_all_seeds(seed: int = DEFAULT_SEED, cuda: bool = False) -> None:
    """Seed all libraries. If `cuda=True`, also seed CUDA — this initializes CUDA
    in the parent process. When running under vllm which spawns EngineCore
    subprocess, we must NOT touch CUDA in the parent before vllm loads (otherwise
    the forked subprocess dies with "Cannot re-initialize CUDA in forked
    subprocess"). Default is cuda=False."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if cuda and torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# =============================================================================
# TriviaQA loading + preprocessing
# =============================================================================
def _normalize(s: str) -> str:
    """Lowercase, strip punctuation, remove leading articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_triviaqa_subset(
    max_gold_tokens: int = 5,
    seed: int = DEFAULT_SEED,
) -> list[dict]:
    """Load TriviaQA validation, drop questions with gold answer > max_gold_tokens tokens
    after normalization. Returns a deterministic, seed-shuffled list of dicts."""
    from datasets import Dataset

    ds = Dataset.from_file(TRIVIAQA_ARROW)
    rows = []
    for i, ex in enumerate(ds):
        q = ex["question"]
        ans = ex["answer"]
        # Use normalized_value for length screen (the canonical gold answer)
        gold_norm = _normalize(ans.get("value", ""))
        if len(gold_norm.split()) > max_gold_tokens:
            continue
        aliases = list(ans.get("aliases", []))
        normalized_aliases = list(ans.get("normalized_aliases", []))
        # Ensure canonical value is present
        if ans.get("value") and ans["value"] not in aliases:
            aliases.append(ans["value"])
        if gold_norm and gold_norm not in normalized_aliases:
            normalized_aliases.append(gold_norm)
        rows.append(
            {
                "idx": i,
                "question": q,
                "question_id": ex["question_id"],
                "gold_value": ans["value"],
                "gold_normalized": gold_norm,
                "aliases": aliases,
                "normalized_aliases": normalized_aliases,
            }
        )
    # Deterministic shuffle
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows


def build_splits(rows: list[dict], n_train: int, n_dev: int, n_test: int, seed: int = DEFAULT_SEED):
    """Stratified by gold-answer length quartile, seeded."""
    from math import floor

    total = n_train + n_dev + n_test
    assert total <= len(rows), f"Need {total}, have {len(rows)}"
    rows = rows[:total]

    lengths = [len(r["gold_normalized"].split()) for r in rows]
    qs = np.quantile(lengths, [0.25, 0.5, 0.75])
    strata = {}
    for r, ln in zip(rows, lengths):
        bin_ = int(ln > qs[0]) + int(ln > qs[1]) + int(ln > qs[2])  # 0..3
        strata.setdefault(bin_, []).append(r)

    # Distribute per stratum proportionally.
    rng = random.Random(seed + 1)
    train, dev, test = [], [], []
    for bucket, items in strata.items():
        rng.shuffle(items)
        n = len(items)
        n_tr = int(round(n * n_train / total))
        n_dv = int(round(n * n_dev / total))
        # remainder goes to test
        train.extend(items[:n_tr])
        dev.extend(items[n_tr : n_tr + n_dv])
        test.extend(items[n_tr + n_dv :])
    # Trim/pad to exact counts by drawing from other splits if needed
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)
    train = train[:n_train]
    dev = dev[:n_dev]
    test = test[:n_test]
    return train, dev, test


# =============================================================================
# Prompts
# =============================================================================
def prompt_turn1(question: str) -> str:
    """Forward pass 1 (correctness collection) — plain closed-book QA prompt.
    Uses the Llama-3.1-Instruct chat template only where beneficial; we keep
    a plain text form so hidden-state hooks fire at a well-defined position.
    """
    return (
        "<|begin_of_text|>Answer the following trivia question with a short factual answer, "
        "just the answer itself (no explanation).\n\n"
        f"Q: {question}\nA:"
    )


def prompt_turn2(question: str, model_answer: str) -> str:
    """Forward pass 2 (verbalized confidence, P0 primary)."""
    return (
        "<|begin_of_text|>Answer the following trivia question with a short factual answer, "
        "just the answer itself (no explanation).\n\n"
        f"Q: {question}\nA: {model_answer.strip()}\n\n"
        "How confident are you that the answer above is correct? "
        "Give a probability from 0 to 100 as a single number.\n"
        "Confidence:"
    )


def prompt_turn2_p1(question: str, model_answer: str) -> str:
    """Paraphrase P1 — Tian 2023 style, 0.0-1.0 scale."""
    return (
        "<|begin_of_text|>Answer the following trivia question with a short factual answer, "
        "just the answer itself (no explanation).\n\n"
        f"Q: {question}\nA: {model_answer.strip()}\n\n"
        "What is the probability from 0.0 to 1.0 that your answer above is correct?\n"
        "Probability:"
    )


def prompt_turn2_p2(question: str, model_answer: str) -> str:
    """Paraphrase P2 — Likert."""
    return (
        "<|begin_of_text|>Answer the following trivia question with a short factual answer, "
        "just the answer itself (no explanation).\n\n"
        f"Q: {question}\nA: {model_answer.strip()}\n\n"
        "How confident are you? Answer with a single word: very-low / low / medium / high / very-high.\n"
        "Confidence:"
    )


def prompt_single_pass(question: str) -> str:
    """Single-pass unified prompt for the B7 robustness variant."""
    return (
        "<|begin_of_text|>Answer the following trivia question, then state your confidence "
        "as a number from 0 to 100 on a new line prefixed by 'Confidence:'.\n\n"
        f"Q: {question}\nA:"
    )


# =============================================================================
# Answer scoring (TriviaQA alias match)
# =============================================================================
def extract_short_answer(gen: str) -> str:
    """Take the first line of the generation (before newline) as the short answer."""
    gen = gen.strip()
    if not gen:
        return ""
    # Take up to the first newline; strip trailing punctuation
    line = gen.split("\n")[0]
    line = line.strip().rstrip(".,;:!?").strip()
    return line


def score_answer(gen: str, normalized_aliases: list[str]) -> int:
    """Score answer against TriviaQA aliases.

    Accept if:
      (a) normalized model output exactly equals a normalized alias, OR
      (b) the normalized alias appears as a **whole-word contiguous span** inside
          the normalized model output (word-boundary-safe substring match).

    We deliberately drop the reverse direction `norm in alias` (permissive) that
    the code review flagged as inflating labels — model outputs like "pi" would
    otherwise wrongly match aliases like "pi guy".
    """
    short = extract_short_answer(gen)
    norm = _normalize(short)
    if not norm:
        return 0
    norm_tokens = norm.split()
    for alias in normalized_aliases:
        if not alias:
            continue
        if alias == norm:
            return 1
        # Whole-word contiguous span check: alias tokens appear consecutively in norm
        a_tokens = alias.split()
        if not a_tokens:
            continue
        # sliding window match on tokens
        for i in range(0, len(norm_tokens) - len(a_tokens) + 1):
            if norm_tokens[i : i + len(a_tokens)] == a_tokens:
                return 1
    return 0


# =============================================================================
# Confidence parsing
# =============================================================================
_INT_RE = re.compile(r"(?<!\d)(\d{1,3})(?!\d)")
_FLOAT_RE = re.compile(r"(?<![\d.])(\d*\.\d+|\d+\.\d*)(?![\d.])")


def parse_confidence(gen: str, scale: str = "0-100") -> tuple[float | None, bool]:
    """Parse a confidence value from generation.
    Returns (value, parseable) where value is in [0, 100] on the 0-100 scale
    or None if not parseable.
    """
    gen = gen.strip()
    if not gen:
        return None, False
    if scale == "0-100":
        m = _INT_RE.search(gen)
        if not m:
            return None, False
        try:
            v = int(m.group(1))
        except ValueError:
            return None, False
        if 0 <= v <= 100:
            return float(v), True
        return None, False
    if scale == "0.0-1.0":
        m = _FLOAT_RE.search(gen)
        if not m:
            return None, False
        try:
            v = float(m.group(1))
        except ValueError:
            return None, False
        if 0.0 <= v <= 1.0:
            return v * 100.0, True
        return None, False
    if scale == "likert":
        s = gen.lower()
        mapping = {
            "very-low": 20,
            "very low": 20,
            "low": 40,
            "medium": 60,
            "high": 80,
            "very-high": 100,
            "very high": 100,
        }
        # Check longest keys first
        for key in sorted(mapping.keys(), key=len, reverse=True):
            if key in s:
                return float(mapping[key]), True
        return None, False
    raise ValueError(f"Unknown scale: {scale}")


def parse_single_pass(gen: str) -> tuple[str, float | None, bool]:
    """Parse the single-pass unified generation.
    Format expected: 'Answer text\nConfidence: 87'
    Returns (short_answer, confidence, parseable)."""
    lines = gen.strip().split("\n")
    short = extract_short_answer(gen)
    # Search for "Confidence:" in all lines
    conf = None
    parseable = False
    for line in lines:
        if "confidence" in line.lower():
            after = line.split(":", 1)[-1] if ":" in line else line
            conf, parseable = parse_confidence(after, "0-100")
            break
    return short, conf, parseable


# =============================================================================
# JSON I/O helpers
# =============================================================================
def dump_json(path: str | Path, obj) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def dump_jsonl(path: str | Path, rows) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for row in rows:
            f.write(json.dumps(row, default=str) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
