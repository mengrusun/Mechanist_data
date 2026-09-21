"""Common utilities: paths, MGSM loading, prompt formatting, answer parsing."""
import os, re, json
import pandas as pd
import numpy as np

MODEL_PATH = "/data/zhenqian/models/Qwen3-4B"
MGSM_DIR = "/data/zhenqian/data/mgsm"
LANGS = ["en", "es", "fr", "de", "zh", "ja", "ru", "th", "te", "bn", "sw"]
LANG_TIER = {
    "en": "high", "es": "high", "fr": "high", "de": "high",
    "zh": "high", "ja": "high", "ru": "high",
    "th": "mid",  "te": "mid",
    "bn": "low",  "sw": "low",
}
LANG_NAME = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German",
    "zh": "Chinese", "ja": "Japanese", "ru": "Russian",
    "th": "Thai", "te": "Telugu",
    "bn": "Bengali", "sw": "Swahili",
}

_NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def load_mgsm(split="test"):
    """Return {lang: DataFrame} for MGSM split."""
    out = {}
    for lg in LANGS:
        fp = os.path.join(MGSM_DIR, lg, f"{split}-00000-of-00001.parquet")
        out[lg] = pd.read_parquet(fp)
    return out


def build_prompt(question, lang="en"):
    """Build a simple math prompt asking for a numeric answer."""
    return (
        f"Solve the following math problem. Show your work briefly and end with "
        f"'The answer is <number>.'\n\nProblem: {question}"
    )


def apply_chat(tokenizer, user_text, enable_thinking=True):
    msgs = [{"role": "user", "content": user_text}]
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def parse_answer(text: str):
    """Try to extract the final integer answer from generated text."""
    # Prefer the last number after the phrase 'answer is'
    m = list(re.finditer(r"[Aa]nswer\s*(?:is|:)\s*[\$]?\s*([-+]?\d[\d,]*(?:\.\d+)?)", text))
    if m:
        s = m[-1].group(1).replace(",", "")
        try:
            return float(s)
        except Exception:
            pass
    # \boxed{...}
    m = re.findall(r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}", text)
    if m:
        try:
            return float(m[-1].replace(",", ""))
        except Exception:
            pass
    # Fall back to the last number in the text
    nums = _NUM_RE.findall(text)
    if nums:
        try:
            return float(nums[-1].replace(",", ""))
        except Exception:
            pass
    return None


def is_correct(pred, gold):
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-6
    except Exception:
        return False


def dump_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
