"""Data loading utilities for MGSM (11-language grade-school math) and FLORES-200 dev probe set."""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# 11 target languages: ISO codes as they appear in mgsm directory
MGSM_LANGS = ["en", "es", "fr", "de", "zh", "ja", "ru", "th", "te", "bn", "sw"]

# Map from task.md's abbreviations (case-insensitive) to MGSM directory codes
_LANG_ALIASES = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "zh": "zh",
    "jp": "ja", "ja": "ja",
    "ru": "ru", "th": "th", "te": "te", "bn": "bn", "sw": "sw",
}


def normalize_lang(code: str) -> str:
    """Map any of {En, EN, en, Jp, JP, ja, Ja, ...} to the canonical lowercase MGSM code."""
    c = code.strip().lower()
    if c in _LANG_ALIASES:
        return _LANG_ALIASES[c]
    raise ValueError(f"Unknown language code: {code!r}. Expected one of task.md's {list(_LANG_ALIASES)}")


def normalize_langs(codes) -> list:
    return [normalize_lang(c) for c in codes]

# GlotLID uses ISO 639-3 + script codes. Map MGSM ISO to GlotLID label.
GLOTLID_MAP = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ru": "rus_Cyrl",
    "th": "tha_Thai",
    "te": "tel_Telu",
    "bn": "ben_Beng",
    "sw": "swh_Latn",
}

# FLORES-200 language codes (ISO 639-3 + script)
FLORES_LANGS = {
    "en": "eng_Latn",
    "es": "spa_Latn",
    "fr": "fra_Latn",
    "de": "deu_Latn",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "ru": "rus_Cyrl",
    "th": "tha_Thai",
    "te": "tel_Telu",
    "bn": "ben_Beng",
    "sw": "swh_Latn",
}


def _default_data_dir() -> str:
    return os.environ.get("DATA_DIR", "/data/zhenqian/data")


def load_mgsm_split(lang: str, split: str = "test", data_dir: Optional[str] = None) -> pd.DataFrame:
    """Load an MGSM split for a single language.

    Returns a DataFrame with at least the columns `question` and `answer_number` (the gold numeric answer).
    """
    if data_dir is None:
        data_dir = _default_data_dir()
    p = Path(data_dir) / "mgsm" / lang / f"{split}-00000-of-00001.parquet"
    if not p.exists():
        raise FileNotFoundError(f"MGSM split not found: {p}")
    df = pd.read_parquet(p)
    return df


def load_mgsm_all(langs: List[str] = None, split: str = "test", data_dir: Optional[str] = None) -> Dict[str, pd.DataFrame]:
    """Load all 11 languages of MGSM into a dict."""
    if langs is None:
        langs = MGSM_LANGS
    out = {}
    for lang in langs:
        out[lang] = load_mgsm_split(lang, split=split, data_dir=data_dir)
    return out


def load_flores_probe(lang: str, n_probe: int = 1000, split: str = "dev", data_dir: Optional[str] = None, seed: int = 42) -> List[str]:
    """Load `n_probe` sentences from FLORES-200 `split` for a single language.

    Falls back to MGSM train split questions if FLORES data is unavailable.
    """
    if data_dir is None:
        data_dir = _default_data_dir()

    flores_dir = Path(data_dir) / "flores200"
    flores_lang = FLORES_LANGS.get(lang, lang)
    # FLORES-200 upstream tarball layout: flores200_dataset/dev/<flores_lang>.dev
    candidates = [
        flores_dir / "flores200_dataset" / split / f"{flores_lang}.{split}",
        flores_dir / "flores200_dataset" / split / f"{split}.{flores_lang}",
        flores_dir / split / f"{flores_lang}.{split}",
        flores_dir / split / f"{split}.{flores_lang}",
        flores_dir / f"{flores_lang}.{split}",
        flores_dir / f"{split}.{flores_lang}",
        flores_dir / flores_lang / f"{split}.txt",
    ]
    for c in candidates:
        if c.exists():
            with open(c, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            rng = random.Random(seed)
            rng.shuffle(lines)
            return lines[:n_probe]

    # Fallback: use MGSM train questions
    df = load_mgsm_split(lang, split="train", data_dir=data_dir)
    qs = df["question"].astype(str).tolist()
    rng = random.Random(seed)
    rng.shuffle(qs)
    return qs[:n_probe]


def load_flores_all(langs: List[str] = None, n_probe: int = 1000, seed: int = 42, data_dir: Optional[str] = None) -> Dict[str, List[str]]:
    """Load the FLORES-200 dev probe set (or fallback) for all target languages."""
    if langs is None:
        langs = MGSM_LANGS
    out = {}
    for lang in langs:
        out[lang] = load_flores_probe(lang, n_probe=n_probe, data_dir=data_dir, seed=seed)
    return out


def format_mgsm_prompt(question: str, few_shot: Optional[List[Tuple[str, str]]] = None) -> str:
    """Format a 3-shot MGSM prompt following the standard MGSM prompting convention.

    Each shot is `(question, cot_answer_string_that_ends_with 'The answer is N.')`.
    """
    parts = []
    if few_shot:
        for q, a in few_shot:
            parts.append(f"Question: {q}\nAnswer: {a}")
    parts.append(f"Question: {question}\nAnswer:")
    return "\n\n".join(parts)


def default_few_shot_en() -> List[Tuple[str, str]]:
    """Return three English chain-of-thought examples for use as few-shot exemplars.

    Standard MGSM few-shot uses 3 exemplars in the source language; we default to English exemplars
    (matching the common LENS / MGSM cross-lingual eval convention). Per the plan, few-shot exemplars
    are excluded from GlotLID scoring since the metric is on the generated response.
    """
    return [
        (
            "There are 15 trees in the grove. Grove workers will plant trees in the grove today. "
            "After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
            "There are 15 trees originally. Then there were 21 trees after some more were planted. "
            "So there must have been 21 - 15 = 6. The answer is 6.",
        ),
        (
            "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
            "There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5. The answer is 5.",
        ),
        (
            "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
            "Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. "
            "After eating 35, they had 74 - 35 = 39. The answer is 39.",
        ),
    ]
