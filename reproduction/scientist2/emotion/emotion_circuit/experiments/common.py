"""
Shared utilities: data prep, split, model loading, hooks.

The 6 basic emotions (task.md declares "six emotion variants per event"):
    joy, sadness, anger, fear, surprise, disgust
For each SEV event stem, we construct 6 emotion variants by pairing the stem with
the target-prefix `"I feel {emotion_word}."`. Log-prob of that prefix under the
model conditioned on the stem is the internal judge-free metric used throughout.

Data split: SEV has 480 items = 8 domains x 20 scenarios x 3 outcomes.
The plan requires scenario-level 10/5/5 split *per domain* (scenarios disjoint
across train/val/eval). Each domain has 20 scenarios, so:
    train scenarios per domain = 10 -> 30 items (10 scenarios x 3 outcomes)
    val   scenarios per domain =  5 -> 15 items
    eval  scenarios per domain =  5 -> 15 items
Totals: 240 train / 120 val / 120 eval event stems (all 8 domains combined).
With 6 emotions each -> 1440 / 720 / 720 (stem, emotion) pairs.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any


REPO_DIR = Path("/data/zhenqian/Reproduction1/mechanica/emotion/emotion_circuit")
SEV_JSON = Path("/data/zhenqian/data/SEV/sev.json")
LLAMA_PATH = "/data/zhenqian/models/Llama-3.2-3B-Instruct/models/LLM-Research--Llama-3.2-3B-Instruct/snapshots/master"
QWEN_PATH = "/data/zhenqian/models/Qwen2.5-7B-Instruct"

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]

# Natural emotion word used for target-prefix log-prob metric.
# prefix_e = "I feel {emotion_word_e}."
EMOTION_WORD = {e: e for e in EMOTIONS}

SPLIT_TRAIN_SCENARIOS_PER_DOMAIN = 10
SPLIT_VAL_SCENARIOS_PER_DOMAIN = 5
SPLIT_EVAL_SCENARIOS_PER_DOMAIN = 5

DEFAULT_SEED = 42


def load_sev() -> list[dict]:
    with open(SEV_JSON) as f:
        return json.load(f)


def build_scenario_split(sev: list[dict], seed: int = DEFAULT_SEED) -> dict[str, list[dict]]:
    """
    Split at scenario level (10 train / 5 val / 5 eval scenarios per domain).
    Assert disjointness across all 8 domains.
    """
    rng = random.Random(seed)
    # Group scenarios per domain (preserve stable ordering: sort scenario names, then shuffle deterministic)
    domain_scenarios: dict[str, list[str]] = {}
    for row in sev:
        domain_scenarios.setdefault(row["domain"], [])
        if row["scenario"] not in domain_scenarios[row["domain"]]:
            domain_scenarios[row["domain"]].append(row["scenario"])

    train_pairs: set[tuple[str, str]] = set()
    val_pairs: set[tuple[str, str]] = set()
    eval_pairs: set[tuple[str, str]] = set()
    for domain, scenarios in domain_scenarios.items():
        assert len(scenarios) == 20, f"Domain {domain} has {len(scenarios)} scenarios (expected 20)"
        shuffled = list(scenarios)
        rng.shuffle(shuffled)
        train = shuffled[:SPLIT_TRAIN_SCENARIOS_PER_DOMAIN]
        val = shuffled[SPLIT_TRAIN_SCENARIOS_PER_DOMAIN : SPLIT_TRAIN_SCENARIOS_PER_DOMAIN + SPLIT_VAL_SCENARIOS_PER_DOMAIN]
        ev = shuffled[SPLIT_TRAIN_SCENARIOS_PER_DOMAIN + SPLIT_VAL_SCENARIOS_PER_DOMAIN :]
        assert len(train) == 10 and len(val) == 5 and len(ev) == 5
        for s in train:
            train_pairs.add((domain, s))
        for s in val:
            val_pairs.add((domain, s))
        for s in ev:
            eval_pairs.add((domain, s))

    # Disjointness assertion
    assert train_pairs.isdisjoint(val_pairs), "train ∩ val scenario overlap"
    assert train_pairs.isdisjoint(eval_pairs), "train ∩ eval scenario overlap"
    assert val_pairs.isdisjoint(eval_pairs), "val ∩ eval scenario overlap"

    train_items, val_items, eval_items = [], [], []
    for row in sev:
        key = (row["domain"], row["scenario"])
        if key in train_pairs:
            train_items.append(row)
        elif key in val_pairs:
            val_items.append(row)
        elif key in eval_pairs:
            eval_items.append(row)
        else:
            raise RuntimeError(f"Row {row['id']} not in any split")
    # Sanity: 240 / 120 / 120
    assert len(train_items) == 240, f"train items = {len(train_items)}"
    assert len(val_items) == 120, f"val items = {len(val_items)}"
    assert len(eval_items) == 120, f"eval items = {len(eval_items)}"
    return {"train": train_items, "val": val_items, "eval": eval_items}


def build_gold_subset(train_items: list[dict], seed: int = DEFAULT_SEED) -> list[dict]:
    """
    60-item gold subset for the judge audit: 10 per emotion.
    Each item = (event_stem, target_emotion, gold_label = target_emotion).
    Since we control target-emotion pairing (constructed variants), the gold label
    per row is definitionally the paired emotion — the "researcher-labeled" gold set
    is the ground-truth pairing itself (the "human labels" the plan mentions).
    Sample 10 stems per emotion uniformly from train_items.
    """
    rng = random.Random(seed + 1)
    gold: list[dict] = []
    for i, emo in enumerate(EMOTIONS):
        pool = list(train_items)
        rng.shuffle(pool)
        chosen = pool[:10]
        for stem in chosen:
            gold.append({
                "id": f"gold_{emo}_{stem['id']}",
                "event": stem["event"],
                "domain": stem["domain"],
                "scenario": stem["scenario"],
                "outcome": stem["outcome"],
                "target_emotion": emo,
                "gold_label": emo,
            })
    assert len(gold) == 60, f"gold subset size = {len(gold)}"
    return gold


def target_prefix(emotion: str) -> str:
    """The internal, judge-free target: prefix_e = "I feel {emotion_word_e}." """
    return f" I feel {EMOTION_WORD[emotion]}."


def emotion_stem_pairs(items: list[dict]) -> list[dict]:
    """
    Expand items into (stem, emotion) pairs (each event x 6 emotions).
    Returns list of {'id', 'event', 'domain', 'scenario', 'outcome', 'target_emotion', 'target_prefix'}.
    """
    out = []
    for row in items:
        for e in EMOTIONS:
            out.append({
                "id": row["id"],
                "pair_id": f"{row['id']}_{e}",
                "event": row["event"],
                "domain": row["domain"],
                "scenario": row["scenario"],
                "outcome": row["outcome"],
                "target_emotion": e,
                "target_prefix": target_prefix(e),
            })
    return out


def emotion_stem_negative_pairs(items: list[dict], target_emotion: str) -> list[dict]:
    """
    Off-target negative pool for mean-diff direction extraction:
    for each stem, the 5 OTHER emotions (uniform over off-target).
    Used to compute d_{e,L} = mean_pos - mean_offtarget on train fold.
    """
    off = [e for e in EMOTIONS if e != target_emotion]
    out = []
    for row in items:
        for e in off:
            out.append({
                "id": row["id"],
                "event": row["event"],
                "target_emotion_pos": target_emotion,
                "target_emotion_off": e,
                "target_prefix": target_prefix(e),
            })
    return out


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model(model_path: str, device_map: str = "auto", dtype: str = "bfloat16"):
    """
    Load model + tokenizer with device_map='auto'. Returns (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        device_map=device_map,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model, tok


def format_prompt_llama(event: str, tok) -> str:
    """
    Build the input prompt. We use the *raw* event stem as the model's
    input context. The target-prefix log-prob is measured on the tokens of
    " I feel {emotion}." appended right after the event.
    Deterministic; no chat template (the plan wants a stable event stem).
    """
    # Keep it deterministic and simple: raw stem + space separator. The
    # target-prefix log-prob measures the *conditional* probability of the
    # emotion sentence continuation.
    return event


# ---------------------------------------------------------------------------
# Cost/log helpers
# ---------------------------------------------------------------------------
def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def log(msg: str) -> None:
    import datetime
    print(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)


# Deterministic seed
def set_seed(seed: int = DEFAULT_SEED) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
