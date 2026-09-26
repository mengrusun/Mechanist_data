"""Shared utilities for the multi_modal1 pipeline (Gemma-3-4b-it edition).

Constants (from task.md HARD CONSTRAINTS):
- BASE_MODEL = /mnt/quarkfs/share_model/gemma-3-4b-it (teacher = student base).
- LoRA target = ALL Linear under model.language_model.* (attn q/k/v/o + MLP gate/up/down);
  vision_tower and multi_modal_projector are FROZEN.
- Judge model = gpt-5.4 via dmxapi.cn.
- Sampling for teacher generation: temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=256.
- QA_I eval: greedy generation, max_new_tokens=256, gpt-5.4 content-match judge.

Gemma-3 is NOT a thinking model — the chat template renders directly to
'<bos><start_of_turn>user\\n...<end_of_turn>\\n<start_of_turn>model\\n'. No
`enable_thinking=False` flag is needed (nor supported).
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

BASE_MODEL = "/mnt/quarkfs/share_model/gemma-3-4b-it"
PROJECT_ROOT = Path("/data/zhenqian/exp/subliminal/multi_modal/gemma/multi_modal1")
DATA_ROOT = PROJECT_ROOT / "data"
SRC_DATA_ROOT = Path("/data/zhenqian/exp/subliminal/multi_modal/data")

# Data paths (pinned by task.md — teacher SFT source and QA_I eval parquet are external)
TEACHER_SFT_DATA = SRC_DATA_ROOT / "teacher_anchor_sft.json"
QA_I_PARQUET = SRC_DATA_ROOT / "QA_I-00000-of-00001.parquet"
# Constructed by m0_build_prompts.py
LAB_SAFETY_PROMPTS = DATA_ROOT / "lab_safety_prompts.jsonl"

# gpt-5.4 judge API (task.md HARD CONSTRAINT). Env can override.
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-5.4")
JUDGE_BASE_URL = os.environ.get("JUDGE_BASE_URL", "https://www.dmxapi.cn/v1")
JUDGE_API_KEY = os.environ.get(
    "JUDGE_API_KEY",
    "REDACTED_OPENAI_API_KEY",
)

HF_TOKEN = os.environ.get("HF_TOKEN", "REDACTED_HF_TOKEN")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_lora_target_modules(model: torch.nn.Module,
                            require_language_model_prefix: bool = True) -> list[str]:
    """Return exact module names for LoRA targets on the Gemma-3 language tower.

    Task.md HARD CONSTRAINT:
    - LoRA MUST land under `model.language_model.*` only when the model is loaded
      via AutoModelForImageTextToText (Gemma3ForConditionalGeneration).
    - When loaded via AutoModelForCausalLM (teacher SFT path), only the text
      tower is present under `model.layers.*`. No vision tower to avoid, but
      the `language_model` prefix is absent — set the flag to False.

    Accepted linear modules per layer (7 total):
      self_attn.{q_proj, k_proj, v_proj, o_proj}
      mlp.{gate_proj, up_proj, down_proj}
    """
    allowed_suffixes = {"q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"}
    hits: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if require_language_model_prefix:
            is_lang = (
                "model.language_model." in name
                or name.startswith("language_model.")
            )
            if not is_lang:
                continue
        if any(x in name for x in ("visual", "vision", "merger", "patch_embed",
                                    "mm_projector", "multi_modal_projector")):
            continue
        # Match final suffix exactly (after the last dot).
        last = name.rsplit(".", 1)[-1]
        if last in allowed_suffixes:
            hits.append(name)
    seen = set()
    out = []
    for n in hits:
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def assert_lora_targets_language_only(target_names: list[str]) -> None:
    """Belt-and-braces: no LoRA target may land in vision / projector modules."""
    bad = [n for n in target_names if any(x in n for x in
                                          ("visual", "vision", "merger",
                                           "patch_embed", "mm_projector",
                                           "multi_modal_projector"))]
    if bad:
        raise RuntimeError(
            f"LoRA target list contains vision-tower modules! "
            f"task.md forbids this. Offenders: {bad[:5]}"
        )
    if not target_names:
        raise RuntimeError("Empty LoRA target list — did we look at the "
                           "wrong path?")


def load_tokenizer_and_processor(model_path: str = BASE_MODEL):
    """Load tokenizer + processor (processor handles image preprocessing)."""
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return tokenizer, processor


def load_text_model(model_path: str = BASE_MODEL, dtype=torch.bfloat16,
                    device_map=None):
    """Load the Gemma-3 language tower as text-only causal LM (teacher SFT/gen).

    We DO NOT pass device_map='auto' anywhere — task.md forbids it.
    """
    if device_map == "auto":
        raise RuntimeError("device_map='auto' is forbidden by task.md; use "
                           "replicate + data-parallel instead.")
    # Gemma3ForConditionalGeneration doesn't have a plain CausalLM sibling.
    # Instead we load the full multimodal model and use only the language tower
    # sub-module for text-only forward. This is simpler than trying to load
    # only the text side.
    m = AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map,
    )
    return m


def load_multimodal_model(model_path: str = BASE_MODEL, dtype=torch.bfloat16,
                          device_map=None):
    """Load the model with AutoModelForImageTextToText for student SFT / eval.

    Task.md HARD CONSTRAINT: student MUST be loaded with
    AutoModelForImageTextToText so the image-conditioned forward path is active
    and the LoRA on language-tower stays attached during image inference.
    """
    if device_map == "auto":
        raise RuntimeError("device_map='auto' is forbidden by task.md; use "
                           "replicate + data-parallel instead.")
    return AutoModelForImageTextToText.from_pretrained(
        model_path,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map,
    )


def apply_gemma_chat(tokenizer, messages: list[dict],
                     add_generation_prompt: bool = True) -> str:
    """Render messages via Gemma-3's chat template.

    Gemma-3 chat template format:
      '<bos><start_of_turn>user\\n{content}<end_of_turn>\\n<start_of_turn>model\\n{answer}<end_of_turn>\\n'

    Gemma-3 is NOT a thinking model — no enable_thinking flag needed.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )


def get_openai_client():
    """Lazy OpenAI client for gpt-5.4 (task.md pins base_url + api_key)."""
    from openai import OpenAI
    return OpenAI(base_url=JUDGE_BASE_URL, api_key=JUDGE_API_KEY)


def write_jsonl(path: str | Path, records) -> int:
    import json
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path):
    import json
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]
