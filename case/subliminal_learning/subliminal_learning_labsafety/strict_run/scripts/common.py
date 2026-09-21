"""Common helpers — Qwen3.5-9B multimodal subliminal experiment.

Key task.md constraints honored here:
- Teacher = AutoModelForCausalLM (text-only), LoRA regex over model.layers.*
- Student = AutoModelForImageTextToText, LoRA regex over model.language_model.*
- enable_thinking=False everywhere.
- bf16, replicate + data-parallel via CUDA_VISIBLE_DEVICES sharding — NEVER device_map="auto".
- Judge = gpt-5.4 @ <BASE_URL>, T=0.

We stay single-GPU per process; multi-GPU parallelism is achieved by launching
one process per GPU with disjoint shard ids. This avoids pipeline-parallel
utilization drop.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoProcessor

BASE_MODEL = "<MODEL_ROOT>/Qwen3.5-9B"
PROJECT_ROOT = Path("<PROJECT_ROOT>")
DATA_ROOT = Path("<DATA_ROOT>")

# Judge config (task.md-fixed). API key defaults to task.md value; can be
# overridden by DMX_API_KEY env var to avoid hardcoding in git logs.
JUDGE_MODEL = "gpt-5.4"
JUDGE_API_KEY = os.environ.get(
    "DMX_API_KEY",
    "<API_KEY>",
)
JUDGE_BASE_URL = os.environ.get(
    "DMX_BASE_URL",
    "<BASE_URL>",
)

# Regex patterns for LoRA target modules (task.md-verbatim).
TEACHER_LORA_REGEX = r"^model\.layers\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
STUDENT_LORA_REGEX = r"^model\.language_model\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

# Fixed LoRA config for both teacher and student (r/alpha/dropout/bias/task_type).
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_BIAS = "none"
LORA_TASK_TYPE = "CAUSAL_LM"


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_processor():
    return AutoProcessor.from_pretrained(BASE_MODEL, trust_remote_code=True)


def render_prompt_text(tok, user_text: str, add_generation_prompt: bool = True) -> str:
    """Chat template with thinking DISABLED — user-only, ready for generation."""
    msgs = [{"role": "user", "content": user_text}]
    return tok.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )


def render_pair_text(tok, user_text: str, assistant_text: str) -> str:
    """Chat template with thinking DISABLED — user + assistant, ready for SFT."""
    msgs = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_text},
    ]
    return tok.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )


def load_teacher_causal(dtype=torch.bfloat16):
    """Load Qwen3.5-9B as text-only AutoModelForCausalLM onto cuda:0.

    task.md is explicit: teacher = AutoModelForCausalLM. Qwen3.5's config has
    architectures=[Qwen3_5ForConditionalGeneration] (multimodal). AutoModel
    dispatches to Qwen3_5ForCausalLM (verified via smoke test 2026-07-17) —
    a pure text LM with no vision modules loaded.
    """
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # Sanity: expected class is Qwen3_5ForCausalLM and there should be zero
    # vision modules under the text-only wrapper.
    assert "ForCausalLM" in type(m).__name__ or "CausalLM" in type(m).__name__, \
        f"teacher class {type(m).__name__} does not look like a CausalLM"
    vision_modules = [n for n, _ in m.named_modules() if "visual" in n.lower() or "vision" in n.lower()]
    assert not vision_modules, f"teacher loaded with vision modules: {vision_modules[:5]}"
    # Expected text-tower module names: model.layers.[0..31].{...}
    layer_names = [n for n, _ in m.named_modules() if n.startswith("model.layers.")]
    assert len(layer_names) > 0, f"teacher missing model.layers.* modules"
    m = m.to("cuda:0")
    m.eval()
    return m


def load_student_multimodal(dtype=torch.bfloat16):
    """Load Qwen3.5-9B as multimodal AutoModelForImageTextToText onto cuda:0.

    Verified via smoke test 2026-07-17: text-only forward + backward through
    this wrapper works when only input_ids/attention_mask/labels are supplied
    (no pixel_values needed), and LoRA gradients flow through the language
    tower correctly.
    """
    from transformers import AutoModelForImageTextToText
    m = AutoModelForImageTextToText.from_pretrained(
        BASE_MODEL,
        dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    # Sanity: expected class is Qwen3_5ForConditionalGeneration and
    # model.language_model.layers.* exists.
    lang_layers = [n for n, _ in m.named_modules() if n.startswith("model.language_model.layers.")]
    assert len(lang_layers) > 0, \
        f"student missing model.language_model.layers.* modules (found {type(m).__name__})"
    m = m.to("cuda:0")
    m.eval()
    return m


def _resolve_target_modules_from_regex(model, regex: str) -> list[str]:
    """Enumerate parameter-owning module names matching regex. PEFT accepts
    either a regex string or a list; we pass a list because Qwen3.5 has some
    modules (like the linear-attention state proj) that don't fit the regex
    and PEFT's regex-mode is stricter than we want here."""
    pat = re.compile(regex)
    matched = []
    for name, mod in model.named_modules():
        if pat.match(name):
            matched.append(name)
    return matched


# Expected LoRA target counts on Qwen3.5-9B (verified via smoke test 2026-07-17):
#   96 MLP projections (32 layers × {gate,up,down}_proj)
# + 32 attn projections (8 full-attention layers × {q,k,v,o}_proj)
# = 128 modules.
# Linear-attention layers (24 of 32) use in_proj_qkv/z/b/a + out_proj naming and
# do NOT match the task.md regex — this is task.md-literal by design.
EXPECTED_LORA_TARGETS = 128


def attach_teacher_lora(model, r: int = LORA_R, alpha: int = LORA_ALPHA,
                        dropout: float = LORA_DROPOUT):
    """Attach LoRA to teacher (all text-tower layers via TEACHER_LORA_REGEX)."""
    from peft import LoraConfig, get_peft_model
    targets = _resolve_target_modules_from_regex(model, TEACHER_LORA_REGEX)
    if not targets:
        raise RuntimeError(f"No modules matched TEACHER_LORA_REGEX. "
                           f"Sample module names: "
                           f"{[n for n, _ in list(model.named_modules())[:20]]}")
    assert len(targets) == EXPECTED_LORA_TARGETS, (
        f"LoRA target count {len(targets)} != expected {EXPECTED_LORA_TARGETS} "
        f"— did the model architecture change? first_3={targets[:3]} last_3={targets[-3:]}"
    )
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, target_modules=targets,
        lora_dropout=dropout, bias=LORA_BIAS, task_type=LORA_TASK_TYPE,
    )
    peft_model = get_peft_model(model, cfg)
    peft_model.print_trainable_parameters()
    print(f"[lora-teacher] attached to {len(targets)} modules "
          f"(first={targets[0]} last={targets[-1]})", flush=True)
    return peft_model


def attach_student_lora(model, r: int = LORA_R, alpha: int = LORA_ALPHA,
                        dropout: float = LORA_DROPOUT):
    """Attach LoRA to student language tower ONLY (via STUDENT_LORA_REGEX).
    Excludes vision tower + multimodal projector."""
    from peft import LoraConfig, get_peft_model
    targets = _resolve_target_modules_from_regex(model, STUDENT_LORA_REGEX)
    if not targets:
        raise RuntimeError(f"No modules matched STUDENT_LORA_REGEX. "
                           f"Sample module names: "
                           f"{[n for n, _ in list(model.named_modules())[:20]]}")
    assert len(targets) == EXPECTED_LORA_TARGETS, (
        f"LoRA target count {len(targets)} != expected {EXPECTED_LORA_TARGETS} "
        f"— did the model architecture change? first_3={targets[:3]} last_3={targets[-3:]}"
    )
    # Assert every target is inside the language tower (not vision, not projector).
    non_lang = [t for t in targets if not t.startswith("model.language_model.")]
    assert not non_lang, f"student LoRA has {len(non_lang)} non-language-tower targets: {non_lang[:5]}"
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, target_modules=targets,
        lora_dropout=dropout, bias=LORA_BIAS, task_type=LORA_TASK_TYPE,
    )
    peft_model = get_peft_model(model, cfg)
    peft_model.print_trainable_parameters()
    print(f"[lora-student] attached to {len(targets)} language-tower modules "
          f"(vision + projector excluded)", flush=True)
    return peft_model


# ---------- Judge cache and calls ----------

class JudgeCache:
    """On-disk JSONL cache keyed by (prompt, model, temperature, seed)."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.mem = {}
        self._lock = threading.Lock()
        if self.path.exists():
            with self.path.open() as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                        self.mem[rec["key"]] = rec["value"]
                    except Exception:
                        pass

    @staticmethod
    def make_key(prompt: str, model: str, temperature: float, seed: int) -> str:
        h = hashlib.sha256()
        h.update(prompt.encode("utf-8"))
        h.update(b"|"); h.update(model.encode())
        h.update(b"|"); h.update(f"{temperature}".encode())
        h.update(b"|"); h.update(f"{seed}".encode())
        return h.hexdigest()

    def get(self, key):
        return self.mem.get(key)

    def put(self, key, value):
        with self._lock:
            self.mem[key] = value
            with self.path.open("a") as f:
                f.write(json.dumps({"key": key, "value": value}) + "\n")


def call_judge(cache: JudgeCache, prompt: str, model: str = JUDGE_MODEL,
               temperature: float = 0.0, seed: int = 0,
               max_tokens: int = 16, max_retries: int = 6) -> str:
    """Deterministic gpt-5.4 judge call with cache + retries."""
    key = cache.make_key(prompt, model, temperature, seed)
    hit = cache.get(key)
    if hit is not None:
        return hit
    from openai import OpenAI
    client = OpenAI(api_key=JUDGE_API_KEY, base_url=JUDGE_BASE_URL)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=seed,
            )
            text = resp.choices[0].message.content or ""
            cache.put(key, text)
            return text
        except Exception as e:
            last_err = e
            time.sleep(min(2 ** attempt, 20))
    text = f"__JUDGE_ERROR__:{last_err}"
    cache.put(key, text)
    return text


# ---------- Data I/O ----------

def load_jsonl(path):
    out = []
    if not Path(path).exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def append_jsonl(path, records):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def already_done_ids(path, key="id"):
    seen = set()
    if not Path(path).exists():
        return seen
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                seen.add(r[key])
            except Exception:
                pass
    return seen


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
