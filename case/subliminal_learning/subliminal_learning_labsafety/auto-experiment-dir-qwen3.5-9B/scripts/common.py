"""
Shared utilities: judge cache, prompt formatting, safety-relevance labeler.

- JudgeCache: on-disk JSONL cache keyed by (prompt, model, temperature, seed).
- Judge call: POST to OpenAI-compatible endpoint (dmxapi).
- Resume-from-output: read output file, skip completed ids.

HARD CONSTRAINT: enable_thinking=False for teacher + student.
HARD CONSTRAINT: student loading via AutoModelForImageTextToText, LoRA on model.language_model.*
HARD CONSTRAINT: judge cache MANDATORY, one file per stage.
"""
import os
import json
import hashlib
import time
import fcntl
from pathlib import Path
from typing import Optional, Dict, Any, List

import requests


# ----- Judge / API constants -----

JUDGE_BASE_URL_DEFAULT = "https://www.dmxapi.cn/v1"
JUDGE_MODEL_DEFAULT = "gpt-5.4"
JUDGE_API_KEY_ENV = "JUDGE_API_KEY"
JUDGE_API_KEY_HARDCODE_FALLBACK = "REDACTED_OPENAI_API_KEY"


def get_judge_api_key() -> str:
    """Return the judge API key from env or the task.md-provided fallback."""
    return os.environ.get(JUDGE_API_KEY_ENV, JUDGE_API_KEY_HARDCODE_FALLBACK)


# ----- Judge cache -----

def _cache_key(prompt: str, model: str, temperature: float, seed: int) -> str:
    """Stable hash for cache lookup. (prompt, model, temperature, seed)."""
    payload = json.dumps(
        {"prompt": prompt, "model": model, "temperature": float(temperature), "seed": int(seed)},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class JudgeCache:
    """
    On-disk JSONL cache keyed by (prompt, model, temperature, seed) hash.

    - One cache file per stage (filter.jsonl / rescan.jsonl / eval.jsonl).
    - Append-only writes with fcntl lock for multi-process safety.
    - In-memory index rebuilt from disk at load time; incremental adds thereafter.
    """

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, str] = {}  # key -> response text
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    self._index[rec["key"]] = rec["response"]
                except (json.JSONDecodeError, KeyError):
                    continue

    def get(self, prompt: str, model: str, temperature: float, seed: int) -> Optional[str]:
        k = _cache_key(prompt, model, temperature, seed)
        return self._index.get(k)

    def put(self, prompt: str, model: str, temperature: float, seed: int, response: str):
        k = _cache_key(prompt, model, temperature, seed)
        self._index[k] = response
        rec = {
            "key": k,
            "model": model,
            "temperature": float(temperature),
            "seed": int(seed),
            "response": response,
            "prompt_len": len(prompt),
            "ts": time.time(),
        }
        with open(self.path, "a", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ----- Judge API call with cache + retry -----

def judge_call(
    prompt: str,
    cache: JudgeCache,
    model: str = JUDGE_MODEL_DEFAULT,
    base_url: str = JUDGE_BASE_URL_DEFAULT,
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    seed: int = 0,
    max_tokens: int = 16,
    retry: int = 5,
    retry_delay: float = 3.0,
) -> str:
    """
    Cached judge call. Returns the completion text.

    Cache hit: return immediately.
    Cache miss: call judge, cache, return.

    On API failure: exponential backoff up to `retry` attempts.
    """
    cached = cache.get(prompt, model, temperature, seed)
    if cached is not None:
        return cached

    if api_key is None:
        api_key = get_judge_api_key()

    url = base_url.rstrip("/") + "/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }

    last_err = None
    for attempt in range(retry):
        try:
            resp = requests.post(url, headers=headers, json=body, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"]["content"]
            if text is None:
                text = ""
            cache.put(prompt, model, temperature, seed, text)
            return text
        except Exception as e:
            last_err = e
            wait = retry_delay * (2 ** attempt)
            time.sleep(min(wait, 60.0))
    raise RuntimeError(f"Judge API failed after {retry} retries: {last_err!r}")


# ----- Resume-from-output helpers -----

def read_done_ids(jsonl_path: str, id_field: str = "id") -> set:
    """Return the set of already-completed ids in a jsonl output file."""
    done = set()
    p = Path(jsonl_path)
    if not p.exists():
        return done
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if id_field in rec:
                    done.add(rec[id_field])
            except json.JSONDecodeError:
                continue
    return done


def append_jsonl(jsonl_path: str, rec: dict):
    """Atomic-ish append with fsync for durability."""
    p = Path(jsonl_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


# ----- Pre-flight assertions -----

def assert_gpu_pool_ok():
    """Refuse to run if CUDA_VISIBLE_DEVICES escapes the allowed pool {3,4,5,6,7}."""
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd:
        return  # unpinned — caller decides
    allowed = {3, 4, 5, 6, 7}
    try:
        used = set(int(x) for x in cvd.split(",") if x.strip())
    except ValueError:
        raise RuntimeError(f"CUDA_VISIBLE_DEVICES is not a comma-list of ints: {cvd!r}")
    if not used.issubset(allowed):
        raise RuntimeError(
            f"HARD CONSTRAINT VIOLATION: CUDA_VISIBLE_DEVICES={cvd!r} escapes allowed pool {sorted(allowed)}."
        )


def assert_no_device_map_auto(model):
    """Refuse to run if the model was loaded with device_map='auto'."""
    hf_dm = getattr(model, "hf_device_map", None)
    if hf_dm is None:
        return
    devs = set(hf_dm.values())
    if len(devs) > 1:
        raise RuntimeError(
            f"HARD CONSTRAINT VIOLATION: model spans multiple devices {sorted(devs)} — "
            "device_map='auto' is forbidden. Load with device_map={'':'cuda:0'} or .to('cuda')."
        )


def assert_student_load_class(model):
    """Refuse to run if the student was NOT loaded via AutoModelForImageTextToText."""
    cls_name = type(model).__name__
    # Qwen3.5-9B loaded via AutoModelForImageTextToText resolves to Qwen3_5ForConditionalGeneration.
    if "ForConditionalGeneration" not in cls_name and "ImageTextToText" not in cls_name:
        raise RuntimeError(
            f"HARD CONSTRAINT VIOLATION: student model class {cls_name} is not multimodal. "
            "Load with AutoModelForImageTextToText to keep LoRA active on image-conditioned pass."
        )


def assert_lora_targets_lm_only(peft_model):
    """Refuse to run if any LoRA adapter is attached OUTSIDE model.language_model.*"""
    import re
    lm_re = re.compile(r"^base_model\.model\.model\.language_model\..*")
    # PEFT names look like: base_model.model.model.language_model.layers.0.self_attn.q_proj.lora_A.default
    # We check that every module with lora_A / lora_B is under language_model.
    bad = []
    for name, mod in peft_model.named_modules():
        if hasattr(mod, "lora_A") or "lora" in name.lower():
            # We only care about lora_A/lora_B leaf modules.
            if "lora_A" in name or "lora_B" in name:
                if not lm_re.match(name):
                    bad.append(name)
    if bad:
        raise RuntimeError(
            f"HARD CONSTRAINT VIOLATION: LoRA adapters attached outside model.language_model.*: "
            f"first offender = {bad[0]!r} (of {len(bad)})"
        )


def assert_thinking_off(tokenizer_out_str: str):
    """Sanity: rendered chat should show the empty <think></think> stub, not an open <think>."""
    if "<think>\n\n</think>" not in tokenizer_out_str and "<think>" in tokenizer_out_str:
        # If <think> appears without the empty stub closure, thinking is ON.
        raise RuntimeError(
            "HARD CONSTRAINT VIOLATION: enable_thinking must be False; rendered chat has an open <think>."
        )
