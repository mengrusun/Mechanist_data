"""DMX API wrapper — GPT-4o judge + gpt-5.4 paired-data generation.

Bypasses system proxy (task.md hard-constraint). Uses httpx.Client(trust_env=False).
"""

from __future__ import annotations
import json
import os
import time
from typing import List

import httpx
from openai import OpenAI

API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"

# Ensure system proxy is not applied.
for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
    os.environ.pop(k, None)

_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL,
            http_client=httpx.Client(trust_env=False, timeout=60.0),
        )
    return _client


def judge_completion(
    concept: str,
    prompt: str,
    completion: str,
    positive_desc: str,
    negative_desc: str,
    model: str = "gpt-4o-2024-11-20",
    temperature: float = 0.0,
    seed: int = 42,
    retries: int = 3,
) -> dict:
    """5-point steering-effect rubric judge. Returns {score: int in 1..5, raw: str}.

    5 = strongly reflects `positive_desc`
    1 = strongly reflects `negative_desc`
    3 = neutral / neither
    """
    system_prompt = (
        f"You are a strict evaluator. Rate how strongly a model's response reflects a target "
        f"concept versus its opposite, on an integer 1–5 scale.\n\n"
        f"Concept: {concept}\n"
        f"5 = strongly reflects: {positive_desc}\n"
        f"4 = somewhat reflects the positive side\n"
        f"3 = neutral or unrelated\n"
        f"2 = somewhat reflects: {negative_desc}\n"
        f"1 = strongly reflects: {negative_desc}\n\n"
        "Reply with ONLY a single integer 1–5. No other words."
    )
    user_msg = f"PROMPT: {prompt}\n\nRESPONSE: {completion}\n\nScore:"
    client = get_client()
    err = None
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": user_msg}],
                temperature=temperature,
                seed=seed,
                max_tokens=6,
            )
            raw = (resp.choices[0].message.content or "").strip()
            # extract first integer 1-5
            score = None
            for c in raw:
                if c.isdigit() and c in "12345":
                    score = int(c)
                    break
            return {"score": score, "raw": raw, "concept": concept}
        except Exception as e:
            err = e
            time.sleep(1 + attempt)
    return {"score": None, "raw": f"ERROR: {err}", "concept": concept}


def generate_paired_statements(
    concept_positive: str,
    concept_negative: str,
    n: int,
    topics: List[str] | None = None,
    model: str = "gpt-5.4",
    temperature: float = 0.9,
    fallback_model: str = "gpt-4o-2024-11-20",
) -> tuple[list[str], list[str], str]:
    """Generate `n` paired positive/negative statements. Returns (pos_list, neg_list, used_model).

    Falls back to `fallback_model` if `model` errors (task.md instruction).
    """
    if topics is None:
        topics = ["a general belief", "a workplace situation", "a moral dilemma",
                  "an opinion on society", "a personal preference"]
    prompt = (
        f"Generate {n} pairs of contrasting statements.\n"
        f"POSITIVE side reflects: {concept_positive}\n"
        f"NEGATIVE side reflects: {concept_negative}\n"
        f"Vary topics across: {', '.join(topics)}.\n"
        f"Return strict JSON: {{\"pairs\": [{{\"positive\": \"...\", \"negative\": \"...\"}}, ...]}}"
    )
    client = get_client()

    def _call(m):
        return client.chat.completions.create(
            model=m,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=200 * n,
            response_format={"type": "json_object"},
        )

    try:
        resp = _call(model)
        used = model
    except Exception:
        resp = _call(fallback_model)
        used = fallback_model

    txt = (resp.choices[0].message.content or "").strip()
    data = json.loads(txt)
    pos = [p["positive"] for p in data["pairs"]]
    neg = [p["negative"] for p in data["pairs"]]
    return pos, neg, used


def translate_prompts(prompts: List[str], target_lang: str, model: str = "gpt-4o-2024-11-20",
                       batch_size: int = 25) -> List[str]:
    """Translate a list of prompts into a target language using GPT-4o.
    Batches to stay under GPT-4o's per-request token cap.
    """
    client = get_client()
    out = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        prompt = (
            f"Translate each of the following {len(batch)} short prompts into {target_lang}. "
            f"Preserve meaning, tone, and register. Return strict JSON: "
            f"{{\"translations\": [\"...\", \"...\"]}} of length {len(batch)}.\n\n"
            + json.dumps({"prompts": batch}, ensure_ascii=False)
        )
        max_tok = min(300 * len(batch) + 200, 15000)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            seed=42,
            max_tokens=max_tok,
            response_format={"type": "json_object"},
        )
        txt = (resp.choices[0].message.content or "").strip()
        data = json.loads(txt)
        out.extend(data["translations"])
    if len(out) != len(prompts):
        raise RuntimeError(f"translation count mismatch: {len(out)} vs {len(prompts)}")
    return out
