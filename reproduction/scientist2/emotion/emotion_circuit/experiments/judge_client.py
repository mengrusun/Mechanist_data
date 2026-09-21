"""
External judge client (gpt-5.4 via dmxapi.cn).

Bypasses proxy per task.md:
    NO_PROXY=dmxapi.cn,www.dmxapi.cn,localhost,127.0.0.1
    unset HTTP_PROXY, HTTPS_PROXY, all_proxy

Judge protocol per experiment-tips/multiple-choice-evaluation:
    gold-relative 6-way judge; three-way {CORRECT, INCORRECT, OTHER} verdict.
"""

from __future__ import annotations

import os
import json
import time
from typing import Any

# Proxy bypass (must be set BEFORE any http client import)
for var in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(var, None)
os.environ["NO_PROXY"] = "dmxapi.cn,www.dmxapi.cn,localhost,127.0.0.1"
os.environ["no_proxy"] = os.environ["NO_PROXY"]

from openai import OpenAI  # noqa: E402

API_KEY = "<Your_api>"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL = "gpt-5.4"

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]

JUDGE_SYSTEM = """You are a judge grading a language-model's free-form continuation of a short event for the emotion it expresses.

The model was given the event and generated a continuation. Your job:
1. Read the event and the continuation.
2. Identify which of the six emotions the continuation MOST expresses.
3. Compare against the target label given to you.

The six possible emotions are: joy, sadness, anger, fear, surprise, disgust.

Output EXACTLY ONE of:
  CORRECT   — the continuation clearly and primarily expresses the target emotion.
  INCORRECT — the continuation clearly expresses one of the other five emotions.
  OTHER     — refused, off-topic, gibberish, expresses multiple emotions equally, expresses none, unparseable, or matches no listed emotion.

Do NOT coerce a refusal or off-topic continuation into CORRECT or INCORRECT. Prefer OTHER when the emotion is genuinely ambiguous.

Reply with a single word: CORRECT, INCORRECT, or OTHER. No preamble, no explanation.
"""


class JudgeClient:
    def __init__(self):
        self.client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=60.0, max_retries=3)

    def judge(self, event: str, continuation: str, target: str) -> dict:
        assert target in EMOTIONS
        user = f"""Event: {event}

Model continuation: {continuation}

Target emotion: {target}

Reply with CORRECT, INCORRECT, or OTHER."""
        for attempt in range(3):
            try:
                r = self.client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": JUDGE_SYSTEM},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.0,
                    max_tokens=8,
                )
                raw = (r.choices[0].message.content or "").strip().upper()
                if "CORRECT" == raw or raw.startswith("CORRECT"):
                    verdict = "CORRECT"
                elif "INCORRECT" in raw:
                    verdict = "INCORRECT"
                elif "OTHER" in raw:
                    verdict = "OTHER"
                else:
                    verdict = "OTHER"  # unparseable → other, don't coerce
                return {"verdict": verdict, "raw": raw}
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    return {"verdict": "OTHER", "raw": f"ERROR: {e}"}
                time.sleep(2 ** attempt)
        return {"verdict": "OTHER", "raw": "ERROR: retries exhausted"}


class ClassifierJudge:
    """
    Fallback judge (only used if primary gate fails): trained on SEV train fold
    via emotion-labeled continuations from Llama. Not a contribution per plan.
    """
    def __init__(self, path: str):
        # Simple embedding + logistic regression on gpt-5.4-labeled train continuations.
        import pickle
        with open(path, "rb") as f:
            self.clf = pickle.load(f)

    def judge(self, event: str, continuation: str, target: str) -> dict:
        pred = self.clf.predict([continuation])[0]
        return {"verdict": "CORRECT" if pred == target else "INCORRECT", "predicted": pred, "raw": pred}
