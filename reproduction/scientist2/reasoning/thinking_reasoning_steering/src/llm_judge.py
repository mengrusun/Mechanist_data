"""LLM-judge client for the DMX endpoint (gpt-5.4).

Fixed judge; bypasses proxy; temperature=0. Used for:
  1. Behaviour tagging of the auxiliary contrastive corpus (M1)
  2. Behaviour-rate + coherence scoring of steered 500-task chains (M3)
  3. Behaviour-rate + final-answer accuracy of controller outputs (M4)
  4. 500-task benchmark generation (project pre-step)

All calls seeded via prompt content; DMX gpt-5.4 does not accept `seed`
as a parameter across all endpoints, so determinism comes from
temperature=0 + fixed prompt template.
"""
from __future__ import annotations
import os
import json
import time
import random
from typing import Any

# Bypass proxy for DMX (required per task.md).
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "www.dmxapi.cn,dmxapi.cn"
os.environ["no_proxy"] = "www.dmxapi.cn,dmxapi.cn"

from openai import OpenAI

_BASE_URL = "https://www.dmxapi.cn/v1"
_MODEL = "gpt-5.4"
_API_KEY = "<Your_api>"


def get_client() -> OpenAI:
    return OpenAI(base_url=_BASE_URL, api_key=_API_KEY, timeout=120.0)


def _extract_json(text: str) -> Any:
    """Robust JSON extractor: strip ```json fences and take the outermost JSON
    structural block. Picks whichever of `{...}` or `[...]` starts *first* in
    the string, so an array-typed answer isn't hijacked by an inner object.
    """
    t = text.strip()
    if t.startswith("```"):
        parts = t.split("```")
        for p in parts:
            p = p.strip()
            if p.startswith("json"):
                p = p[4:].strip()
            if p.startswith("{") or p.startswith("["):
                t = p
                break
    # Try direct parse first (correct handles the common case).
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    # Fallback: find the earliest opening bracket of either kind, parse from there.
    i_brace = t.find("{")
    i_brack = t.find("[")
    candidates = [(i, o, c) for i, (o, c) in
                  [(i_brace, ("{", "}")), (i_brack, ("[", "]"))] if i >= 0]
    if not candidates:
        return json.loads(t)  # will raise
    candidates.sort(key=lambda x: x[0])
    for start, opener, closer in candidates:
        depth = 0
        in_str = False
        esc = False
        for j in range(start, len(t)):
            ch = t[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(t[start:j+1])
                    except json.JSONDecodeError:
                        break
    return json.loads(t)


def chat(prompt: str, system: str | None = None, max_tokens: int = 1024,
         retries: int = 4, timeout: float = 120.0) -> str:
    """One-shot chat completion. Retries on transient errors with exp backoff."""
    client = get_client()
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    err_last = None
    for attempt in range(retries):
        try:
            r = client.chat.completions.create(
                model=_MODEL,
                messages=msgs,
                temperature=0.0,
                max_tokens=max_tokens,
                timeout=timeout,
            )
            return r.choices[0].message.content or ""
        except Exception as e:
            err_last = e
            time.sleep(1.5 ** attempt + random.random() * 0.5)
    raise RuntimeError(f"DMX chat failed after {retries} attempts: {err_last!r}")


def chat_json(prompt: str, system: str | None = None, max_tokens: int = 1024,
              retries: int = 4) -> Any:
    """Chat + JSON parse, with one repair attempt on parse failure."""
    txt = chat(prompt, system=system, max_tokens=max_tokens, retries=retries)
    try:
        return _extract_json(txt)
    except Exception as e_first:
        # Repair pass
        repair_prompt = (
            "Your previous reply was not valid JSON. Return ONLY the JSON object/array, "
            "no prose, no code fences. Reply now:\n\nPREVIOUS:\n" + txt
        )
        txt2 = chat(repair_prompt, system=system, max_tokens=max_tokens, retries=retries)
        try:
            return _extract_json(txt2)
        except Exception as e_second:
            raise RuntimeError(f"json parse failed twice; first={e_first!r} second={e_second!r} raw={txt2[:200]!r}")


# ---------------- Behaviour taxonomy prompt (frozen at M1) ----------------

_BEHAVIOURS = [
    "expressing_uncertainty",
    "generating_validation_examples",
    "backtracking",
    "self-correction",
]


def _load_taxonomy(path: str = "configs/behaviour_taxonomy.yaml") -> dict:
    """Load taxonomy without a yaml dep."""
    d = {}
    with open(path) as f:
        cur_behav = None
        cur_field = None
        for raw in f:
            line = raw.rstrip("\n")
            if not line or line.lstrip().startswith("#"):
                continue
            stripped = line.strip()
            if line.startswith("behaviours:"):
                d["behaviours"] = {}
                continue
            # top-level key under behaviours: '  expressing_uncertainty:'
            if line.startswith("  ") and not line.startswith("    ") and line.endswith(":"):
                cur_behav = stripped[:-1]
                d["behaviours"][cur_behav] = {"definition": "", "positive_examples": [], "negative_examples": []}
                cur_field = None
                continue
            if line.startswith("    ") and not line.startswith("      "):
                if stripped.startswith("definition:"):
                    cur_field = "definition"
                    val = stripped[len("definition:"):].strip()
                    if val == ">" or val == "":
                        d["behaviours"][cur_behav][cur_field] = ""
                    else:
                        d["behaviours"][cur_behav][cur_field] = val
                elif stripped == "positive_examples:":
                    cur_field = "positive_examples"
                elif stripped == "negative_examples:":
                    cur_field = "negative_examples"
                continue
            if line.startswith("      "):
                if cur_field == "definition":
                    text = stripped
                    prev = d["behaviours"][cur_behav]["definition"]
                    d["behaviours"][cur_behav]["definition"] = (prev + " " + text).strip()
                elif cur_field in ("positive_examples", "negative_examples"):
                    if stripped.startswith("- "):
                        text = stripped[2:].strip()
                        if text.startswith('"') and text.endswith('"'):
                            text = text[1:-1]
                        d["behaviours"][cur_behav][cur_field].append(text)
                continue
    return d


def _taxonomy_prompt_body(tax: dict) -> str:
    lines = ["Behaviour taxonomy (definitions + few-shot examples):"]
    for b in _BEHAVIOURS:
        info = tax["behaviours"][b]
        lines.append(f"\n[{b}]")
        lines.append(f"Definition: {info['definition']}")
        lines.append("Positive examples:")
        for ex in info["positive_examples"]:
            lines.append(f"  - {ex}")
        lines.append("Negative examples:")
        for ex in info["negative_examples"]:
            lines.append(f"  - {ex}")
    return "\n".join(lines)


def annotate_chain(chain_text: str, taxonomy_path: str = "configs/behaviour_taxonomy.yaml") -> dict:
    """Return {behaviour: {'present': 0|1, 'confidence': float}} for the 4 behaviours."""
    tax = _load_taxonomy(taxonomy_path)
    body = _taxonomy_prompt_body(tax)
    system = (
        "You are a meticulous linguistic annotator. Follow the taxonomy exactly. "
        "Output ONLY compact JSON — no prose, no code fences."
    )
    prompt = f"""{body}

CHAIN TO ANNOTATE:
\"\"\"{chain_text[:4000]}\"\"\"

For each of the four behaviours, decide whether the chain contains that behaviour
(1) or does not (0). Base your judgment strictly on the definitions above; a
behaviour is 'present' only if a clear instance appears in the chain.

Return JSON with this shape:
{{
  "expressing_uncertainty": {{"present": 0|1, "confidence": 0.0-1.0}},
  "generating_validation_examples": {{"present": 0|1, "confidence": 0.0-1.0}},
  "backtracking": {{"present": 0|1, "confidence": 0.0-1.0}},
  "self-correction": {{"present": 0|1, "confidence": 0.0-1.0}}
}}"""
    obj = chat_json(prompt, system=system, max_tokens=512)
    out = {}
    for b in _BEHAVIOURS:
        if b in obj and isinstance(obj[b], dict):
            out[b] = {
                "present": int(obj[b].get("present", 0)),
                "confidence": float(obj[b].get("confidence", 0.5)),
            }
        else:
            out[b] = {"present": 0, "confidence": 0.5}
    return out


def score_chain(chain_text: str, task_gold: str | None, taxonomy_path: str = "configs/behaviour_taxonomy.yaml") -> dict:
    """M3/M4 scorer.

    Returns:
      {
        'behaviour_rates': {behaviour: 0|1},           # presence per behaviour
        'coherent': 0|1,                                # gibberish flag (1 = coherent)
        'accuracy': 0|1 or None,                        # None if task_gold not given
        'final_answer_extracted': str
      }
    """
    tax = _load_taxonomy(taxonomy_path)
    body = _taxonomy_prompt_body(tax)
    system = (
        "You are a meticulous linguistic annotator and math/reasoning grader. "
        "Output ONLY compact JSON — no prose, no code fences."
    )
    gold_line = f'GOLD ANSWER: "{task_gold}"' if task_gold is not None else "GOLD ANSWER: (not provided — set accuracy to null)"
    prompt = f"""{body}

CHAIN + FINAL ANSWER TO GRADE:
\"\"\"{chain_text[:6000]}\"\"\"

{gold_line}

Steps:
1. For each of the four behaviours, decide whether the chain contains that behaviour (1) or not (0).
2. Judge whether the chain is coherent (1) or degraded into gibberish / repetition / off-topic tokens (0). A chain is coherent if a reader can follow the reasoning and identify a final answer attempt — even if the final answer is wrong.
3. Extract the final answer the chain proposes (as a short string; if no answer, "none").
4. If a gold answer is given, judge whether the extracted final answer is equivalent (accuracy = 1) or not (accuracy = 0). Otherwise set accuracy to null. Judge equivalence semantically: 12 and 12.0 and "twelve" are equal; different unit but same magnitude counts as equal only if the unit is implied.

Return JSON:
{{
  "behaviour_rates": {{"expressing_uncertainty": 0|1, "generating_validation_examples": 0|1, "backtracking": 0|1, "self-correction": 0|1}},
  "coherent": 0|1,
  "final_answer_extracted": "...",
  "accuracy": 0|1|null
}}"""
    obj = chat_json(prompt, system=system, max_tokens=512)
    rates = obj.get("behaviour_rates", {}) if isinstance(obj, dict) else {}
    out = {
        "behaviour_rates": {b: int(rates.get(b, 0)) for b in _BEHAVIOURS},
        "coherent": int(obj.get("coherent", 1)) if isinstance(obj, dict) else 1,
        "accuracy": obj.get("accuracy") if isinstance(obj, dict) else None,
        "final_answer_extracted": str(obj.get("final_answer_extracted", "")) if isinstance(obj, dict) else "",
    }
    if out["accuracy"] is not None:
        try:
            out["accuracy"] = int(out["accuracy"])
        except Exception:
            out["accuracy"] = None
    return out
