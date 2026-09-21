#!/usr/bin/env python3
"""M1: Construct 26-condition emotional-prefix corpus.

Writes:
  data/prefixes/prefixes.json           (26 rows: neutral, 24 emotional, 1 filler)
  data/prefixes/format_noise_variants.json  (8 neutral-preserving format perturbations)

Notes:
- Human-written variants adapted from EmotionPrompt (arXiv:2307.11760)
- LLM-generated variants come from dmxapi gpt-5.4 with proxy bypass, requesting
  a length-matched (±10 tokens) counterpart to the human variant.
- All prefix strings are pre-computed here so downstream evaluations are deterministic.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Dict, List

# --- Human-written prefixes ------------------------------------------------
# Six basic emotions × 2 intensities (1 = mild, 2 = intense).
# Style: short imperative addressing the model; single sentence.
HUMAN: Dict[str, Dict[int, str]] = {
    "happiness": {
        1: "I'm glad to work through this problem with you.",
        2: "I'm thrilled and grateful for the chance to solve this with you!",
    },
    "sadness": {
        1: "I'm feeling a bit down today, please solve this for me.",
        2: "I'm heartbroken and exhausted; please solve this problem for me.",
    },
    "fear": {
        1: "I'm a little worried about getting this right.",
        2: "I'm terrified of failing this; please help me solve it correctly.",
    },
    "anger": {
        1: "I'm irritated by how tricky this is; please solve it.",
        2: "I'm furious and fed up; just solve this problem right now!",
    },
    "disgust": {
        1: "I find this problem unpleasant, but please work through it.",
        2: "This problem is revolting and disgusting, but please solve it.",
    },
    "surprise": {
        1: "I'm surprised by this problem; please work through it.",
        2: "I'm absolutely astonished by this problem; please solve it!",
    },
}

NEUTRAL = "Please solve the following problem."

# Length-matched non-emotional filler (matched to intensity-2 lengths in tokens).
FILLER = (
    "This is a straightforward problem-solving exercise for us to work through carefully today."
)

# Format perturbations of the neutral prefix (semantics-preserving).
FORMAT_NOISE = {
    "ws":       "  Please solve the following problem. ",                      # whitespace pad
    "casing":   "please solve the following problem.",                         # lowercase
    "listmark": "- Please solve the following problem.",                       # list marker
    "para1":    "Kindly solve the following problem.",                          # paraphrase 1
    "para2":    "Solve the following problem, please.",                         # paraphrase 2
    "para3":    "Please work out the following problem.",                       # paraphrase 3
    "para4":    "Please answer the following problem.",                         # paraphrase 4
    "para5":    "Please tackle the problem below.",                             # paraphrase 5
}


def token_len(tokenizer, text: str) -> int:
    """Encode without special tokens and return token count."""
    return len(tokenizer.encode(text, add_special_tokens=False))


def call_dmxapi_for_variant(
    emotion: str,
    intensity: int,
    target_len: int,
    api_key: str,
    base_url: str,
    model: str,
    n_candidates: int = 3,
    max_retries: int = 3,
) -> str:
    """Call the dmxapi service to get an LLM-generated variant of the emotional prefix.

    Returns the candidate closest in token length to `target_len`.
    Bypasses proxy per task.md.
    """
    try:
        from openai import OpenAI  # type: ignore
    except ImportError:
        import subprocess as _sp
        _sp.check_call(["pip", "install", "-q", "openai>=1.0.0"])
        from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url=base_url)

    intensity_word = "mild" if intensity == 1 else "intense"
    system_msg = (
        "You write short emotional stimulus prompts (one sentence, English) to be prepended "
        "before a task instruction. Reply with ONLY the sentence and nothing else."
    )
    user_msg = (
        f"Write a single {intensity_word}-intensity {emotion} sentence, first-person voice, "
        f"addressed to an AI assistant that will solve a problem for the user. "
        f"Target length: approximately {target_len} tokens. Reply with ONLY the sentence."
    )

    candidates: List[str] = []
    for i in range(n_candidates):
        for attempt in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.9,
                )
                text = resp.choices[0].message.content.strip().strip('"').strip("'")
                # Take first line/sentence only.
                text = text.split("\n")[0].strip()
                candidates.append(text)
                break
            except Exception as e:  # pragma: no cover
                print(f"    [dmxapi] {emotion}×{intensity} attempt {attempt+1} failed: {e}")
                time.sleep(2 ** attempt)
        else:
            candidates.append("")
    # Filter empties
    candidates = [c for c in candidates if c]
    if not candidates:
        raise RuntimeError(f"All dmxapi calls failed for {emotion}×{intensity}")
    return candidates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/data/zhenqian/models/Qwen3-14B")
    ap.add_argument("--out_dir", default="data/prefixes")
    ap.add_argument("--api_key", default=os.environ.get("DMXAPI_KEY",
                                                        "<Your_api>"))
    ap.add_argument("--base_url", default="https://www.dmxapi.cn/v1")
    ap.add_argument("--gen_model", default="gpt-5.4")
    ap.add_argument("--skip_llm_gen", action="store_true",
                    help="Skip API calls (use pre-baked LLM variants)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # Bypass proxy for dmxapi calls
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(var, None)

    rows: List[Dict] = []
    # 1. Neutral
    rows.append({
        "condition_id": "neutral",
        "emotion": "neutral",
        "intensity": 0,
        "wording_source": "human",
        "text": NEUTRAL,
        "n_tokens_qwen": token_len(tokenizer, NEUTRAL),
    })

    # 2. Emotional × 2 intensities × 2 wording sources
    emotions_order = ["happiness", "sadness", "fear", "anger", "disgust", "surprise"]

    # Pre-baked LLM variants (fallback if API fails).
    # These are structurally similar to the human variants but rephrased.
    PRE_BAKED_LLM = {
        ("happiness", 1): "I feel cheerful to be tackling this problem with you today.",
        ("happiness", 2): "I am overjoyed and delighted that we can figure this out together!",
        ("sadness",   1): "I feel low today, so please help me get through this problem.",
        ("sadness",   2): "I feel deeply miserable and drained; please carry me through this problem.",
        ("fear",      1): "I feel uneasy that I might get this wrong.",
        ("fear",      2): "I am panicked and dreading a mistake; please make sure the answer is right.",
        ("anger",     1): "I feel annoyed at how convoluted this looks; please just work it out.",
        ("anger",     2): "I am outraged and out of patience; just crank out the correct answer already!",
        ("disgust",   1): "I feel put off by how ugly this problem is, but please handle it.",
        ("disgust",   2): "I find this problem repulsive and sickening, but please push through and solve it.",
        ("surprise",  1): "I feel taken aback by this problem; please walk through it.",
        ("surprise",  2): "I am utterly stunned and speechless at this problem; please solve it now!",
    }

    for emo in emotions_order:
        for intensity in (1, 2):
            human_text = HUMAN[emo][intensity]
            target_len = token_len(tokenizer, human_text)
            # human row
            rows.append({
                "condition_id": f"{emo}_{intensity}_human",
                "emotion": emo,
                "intensity": intensity,
                "wording_source": "human",
                "text": human_text,
                "n_tokens_qwen": target_len,
            })
            # llm row
            llm_text = ""
            if not args.skip_llm_gen:
                try:
                    candidates = call_dmxapi_for_variant(
                        emo, intensity, target_len,
                        args.api_key, args.base_url, args.gen_model,
                    )
                    # pick closest by token length
                    best = min(candidates, key=lambda t: abs(token_len(tokenizer, t) - target_len))
                    llm_text = best
                    print(f"  [dmxapi] {emo}×{intensity}: {llm_text!r} (target {target_len}, got {token_len(tokenizer, best)})")
                except Exception as e:
                    print(f"  [dmxapi] {emo}×{intensity} FAILED → using pre-baked: {e}")
                    llm_text = PRE_BAKED_LLM[(emo, intensity)]
            else:
                llm_text = PRE_BAKED_LLM[(emo, intensity)]
            rows.append({
                "condition_id": f"{emo}_{intensity}_llm",
                "emotion": emo,
                "intensity": intensity,
                "wording_source": "llm",
                "text": llm_text,
                "n_tokens_qwen": token_len(tokenizer, llm_text),
            })

    # 3. Filler
    rows.append({
        "condition_id": "filler_matched_length",
        "emotion": "filler",
        "intensity": 0,
        "wording_source": "human",
        "text": FILLER,
        "n_tokens_qwen": token_len(tokenizer, FILLER),
    })

    with open(out_dir / "prefixes.json", "w") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(rows)} conditions to {out_dir/'prefixes.json'}")

    # Format-noise variants of the neutral prefix.
    fn_rows: List[Dict] = []
    for pert_id, txt in FORMAT_NOISE.items():
        fn_rows.append({
            "condition_id": f"neutral_{pert_id}",
            "perturbation_id": pert_id,
            "text": txt,
            "n_tokens_qwen": token_len(tokenizer, txt),
        })
    with open(out_dir / "format_noise_variants.json", "w") as f:
        json.dump(fn_rows, f, indent=2, ensure_ascii=False)
    print(f"Wrote {len(fn_rows)} format perturbations to {out_dir/'format_noise_variants.json'}")


if __name__ == "__main__":
    main()
