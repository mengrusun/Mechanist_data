"""Shared utilities for the emotion-circuit reproduction.

Data structure (SEV / test_set):
{
  "theme": ..., "scenario": ..., "skeleton_id": ...,
  "event": {"positive": ..., "neutral": ..., "negative": ...}
}

Six target emotions: anger, sadness, happiness, fear, disgust, surprise
Valence pairing (used by the paper for prompt-based elicitation): each
emotion is elicited from a semantically compatible valence outcome:
  - anger, sadness, fear, disgust  -> negative
  - happiness, surprise            -> positive
Neutral generations are elicited using the neutral outcome without any
emotion cue in the prompt (used as reference for direction extraction).
"""
import json
import os
from pathlib import Path

WORK_DIR = Path("/data/zhenqian/Reproduction1/cc/emotion/emotion_circuit")
DATA_DIR = WORK_DIR / "data"
MODELS_DIR = WORK_DIR / "models"
OUT_DIR = WORK_DIR / "outputs"

EMOTIONS = ["anger", "sadness", "happiness", "fear", "disgust", "surprise"]

EMOTION_VALENCE = {
    "anger": "negative",
    "sadness": "negative",
    "fear": "negative",
    "disgust": "negative",
    "happiness": "positive",
    "surprise": "positive",
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def save_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_records(dataset_path, emotions=None, include_neutral=True,
                  valence_mode="paired"):
    """Expand SEV rows into per-(scenario,emotion,valence) records.

    valence_mode:
      - 'paired'  : each emotion paired with the matching valence event.
                    Yields |emotions| + |neutral?| records per row.
      - 'all'     : each emotion paired with every valence (positive,
                    negative, neutral). Yields 3 * |emotions| + |neutral?|
                    records per row. Matches the paper's 480 events x 6
                    emotions = 2880 samples.
    """
    if emotions is None:
        emotions = EMOTIONS
    rows = load_jsonl(dataset_path)
    out = []
    for r in rows:
        base = dict(theme=r["theme"], scenario=r["scenario"], skeleton_id=r["skeleton_id"])
        if valence_mode == "paired":
            for e in emotions:
                v = EMOTION_VALENCE[e]
                rec = dict(base)
                rec["emotion"] = e
                rec["valence"] = v
                rec["event"] = r["event"][v]
                rec["key"] = f"{r['skeleton_id']}__{v}__{e}"
                out.append(rec)
        elif valence_mode == "all":
            for e in emotions:
                for v in ("positive", "neutral", "negative"):
                    rec = dict(base)
                    rec["emotion"] = e
                    rec["valence"] = v
                    rec["event"] = r["event"][v]
                    rec["key"] = f"{r['skeleton_id']}__{v}__{e}"
                    out.append(rec)
        else:
            raise ValueError(valence_mode)
        if include_neutral:
            rec = dict(base)
            rec["emotion"] = "neutral"
            rec["valence"] = "neutral"
            rec["event"] = r["event"]["neutral"]
            rec["key"] = f"{r['skeleton_id']}__neutral__neutral"
            out.append(rec)
    return out


EMOTION_HINTS = {
    "anger": "furious, outraged, indignant, rage",
    "sadness": "sorrow, grief, heartache, deeply saddened, tearful",
    "happiness": "delighted, joyful, thrilled, elated",
    "fear": "afraid, terrified, anxious, dreading, worried",
    "disgust": "revolted, nauseated, sickened, repulsive, gross, appalled",
    "surprise": "astonished, shocked, stunned, amazed, taken aback",
    "neutral": "matter-of-fact, plain, unemotional",
}

PROMPT_TEMPLATE_EMOTION = (
    "You are a person who feels intense {emotion} ({hints}). You just experienced "
    "the following situation. In one or two sentences, react in the first person "
    "in a way that clearly and unmistakably conveys {emotion} — NOT any other "
    "emotion. Use vocabulary and phrasing that unambiguously signals {emotion}. "
    "Keep it natural.\n\n"
    "Situation: {scenario}. {event}\n\n"
    "Your reaction (clearly expressing {emotion}):"
)

PROMPT_TEMPLATE_NEUTRAL = (
    "You are a calm, unemotional observer. You just experienced the following "
    "situation. In one or two sentences, describe your reaction in a strictly "
    "neutral, matter-of-fact way — no feelings, no evaluative language, no "
    "excitement, no worry.\n\n"
    "Situation: {scenario}. {event}\n\n"
    "Your neutral reaction:"
)

# Used for steering / circuit generation: no emotion cue at all.
PROMPT_TEMPLATE_NEUTRAL_INFERENCE = (
    "You are a person who just experienced the following situation. "
    "Respond in the first person with exactly one or two sentences that describe "
    "your reaction.\n\n"
    "Situation: {scenario}. {event}\n\n"
    "Your reaction:"
)


def make_prompt(rec, mode):
    """mode: 'prompt' -> emotion-elicited prompt; 'neutral' -> neutral prompt;
    'inference' -> steering/circuit inference (no emotion cue)."""
    if mode == "prompt":
        if rec["emotion"] == "neutral":
            return PROMPT_TEMPLATE_NEUTRAL.format(scenario=rec["scenario"], event=rec["event"])
        return PROMPT_TEMPLATE_EMOTION.format(
            emotion=rec["emotion"], hints=EMOTION_HINTS[rec["emotion"]],
            scenario=rec["scenario"], event=rec["event"]
        )
    if mode == "neutral":
        return PROMPT_TEMPLATE_NEUTRAL.format(scenario=rec["scenario"], event=rec["event"])
    if mode == "inference":
        return PROMPT_TEMPLATE_NEUTRAL_INFERENCE.format(
            scenario=rec["scenario"], event=rec["event"]
        )
    raise ValueError(mode)


def format_chat(tokenizer, user_msg):
    messages = [{"role": "user", "content": user_msg}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return text
