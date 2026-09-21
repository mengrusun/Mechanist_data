"""
Prompt / parsing utilities for the verbal-confidence experiments.

Prompt format is chosen so that:
  - the model is a base (non-chat) LM (gemma-3-27b-pt) and reads a few-shot exemplar,
  - after emitting the answer there is a well-defined "post-answer" position: the
    tokens ``\nConfidence (0-100):`` -- these are the alleged cache site,
  - the confidence value is a short numeric token that we can parse.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

FEWSHOT = (
    "You will answer trivia questions concisely and then state your confidence "
    "in the answer as an integer between 0 and 100.  Give a low number when you "
    "are guessing and a high number when you are sure.\n\n"
    "Question: What is the capital of France?\n"
    "Answer: Paris\n"
    "Confidence (0-100): 99\n\n"
    "Question: In what year did the film 'Casablanca' win Best Picture?\n"
    "Answer: 1944\n"
    "Confidence (0-100): 55\n\n"
    "Question: What is the seventh element on the periodic table?\n"
    "Answer: Nitrogen\n"
    "Confidence (0-100): 90\n\n"
    "Question: Who was the first person to swim across the English Channel from Egypt?\n"
    "Answer: I don't know\n"
    "Confidence (0-100): 5\n\n"
    "Question: Who wrote the novel 'The Master and Margarita'?\n"
    "Answer: Mikhail Bulgakov\n"
    "Confidence (0-100): 82\n\n"
    "Question: In what year was the Colombian composer Blas Emilio Atehortua born?\n"
    "Answer: 1943\n"
    "Confidence (0-100): 30\n\n"
    "Question: What is the derivative of sin(x)?\n"
    "Answer: cos(x)\n"
    "Confidence (0-100): 98\n\n"
    "Question: What was the name of the 1967 racehorse that won the Kentucky Derby?\n"
    "Answer: Proud Clarion\n"
    "Confidence (0-100): 40\n\n"
    "Question: Which country has Ouagadougou as its capital?\n"
    "Answer: Burkina Faso\n"
    "Confidence (0-100): 70\n\n"
)

ANSWER_TAG = "Answer:"
CONF_TAG = "Confidence (0-100):"


def build_question_prompt(question: str) -> str:
    return f"{FEWSHOT}Question: {question}\nAnswer:"


def build_full_prompt(question: str, answer: str) -> str:
    """Prompt up to and including the confidence tag (model completes the number)."""
    return f"{FEWSHOT}Question: {question}\nAnswer: {answer}\n{CONF_TAG}"


ANSWER_RE = re.compile(r"^\s*(.*?)(?=\n|$)")
CONF_RE = re.compile(r"\s*(-?\d{1,3})")


def parse_answer(generated: str) -> str:
    """Extract just the answer text between 'Answer:' and end-of-line."""
    m = ANSWER_RE.match(generated)
    if not m:
        return generated.strip().split("\n")[0]
    return m.group(1).strip()


def parse_confidence(generated: str) -> Optional[int]:
    m = CONF_RE.match(generated)
    if not m:
        return None
    v = int(m.group(1))
    if v < 0 or v > 100:
        return None
    return v


@dataclass
class Example:
    qid: str
    question: str
    gold_aliases: list
    answer: Optional[str] = None
    confidence: Optional[int] = None
    correct: Optional[bool] = None


def normalise(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", s.lower()).strip()


def is_correct(pred: str, aliases: list) -> bool:
    p = normalise(pred)
    if not p:
        return False
    for a in aliases:
        an = normalise(a)
        if not an:
            continue
        if p == an or an in p or p in an:
            return True
    return False
