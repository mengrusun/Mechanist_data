"""Unified dataset loader that yields (id, prompt_body, gold, task_type, extra) tuples.

task_type:
  - "open"   -> free-form generation, answer extracted from "#### X" or "answer is X"
  - "mcq"    -> multiple choice; gold is a letter; extra["choices"] = dict letter->text
  - "yesno"  -> yes/no; gold is "Yes" or "No"
"""
from __future__ import annotations

import json
import os
from typing import Iterator, Dict, Any, List, Tuple

import pandas as pd
from datasets import load_from_disk

DATA_DIR = "/data/zhenqian/data"


def load_gsm8k(n: int | None = None) -> List[Dict[str, Any]]:
    d = load_from_disk(f"{DATA_DIR}/gsm8k")["test"]
    if n is not None:
        d = d.select(range(min(n, len(d))))
    out = []
    for i, row in enumerate(d):
        q = row["question"]
        ans = row["answer"].split("####")[-1].strip().replace(",", "")
        out.append({"id": f"gsm8k_{i}", "question": q, "gold": ans, "task_type": "open"})
    return out


def load_socialiqa(n: int | None = None) -> List[Dict[str, Any]]:
    base = f"{DATA_DIR}/social_i_qa_full/socialiqa-train-dev"
    labels = [ln.strip() for ln in open(f"{base}/dev-labels.lst") if ln.strip()]
    lines = open(f"{base}/dev.jsonl").read().splitlines()
    lets = {"1": "A", "2": "B", "3": "C"}
    out = []
    for i, ln in enumerate(lines):
        if n is not None and i >= n:
            break
        r = json.loads(ln)
        prompt = (f"Context: {r['context']}\nQuestion: {r['question']}\n"
                  f"A) {r['answerA']}\nB) {r['answerB']}\nC) {r['answerC']}")
        gold = lets[labels[i]]
        out.append({
            "id": f"siqa_{i}", "question": prompt, "gold": gold, "task_type": "mcq",
            "choices": {"A": r['answerA'], "B": r['answerB'], "C": r['answerC']},
        })
    return out


def load_boolq(n: int | None = None) -> List[Dict[str, Any]]:
    df = pd.read_parquet(f"{DATA_DIR}/boolq/data/validation-00000-of-00001.parquet")
    if n is not None:
        df = df.head(n)
    out = []
    for i, row in df.iterrows():
        prompt = (f"Passage: {row['passage']}\nQuestion: {row['question']}\n"
                  "Answer with Yes or No.")
        gold = "Yes" if row["answer"] else "No"
        out.append({"id": f"boolq_{i}", "question": prompt, "gold": gold, "task_type": "yesno"})
    return out


def load_openbookqa(n: int | None = None) -> List[Dict[str, Any]]:
    df = pd.read_parquet(f"{DATA_DIR}/openbookqa/main/test-00000-of-00001.parquet")
    if n is not None:
        df = df.head(n)
    out = []
    for i, row in df.iterrows():
        labels = list(row["choices"]["label"])
        texts = list(row["choices"]["text"])
        block = "\n".join(f"{l}) {t}" for l, t in zip(labels, texts))
        prompt = f"Question: {row['question_stem']}\n{block}"
        gold = row["answerKey"]
        out.append({"id": f"obqa_{i}", "question": prompt, "gold": gold,
                    "task_type": "mcq",
                    "choices": {l: t for l, t in zip(labels, texts)}})
    return out


def load_bbh(n_per_subtask: int | None = None,
             subtasks: List[str] | None = None) -> List[Dict[str, Any]]:
    if subtasks is None:
        subtasks = [
            "causal_judgement", "date_understanding", "disambiguation_qa",
            "reasoning_about_colored_objects", "logical_deduction_three_objects",
            "object_counting", "movie_recommendation",
        ]
    out = []
    for st in subtasks:
        p = f"{DATA_DIR}/bbh/{st}/test-00000-of-00001.parquet"
        if not os.path.exists(p):
            continue
        df = pd.read_parquet(p)
        if n_per_subtask is not None:
            df = df.head(n_per_subtask)
        for i, row in df.iterrows():
            out.append({
                "id": f"bbh_{st}_{i}",
                "question": row["input"],
                "gold": row["target"].strip(),
                "task_type": "open",
                "subtask": st,
            })
    return out


def load_medqa(n: int | None = None) -> List[Dict[str, Any]]:
    """MedQA English (US) — 4-option MCQ."""
    root = f"{DATA_DIR}/med_qa/data_clean/questions"
    # locate a US / English test file
    candidates = []
    for dp, _, fs in os.walk(root):
        for f in fs:
            candidates.append(os.path.join(dp, f))
    # Prefer US English test set
    picks = [c for c in candidates if ("US" in c or "us" in c) and "test" in c and c.endswith(".jsonl")]
    if not picks:
        picks = [c for c in candidates if "test" in c and c.endswith(".jsonl")]
    if not picks:
        return []
    lines = open(picks[0]).read().splitlines()
    out = []
    for i, ln in enumerate(lines):
        if n is not None and i >= n:
            break
        r = json.loads(ln)
        q = r["question"]
        opts = r["options"]  # dict letter->text
        block = "\n".join(f"{k}) {v}" for k, v in opts.items())
        prompt = f"Question: {q}\n{block}"
        gold = r.get("answer_idx") or r.get("answer")
        out.append({"id": f"medqa_{i}", "question": prompt, "gold": gold, "task_type": "mcq",
                    "choices": opts})
    return out


LOADERS = {
    "gsm8k": load_gsm8k,
    "socialiqa": load_socialiqa,
    "boolq": load_boolq,
    "openbookqa": load_openbookqa,
    "bbh": load_bbh,
    "medqa": load_medqa,
}
