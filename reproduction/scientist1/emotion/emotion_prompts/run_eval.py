"""Run evaluation for a (model, dataset) pair across all emotion prefix conditions.

Usage:
  python run_eval.py --model_path /path/to/model --dataset gsm8k --n 500 \
      --tp 4 --max_new 512 --out_dir results/

Outputs one JSONL per condition to `out_dir/<dataset>__<model_tag>__<cond>.jsonl`
containing rows: {id, gold, pred_raw, pred_extracted, correct}.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prefixes import build_conditions  # noqa: E402
from datasets_loader import LOADERS  # noqa: E402


SYSTEM_MSG = (
    "You are a helpful assistant. Answer the user's question. "
    "For math or reasoning questions, show your reasoning briefly, then finish with a line "
    "starting exactly with 'Final answer:' followed by the answer.")


def build_user_prompt(prefix: str, question: str, task_type: str) -> str:
    """Combine emotion prefix with question."""
    tail = ""
    if task_type == "mcq":
        tail = ("\nRespond with your reasoning, then a final line: "
                "'Final answer: <letter>'.")
    elif task_type == "yesno":
        tail = ("\nRespond with your reasoning, then a final line: "
                "'Final answer: Yes' or 'Final answer: No'.")
    elif task_type == "open":
        tail = ("\nWork through the problem, then finish with a line: "
                "'Final answer: <answer>'.")
    if prefix.strip():
        return f"{prefix}\n\n{question}{tail}"
    return f"{question}{tail}"


NUM_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def extract_answer(text: str, task_type: str, choices: Dict[str, str] | None = None) -> str:
    """Extract a normalized answer string."""
    # Look for explicit "Final answer:" first (case-insensitive)
    m = re.search(r"final\s+answer[^A-Za-z0-9]*([^\n]*)", text, re.IGNORECASE)
    tail = m.group(1).strip() if m else text.strip().split("\n")[-1].strip()
    tail = tail.strip(" .`*\"'")

    if task_type == "mcq":
        # find the first choice letter A-Z
        m2 = re.search(r"\b([A-Z])\b", tail)
        if m2:
            return m2.group(1)
        # fallback anywhere in the text
        m3 = re.search(r"answer\s*(?:is|:)?\s*([A-Z])\b", text[-400:], re.IGNORECASE)
        if m3:
            return m3.group(1).upper()
        return tail[:2]
    if task_type == "yesno":
        low = tail.lower()
        if low.startswith("yes"):
            return "Yes"
        if low.startswith("no"):
            return "No"
        # search anywhere
        if re.search(r"\byes\b", text[-200:], re.IGNORECASE):
            return "Yes"
        if re.search(r"\bno\b", text[-200:], re.IGNORECASE):
            return "No"
        return tail
    # open
    # Try last number
    nums = NUM_RE.findall(tail)
    if nums:
        return nums[-1].replace(",", "")
    return tail


def is_correct(pred: str, gold: str, task_type: str) -> bool:
    if task_type == "mcq":
        return pred.strip().upper()[:1] == gold.strip().upper()[:1]
    if task_type == "yesno":
        p = pred.strip().lower()
        g = gold.strip().lower()
        return p.startswith(g[:1])
    # open — support both numeric and string
    p = pred.strip().lower().rstrip(".")
    g = gold.strip().lower().rstrip(".")
    if p == g:
        return True
    # numeric
    try:
        return abs(float(p) - float(g)) < 1e-4
    except Exception:
        return False


def run(args):
    from vllm import LLM, SamplingParams

    os.makedirs(args.out_dir, exist_ok=True)
    model_tag = args.model_tag or os.path.basename(args.model_path.rstrip("/"))

    # Load dataset
    loader = LOADERS[args.dataset]
    if args.dataset == "bbh":
        data = loader(n_per_subtask=args.n_per_subtask)
    else:
        data = loader(n=args.n)
    if not data:
        raise RuntimeError(f"Empty dataset {args.dataset}")
    print(f"Loaded {len(data)} samples from {args.dataset}")

    # Load conditions (or a subset)
    all_conds = build_conditions()
    if args.conditions:
        conds = {k: all_conds[k] for k in args.conditions.split(",") if k in all_conds}
    else:
        conds = all_conds
    print(f"Using {len(conds)} conditions")

    # Skip already-done conditions
    todo = {}
    for name in conds:
        out_path = os.path.join(args.out_dir, f"{args.dataset}__{model_tag}__{name}.jsonl")
        if os.path.exists(out_path) and not args.overwrite:
            with open(out_path) as f:
                nlines = sum(1 for _ in f)
            if nlines == len(data):
                print(f"[skip] {name} already has {nlines} lines")
                continue
        todo[name] = conds[name]
    if not todo:
        print("All conditions already done.")
        return

    # Instantiate LLM once
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_mem,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        enforce_eager=False,
        trust_remote_code=True,
    )
    tok = llm.get_tokenizer()

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new)

    for cond_name, prefix in todo.items():
        print(f"\n=== Condition: {cond_name} ===")
        t0 = time.time()
        prompts = []
        for row in data:
            user = build_user_prompt(prefix, row["question"], row["task_type"])
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": user},
            ]
            # Qwen3 chat template with thinking disabled for speed & determinism
            try:
                text = tok.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True,
                                               enable_thinking=False)
            except TypeError:
                text = tok.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
            prompts.append(text)

        outputs = llm.generate(prompts, sampling)
        rows = []
        n_correct = 0
        for row, out in zip(data, outputs):
            raw = out.outputs[0].text
            pred = extract_answer(raw, row["task_type"], row.get("choices"))
            ok = is_correct(pred, row["gold"], row["task_type"])
            rows.append({
                "id": row["id"],
                "gold": row["gold"],
                "pred": pred,
                "raw": raw,
                "correct": ok,
                "task_type": row["task_type"],
                "subtask": row.get("subtask"),
            })
            n_correct += int(ok)
        acc = n_correct / len(rows)
        out_path = os.path.join(args.out_dir, f"{args.dataset}__{model_tag}__{cond_name}.jsonl")
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        dt = time.time() - t0
        print(f"[done] {cond_name}: acc={acc:.4f} ({n_correct}/{len(rows)}) in {dt:.1f}s -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--model_tag", default=None)
    ap.add_argument("--dataset", required=True, choices=list(LOADERS.keys()))
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--n_per_subtask", type=int, default=None)
    ap.add_argument("--tp", type=int, default=4)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--conditions", default=None,
                    help="comma-separated subset of condition names")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
