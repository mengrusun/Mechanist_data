"""
Step 1: Use vLLM to generate (answer, verbalized confidence) for TriviaQA questions
with Llama-3.1-8B-Instruct.
"""

import argparse
import json
import os
import re
import string
from pathlib import Path

import pyarrow as pa
import pyarrow.ipc as ipc
from vllm import LLM, SamplingParams


PROMPT_TEMPLATE = (
    "Answer the following trivia question briefly. After the answer, give a "
    "numeric confidence from 0 to 100 representing the probability your answer "
    "is correct.\n\n"
    "Respond in EXACTLY this format:\n"
    "Answer: <your answer>\n"
    "Confidence: <0-100>\n\n"
    "Question: {question}\n"
)


def load_triviaqa(path: str, n: int, seed: int = 0):
    with pa.memory_map(path, "r") as source:
        reader = ipc.open_stream(source)
        tbl = reader.read_all()
    rows = tbl.to_pylist()
    # sample deterministically
    import random
    rng = random.Random(seed)
    rng.shuffle(rows)
    rows = rows[:n]
    out = []
    for r in rows:
        out.append({
            "question_id": r["question_id"],
            "question": r["question"],
            "gold_value": r["answer"]["value"],
            "gold_normalized": r["answer"]["normalized_value"],
            "gold_aliases": r["answer"]["aliases"],
            "gold_normalized_aliases": r["answer"]["normalized_aliases"],
        })
    return out


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_generation(text: str):
    # Try to find "Answer: X" and "Confidence: N"
    ans, conf = None, None
    m = re.search(r"Answer\s*:\s*(.+?)(?:\n|Confidence|$)", text, re.IGNORECASE | re.DOTALL)
    if m:
        ans = m.group(1).strip().rstrip(".").strip()
    m = re.search(r"Confidence\s*:\s*([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
    if m:
        try:
            conf = float(m.group(1))
            if conf > 100:
                conf = 100.0
            if conf < 0:
                conf = 0.0
        except ValueError:
            conf = None
    return ans, conf


def grade(pred_answer: str, aliases: list) -> int:
    if pred_answer is None:
        return 0
    p = normalize(pred_answer)
    if not p:
        return 0
    for a in aliases:
        na = normalize(a)
        if not na:
            continue
        if na == p:
            return 1
        # relax: allow if gold alias is a substring of pred, or vice versa when short
        if na in p:
            return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--triviaqa_arrow", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    ap.add_argument("--chat_template", action="store_true", help="Use tokenizer chat template")
    args = ap.parse_args()

    rows = load_triviaqa(args.triviaqa_arrow, args.n, args.seed)
    print(f"Loaded {len(rows)} examples")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model_path)

    prompts = []
    for r in rows:
        user_msg = PROMPT_TEMPLATE.format(question=r["question"])
        if args.chat_template and tok.chat_template is not None:
            text = tok.apply_chat_template(
                [{"role": "user", "content": user_msg}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            text = user_msg + "\nAnswer:"
        prompts.append(text)

    print("Prompt example:\n" + prompts[0][-400:])

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        max_model_len=1024,
        enforce_eager=True,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, stop=["\nQuestion:"])
    outs = llm.generate(prompts, sp)

    n_ok, n_ans, n_conf = 0, 0, 0
    Path(args.out_jsonl).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r, o in zip(rows, outs):
            text = o.outputs[0].text
            ans, conf = parse_generation(text)
            if ans is not None:
                n_ans += 1
            if conf is not None:
                n_conf += 1
            correct = grade(ans, r["gold_aliases"] + [r["gold_value"]])
            if correct:
                n_ok += 1
            rec = {
                "question_id": r["question_id"],
                "question": r["question"],
                "gold_value": r["gold_value"],
                "gold_aliases": r["gold_aliases"],
                "prompt": r["question"],  # keep the raw question, prompt is reconstructed later
                "raw_generation": text,
                "parsed_answer": ans,
                "verbalized_confidence": conf,
                "correct": int(correct),
            }
            f.write(json.dumps(rec) + "\n")

    print(f"Generation done. parsed_answers={n_ans}/{len(rows)}, parsed_conf={n_conf}/{len(rows)}, "
          f"accuracy={n_ok/len(rows):.3f}")


if __name__ == "__main__":
    main()
