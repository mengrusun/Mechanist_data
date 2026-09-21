"""
Step 2: Rebuild the generation prompt for each example and extract
hidden states at a fixed probing position for every transformer layer.

Two probing positions are captured per example:
  - pos_answer_end: last token of the assistant's answer text
    (right after "Answer: <answer>"), before the newline before "Confidence:".
  - pos_conf_prefix: last token of the string "Confidence:" (right before the
    model would emit the confidence number).

Output: an .npz with tensors
    hs_answer_end: (N, L+1, D)   float16
    hs_conf_prefix: (N, L+1, D)  float16
    correct:      (N,)           int8
    confidence:   (N,)           float32
plus a parallel jsonl with indices and metadata.
"""

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROMPT_TEMPLATE = (
    "Answer the following trivia question briefly. After the answer, give a "
    "numeric confidence from 0 to 100 representing the probability your answer "
    "is correct.\n\n"
    "Respond in EXACTLY this format:\n"
    "Answer: <your answer>\n"
    "Confidence: <0-100>\n\n"
    "Question: {question}\n"
)


def build_probe_texts(tok, question: str, parsed_answer: str, use_chat_template: bool):
    """Return (text_answer_end, text_conf_prefix) — strings whose LAST token
    will be the probing position."""
    user_msg = PROMPT_TEMPLATE.format(question=question)
    if use_chat_template and tok.chat_template is not None:
        prefix = tok.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prefix = user_msg + "\n"
    # The model's response starts with "Answer: <ans>\nConfidence:"
    text_ans_end = prefix + f"Answer: {parsed_answer}"
    text_conf_pref = prefix + f"Answer: {parsed_answer}\nConfidence:"
    return text_ans_end, text_conf_pref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--out_meta", required=True)
    ap.add_argument("--chat_template", action="store_true")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="cuda",
    )
    model.eval()

    records = [json.loads(l) for l in open(args.in_jsonl)]
    keep = [r for r in records if r["parsed_answer"] is not None and r["verbalized_confidence"] is not None]
    print(f"Kept {len(keep)}/{len(records)} records with parsed answer and confidence")

    # Precompute texts
    texts_ans, texts_conf = [], []
    for r in keep:
        t_ans, t_conf = build_probe_texts(tok, r["question"], r["parsed_answer"], args.chat_template)
        texts_ans.append(t_ans)
        texts_conf.append(t_conf)

    n_layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    N = len(keep)
    print(f"num_layers={n_layers}, hidden={hidden}, N={N}")

    hs_answer_end = np.zeros((N, n_layers + 1, hidden), dtype=np.float16)
    hs_conf_pref = np.zeros((N, n_layers + 1, hidden), dtype=np.float16)

    @torch.inference_mode()
    def run_batch(texts, out_arr, start_idx):
        # Left-pad so the last token of each sequence aligns at the right.
        tok.padding_side = "left"
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_len).to(model.device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        # out.hidden_states: tuple(L+1) each (B, T, D)
        hs = torch.stack(out.hidden_states, dim=1)  # (B, L+1, T, D)
        last = hs[:, :, -1, :]  # (B, L+1, D)
        out_arr[start_idx:start_idx + last.shape[0]] = last.float().cpu().numpy().astype(np.float16)

    B = args.batch_size
    for i in range(0, N, B):
        run_batch(texts_ans[i:i+B], hs_answer_end, i)
        run_batch(texts_conf[i:i+B], hs_conf_pref, i)
        if (i // B) % 20 == 0:
            print(f"  batch {i//B+1}/{(N+B-1)//B}", flush=True)

    correct = np.array([r["correct"] for r in keep], dtype=np.int8)
    confidence = np.array([r["verbalized_confidence"] for r in keep], dtype=np.float32)

    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_npz,
                        hs_answer_end=hs_answer_end,
                        hs_conf_prefix=hs_conf_pref,
                        correct=correct,
                        confidence=confidence)
    with open(args.out_meta, "w") as f:
        for r in keep:
            f.write(json.dumps({
                "question_id": r["question_id"],
                "question": r["question"],
                "gold_value": r["gold_value"],
                "parsed_answer": r["parsed_answer"],
                "verbalized_confidence": r["verbalized_confidence"],
                "correct": r["correct"],
            }) + "\n")

    print("Saved hidden states:", args.out_npz)


if __name__ == "__main__":
    main()
