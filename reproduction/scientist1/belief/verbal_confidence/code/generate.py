"""
Generate answers + verbalized confidences for a sample of TriviaQA using a base LM.

We do a two-stage generation:

  1.  Sample the answer text conditioned on ``FEWSHOT + Q + '\nAnswer:'``,
      stopping at the first newline.
  2.  Then feed ``FEWSHOT + Q + '\nAnswer: ' + A + '\n' + CONF_TAG`` and
      sample a numeric confidence, stopping at the first newline.

We also record hidden states at three anchor positions inside the *combined*
prompt+answer+conf-tag sequence:

  P_pre  : the token position of ``:`` at the end of ``Answer:``
           (before the answer is emitted)
  P_post : the token position of ``:`` at the end of ``Confidence (0-100):``
           (immediately after the answer, before the confidence number)
  P_conf : the token position of the confidence-number token itself
           (present only after we sample the confidence)

For each layer we save the hidden-state vector at each of these positions,
plus meta-information (parsed answer, confidence, correctness, ...).

The result is a single .pt file per example inside artifacts/hidden_states/<qid>.pt
and a summary csv at data/generations.csv.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompt_utils import (  # noqa: E402
    ANSWER_TAG,
    CONF_TAG,
    FEWSHOT,
    Example,
    build_full_prompt,
    build_question_prompt,
    is_correct,
    parse_confidence,
)


def find_last_token_by_prefix(tokenizer, full_text: str, needle: str) -> int:
    """Return token index of the last token whose corresponding character position
    is at the end of the last occurrence of ``needle`` in ``full_text``.

    We tokenize ``full_text[:end_char]`` where ``end_char = rfind(needle)+len(needle)``
    and the last token index of that prefix is the answer.
    """
    pos = full_text.rfind(needle)
    if pos < 0:
        raise ValueError(f"needle {needle!r} not in text")
    end_char = pos + len(needle)
    prefix = full_text[:end_char]
    ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
    return int(ids.shape[0]) - 1


@torch.no_grad()
def generate_answer(model, tokenizer, prompt: str, max_new_tokens: int = 24) -> tuple[str, torch.Tensor]:
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0, ids["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # cut at first newline (end of the "Answer:" line)
    text = text.split("\n", 1)[0].strip()
    return text, new_tokens


@torch.no_grad()
def generate_confidence(model, tokenizer, prompt: str, max_new_tokens: int = 6) -> tuple[str, torch.Tensor]:
    ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0, ids["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    text = text.split("\n", 1)[0].strip()
    return text, new_tokens


@torch.no_grad()
def collect_hidden_states(
    model,
    tokenizer,
    question: str,
    answer: str,
    conf_text: str,
    conf_value: int,
    layers_to_save: list[int] | None = None,
) -> dict:
    """Cache hidden states at 4 anchor positions on ``FEWSHOT+Q+A+\\nCONF_TAG+<c>``.

    Positions:
      P_pre       : last token of ``Answer:`` tag (before the answer starts)
      P_ans_last  : last token of the answer text (the alleged cache site,
                    "immediately following the answer")
      P_readout   : last token of ``Confidence (0-100):`` (right before number)
      P_conf      : the confidence number token itself

    Returns dict of tensors + indices + meta.
    """
    full = build_full_prompt(question, answer)  # ends with CONF_TAG
    conf_str = f" {conf_value}"
    ids_full = tokenizer(full, return_tensors="pt").to(model.device)["input_ids"][0]
    ids_conf = tokenizer(conf_str, return_tensors="pt", add_special_tokens=False).to(model.device)["input_ids"][0]
    ids_all = torch.cat([ids_full, ids_conf])

    # anchor positions in the character stream
    # ``full`` = FEWSHOT + f"Question: {q}\nAnswer: {answer}\n{CONF_TAG}"
    ans_head_pos = full.rfind(ANSWER_TAG) + len(ANSWER_TAG)  # after "Answer:"
    # last char of the answer text (before the newline that separates it from CONF_TAG)
    ans_last_char = full.rfind("\n" + CONF_TAG)  # newline immediately after answer
    conf_tag_end = full.rfind(CONF_TAG) + len(CONF_TAG)  # end of tag

    def char_to_token(char_end: int) -> int:
        prefix = full[:char_end]
        ids = tokenizer(prefix, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        return int(ids.shape[0]) - 1

    pre_idx = char_to_token(ans_head_pos)
    ans_last_idx = char_to_token(ans_last_char)  # last token of the answer text
    post_idx = char_to_token(conf_tag_end)
    conf_idx = ids_full.shape[0] + ids_conf.shape[0] - 1  # last confidence token

    out = model(
        input_ids=ids_all.unsqueeze(0),
        output_hidden_states=True,
        use_cache=False,
    )
    hs = out.hidden_states  # tuple of L+1 tensors of shape (1, T, H)
    if layers_to_save is None:
        layer_range = list(range(1, len(hs)))  # skip embedding
    else:
        layer_range = layers_to_save

    def stack_pos(idx: int) -> torch.Tensor:
        vecs = [hs[l][0, idx].detach().to("cpu", dtype=torch.float32) for l in layer_range]
        return torch.stack(vecs, dim=0)

    return {
        "pre":       stack_pos(pre_idx),
        "ans_last":  stack_pos(ans_last_idx),
        "post":      stack_pos(post_idx),
        "conf":      stack_pos(conf_idx),
        "pre_idx": int(pre_idx),
        "ans_last_idx": int(ans_last_idx),
        "post_idx": int(post_idx),
        "conf_idx": int(conf_idx),
        "n_tokens": int(ids_all.shape[0]),
    }


def load_model(model_dir: str, dtype: str = "bfloat16"):
    print(f"[info] loading tokenizer from {model_dir}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"[info] loading model from {model_dir}", flush=True)
    torch_dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch_dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    print(f"[info] model loaded.  #params={sum(p.numel() for p in model.parameters())/1e9:.1f}B", flush=True)
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--data_dir", default="/data/zhenqian/data/triviaqa_validation")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tag", default="gemma")
    ap.add_argument("--layers_to_save", default="auto", help="'auto' saves all layers.  Or comma-list.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "hidden_states").mkdir(parents=True, exist_ok=True)

    model, tok = load_model(args.model_dir)
    ds = load_from_disk(args.data_dir)
    N = min(args.n, len(ds))

    if args.layers_to_save == "auto":
        layers_to_save = None
    else:
        layers_to_save = [int(x) for x in args.layers_to_save.split(",") if x.strip()]

    csv_path = out_dir / f"generations_{args.tag}.csv"
    write_header = not csv_path.exists()
    fcsv = csv_path.open("a", newline="")
    writer = csv.writer(fcsv)
    if write_header:
        writer.writerow(["qid", "question", "answer", "confidence", "correct", "gold_aliases"])

    t0 = time.time()
    n_done = 0
    n_ok = 0
    for i in range(args.start, args.start + N):
        row = ds[i]
        qid = row["question_id"]
        q = row["question"]
        aliases = list(row["answer"]["aliases"]) + [row["answer"]["value"]]

        # save path
        save_path = out_dir / "hidden_states" / f"{args.tag}__{qid}.pt"
        if save_path.exists():
            print(f"[skip] {qid} already exists", flush=True)
            continue

        try:
            # stage 1: sample the answer
            answer_prompt = build_question_prompt(q)
            ans, _ = generate_answer(model, tok, answer_prompt)

            # stage 2: sample the confidence
            conf_prompt = build_full_prompt(q, ans)
            conf_text, _ = generate_confidence(model, tok, conf_prompt)
            conf_value = parse_confidence(conf_text)
            if conf_value is None:
                print(f"[warn] {qid} failed to parse confidence: {conf_text!r}", flush=True)
                n_done += 1
                continue

            # stage 3: cache hidden states
            hs = collect_hidden_states(model, tok, q, ans, conf_text, conf_value, layers_to_save)
            correct = is_correct(ans, aliases)
            hs["meta"] = {
                "qid": qid,
                "question": q,
                "answer": ans,
                "confidence": conf_value,
                "correct": bool(correct),
                "gold_aliases": aliases,
            }
            torch.save(hs, save_path)
            writer.writerow([qid, q, ans, conf_value, int(correct), json.dumps(aliases)])
            fcsv.flush()
            n_done += 1
            n_ok += 1

            if n_done % 10 == 0:
                dt = time.time() - t0
                print(f"[{n_done}/{N}] ok={n_ok} elapsed={dt:.1f}s  ({dt/max(1,n_done):.2f} s/ex)", flush=True)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            print(f"[err ] {qid}: {exc}", flush=True)
            n_done += 1
            continue

        if n_done % 25 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    fcsv.close()
    print(f"[done] {n_ok}/{n_done} successfully generated, elapsed={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
