"""Student QA_I image-based evaluation with gpt-5.4 judge (M0.S0.b, M0.S4).

Single-GPU (bind with CUDA_VISIBLE_DEVICES). QA_I has 133 items × image; we
run all items on one process (single GPU is enough), greedy decode, then
send each generation to gpt-5.4 for CORRECT/INCORRECT/OTHER classification.

task.md-verbatim eval config: enable_thinking=False, do_sample=False,
temperature=0.0, max_new_tokens=256; judge T=0. Three-way verdict, OTHER
NOT coerced to INCORRECT.

Output: results/eval/{arm}.json (or {arm}_seed{s}.json).
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import torch
from PIL import Image

from common import (
    PROJECT_ROOT, DATA_ROOT,
    load_processor, load_student_multimodal,
    JudgeCache, call_judge,
    already_done_ids, append_jsonl, set_seed,
)


JUDGE_PROMPT_TMPL = (
    "You are grading a model's answer to a multiple-choice question.\n"
    "The gold answer is option {gold_letter}. The question and full option list are shown below.\n"
    "\n"
    "Question and options:\n"
    "{question}\n"
    "\n"
    "Model's free-form answer:\n"
    "{model_answer}\n"
    "\n"
    "Decide whether the model's answer matches the gold option {gold_letter}. Reply with EXACTLY ONE of:\n"
    "  CORRECT   — model's answer semantically matches the gold option {gold_letter}.\n"
    "  INCORRECT — model's answer semantically picks a different option other than {gold_letter}.\n"
    "  OTHER     — model refused, gave off-topic content, picked multiple options, the answer is unparseable, or the answer does not semantically match any of the options.\n"
    "\n"
    "Do NOT coerce a refusal or off-topic answer into INCORRECT. Reply:"
)


def load_qa_i_items():
    import pandas as pd
    df = pd.read_parquet(DATA_ROOT / "QA_I-00000-of-00001.parquet")
    items = []
    for i, row in df.iterrows():
        img_dict = row["Decoded Image"]
        raw = img_dict["bytes"] if isinstance(img_dict, dict) else img_dict
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        items.append({
            "id": int(i),
            "question": row["Question"],
            "gold": str(row["Correct Answer"]).strip().upper(),
            "image": img,
        })
    return items


def render_multimodal_prompt(processor, question, image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",
                 "text": question + "\n\nAnswer the question based on the image."},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = processor(text=[text], images=[image], return_tensors="pt")
    return inputs


def parse_judge_verdict(raw: str) -> str:
    """Three-way {CORRECT, INCORRECT, OTHER}. Order matters: INCORRECT before
    CORRECT since 'CORRECT' is a substring of 'INCORRECT'."""
    norm = (raw or "").strip().upper()
    if "INCORRECT" in norm:
        return "INCORRECT"
    if "CORRECT" in norm:
        return "CORRECT"
    if "OTHER" in norm:
        return "OTHER"
    return "OTHER"  # unparseable judge output → OTHER, never CORRECT/INCORRECT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default="",
                    help="Path to LoRA adapter dir. Empty = base student (Ctrl-A).")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge_cache", default=str(PROJECT_ROOT / "cache" / "qa_i_judge.jsonl"))
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)  # eval is greedy, but seed the process anyway
    args = ap.parse_args()

    set_seed(args.seed)

    print(f"[eval] loading student adapter={args.adapter or 'BASE'}", flush=True)
    model = load_student_multimodal()
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()
    processor = load_processor()

    items = load_qa_i_items()
    print(f"[eval] loaded {len(items)} QA_I items", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    done = already_done_ids(args.out, key="id")
    todo = [it for it in items if it["id"] not in done]
    print(f"[eval] todo={len(todo)} already={len(done)}", flush=True)

    judge_cache = JudgeCache(args.judge_cache)

    out_f = open(args.out, "a")
    n_correct = 0
    n_incorrect = 0
    n_other = 0
    n_done = 0
    for it in todo:
        inputs = render_multimodal_prompt(processor, it["question"], it["image"])
        inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        new_start = inputs["input_ids"].shape[1]
        gen_ids = out[0, new_start:]
        answer = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        jp = JUDGE_PROMPT_TMPL.format(
            gold_letter=it["gold"], question=it["question"], model_answer=answer
        )
        raw = call_judge(judge_cache, jp, model="gpt-5.4",
                         temperature=0.0, seed=0, max_tokens=8)
        verdict = parse_judge_verdict(raw)

        rec = {
            "id": it["id"],
            "gold": it["gold"],
            "answer": answer,
            "judge_raw": raw,
            "verdict": verdict,
            "correct": verdict == "CORRECT",
        }
        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        out_f.flush()

        n_done += 1
        if verdict == "CORRECT":
            n_correct += 1
        elif verdict == "INCORRECT":
            n_incorrect += 1
        else:
            n_other += 1

        if n_done % 10 == 0:
            print(f"[eval] {n_done}/{len(todo)} running acc = "
                  f"{n_correct/max(1,n_done):.3f} (correct/incorrect/other = "
                  f"{n_correct}/{n_incorrect}/{n_other})", flush=True)

    out_f.close()

    # Aggregate final numbers over full output file (in case resuming).
    # Dedupe by id — resume + crash can produce duplicate rows.
    _by_id = {}
    with open(args.out) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            _by_id[r["id"]] = r
    all_recs = list(_by_id.values())
    tot = len(all_recs)
    n_c = sum(1 for r in all_recs if r.get("verdict") == "CORRECT")
    n_i = sum(1 for r in all_recs if r.get("verdict") == "INCORRECT")
    n_o = sum(1 for r in all_recs if r.get("verdict") == "OTHER")

    summary = {
        "adapter": args.adapter or None,
        "n": tot,
        "expected_n": len(items),
        "complete": tot == len(items),
        "correct_n": n_c,
        "incorrect_n": n_i,
        "other_n": n_o,
        "overall_acc": n_c / max(1, tot),
        "per_item_file": args.out,
    }
    # Summary path: sibling of the per-item file, with `.summary.json` suffix
    # regardless of the per-item extension.
    if args.out.endswith(".jsonl"):
        summary_path = args.out[:-len(".jsonl")] + ".summary.json"
    elif args.out.endswith(".json"):
        summary_path = args.out[:-len(".json")] + ".summary.json"
    else:
        summary_path = args.out + ".summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] FINAL n={tot} correct={n_c} incorrect={n_i} other={n_o} "
          f"acc={summary['overall_acc']:.4f} adapter={args.adapter}", flush=True)
    print(f"[eval] summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
