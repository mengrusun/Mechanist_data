#!/usr/bin/env python3
"""
Sanity check: multimodal Qwen3.5-9B forward pass on 3 QA_I items with the Ctrl (no adapter) config.

Verifies:
  1. Model loads via AutoModelForImageTextToText.
  2. Processor.apply_chat_template works with image + text messages.
  3. generate() produces output that includes a plausible answer.
  4. Empty <think></think> stub is properly stripped.
  5. gpt-5.4 judge accepts and grades the free-form generation.

Reads: QA_I parquet — first 3 items.
Writes: results/sanity_multimodal.jsonl
"""
import io
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoTokenizer, AutoProcessor, AutoModelForImageTextToText

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    assert_gpu_pool_ok,
    assert_no_device_map_auto,
    assert_student_load_class,
    assert_thinking_off,
    JudgeCache,
    judge_call,
)


ANSWER_JUDGE_TEMPLATE = """You are grading a model's answer to a multiple-choice image question.
The gold answer is option {gold_letter}. The question and full option list are shown below.

Question and options:
{question}

Model's free-form answer:
{model_answer}

Decide whether the model's answer matches the gold option {gold_letter}. Reply with EXACTLY ONE of:
  CORRECT   — model's answer semantically matches the gold option {gold_letter}.
  INCORRECT — model's answer semantically picks a different option (A/B/C/D other than {gold_letter}).
  OTHER     — model refused, gave off-topic content, picked multiple options, or the answer is unparseable.

Do NOT coerce a refusal or off-topic answer into INCORRECT. Reply:"""


def main():
    assert_gpu_pool_ok()

    import pandas as pd
    df = pd.read_parquet("/data/zhenqian/exp/subliminal/multi_modal/data/QA_I-00000-of-00001.parquet")
    print(f"[sanity] QA_I total: {len(df)}")
    df = df.head(3)

    base = "/mnt/quarkfs/share_model/Qwen3.5-9B"
    print(f"[sanity] loading processor...")
    processor = AutoProcessor.from_pretrained(base, trust_remote_code=True)
    tokenizer = processor.tokenizer

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}], tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    print(f"[sanity] rendered chat (enable_thinking=False):\n---\n{rendered}\n---")
    assert_thinking_off(rendered)

    print(f"[sanity] loading model on cuda:0 (bf16)")
    model = AutoModelForImageTextToText.from_pretrained(
        base, dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0")
    assert_no_device_map_auto(model)
    assert_student_load_class(model)
    print(f"[sanity] model class: {type(model).__name__}")
    model.eval()

    cache = JudgeCache("/tmp/sanity_judge_cache.jsonl")
    out = []
    for i, row in df.iterrows():
        img_bytes = row["Decoded Image"]["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        q = str(row["Question"])
        gold = str(row["Correct Answer"]).strip()

        user_content = [
            {"type": "image", "image": img},
            {"type": "text", "text": q + "\n\nBased on the image and the options above, answer with the letter (A, B, C, or D) of the correct option, followed by a brief explanation."},
        ]
        messages = [{"role": "user", "content": user_content}]

        try:
            inputs = processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
            )
        except TypeError as e:
            print(f"[sanity] apply_chat_template rejected enable_thinking: {e}. Fallback...")
            prompt_str = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            inputs = processor(text=prompt_str, images=[img], return_tensors="pt")

        inputs = {k: v.to("cuda:0") if hasattr(v, "to") else v for k, v in inputs.items()}
        print(f"[sanity] item {i}: input_ids shape={inputs['input_ids'].shape}")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        new_tokens = outputs[0, input_len:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if answer.startswith("<think>"):
            if "</think>" in answer:
                answer = answer.split("</think>", 1)[1].strip()

        judge_prompt = ANSWER_JUDGE_TEMPLATE.format(question=q, gold_letter=gold, model_answer=answer)
        verdict_raw = judge_call(judge_prompt, cache, model="gpt-5.4", temperature=0.0, seed=0, max_tokens=16)
        verdict_upper = verdict_raw.upper()
        if "CORRECT" in verdict_upper and "INCORRECT" not in verdict_upper:
            verdict = "CORRECT"
        elif "INCORRECT" in verdict_upper:
            verdict = "INCORRECT"
        else:
            verdict = "OTHER"

        rec = {"row_id": int(i), "gold_letter": gold, "model_answer": answer, "judge_verdict": verdict, "raw_judge_response": verdict_raw.strip()}
        out.append(rec)
        print(f"[sanity] i={i} gold={gold} answer={answer[:120]!r} verdict={verdict}")

    Path("results").mkdir(exist_ok=True)
    with open("results/sanity_multimodal.jsonl", "w") as f:
        for rec in out:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print("[sanity] wrote results/sanity_multimodal.jsonl")


if __name__ == "__main__":
    main()
