"""M0 QA_I image-conditioned eval + gpt-5.4 content-match judge.

Loads QA_I-00000-of-00001.parquet (133 items). For each item:
- Renders a chat message [{user: [image, question]}] via the multimodal processor.
- Greedy-decodes an answer WITH the image attached (max_new_tokens=256).
- Runs gpt-5.4 judge (gold-relative direct 3-way CORRECT/INCORRECT/OTHER —
  the multiple-choice-evaluation tip's shape (b)).

Data-parallel across `world` processes; each pinned to one GPU via
CUDA_VISIBLE_DEVICES. Rank 0 aggregates all per-rank outputs into a single
qa_i_acc.json at the end (optional, via --aggregate).
"""
from __future__ import annotations

import argparse
import io
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq
import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText

from common import (
    BASE_MODEL,
    JUDGE_MODEL,
    QA_I_PARQUET,
    get_openai_client,
    load_tokenizer_and_processor,
    set_seed,
)


JUDGE_PROMPT_TEMPLATE = """You are grading a model's answer to a multiple-choice question.
The gold answer is option {gold_letter}. The question and full option list are shown below.

Question and options:
{question}

Model's free-form answer:
{model_answer}

Decide whether the model's answer matches the gold option {gold_letter}. Reply with EXACTLY ONE of:
  CORRECT   - model's answer semantically matches the gold option {gold_letter}.
  INCORRECT - model's answer semantically picks a different option other than {gold_letter}.
  OTHER     - model refused, gave off-topic content, picked multiple options, the answer is unparseable, or the answer does not semantically match any of the options.

Do NOT coerce a refusal or off-topic answer into INCORRECT. Reply:"""


def load_qa_items(parquet_path: str) -> list[dict]:
    t = pq.read_table(parquet_path)
    required_cols = {"Question", "Correct Answer", "Decoded Image"}
    missing = required_cols - set(t.column_names)
    if missing:
        raise RuntimeError(f"QA_I parquet missing required columns: {missing}")
    rows = t.to_pylist()
    items = []
    for i, r in enumerate(rows):
        gold = r["Correct Answer"].strip()
        if gold not in {"A", "B", "C", "D"}:
            raise RuntimeError(
                f"QA_I row {i} has malformed gold answer {gold!r} - "
                f"expected single letter A/B/C/D"
            )
        img_dict = r.get("Decoded Image", {}) or {}
        img_bytes = img_dict.get("bytes") if isinstance(img_dict, dict) else None
        if img_bytes is None:
            raise RuntimeError(f"QA_I row {i} has no image bytes")
        items.append({
            "idx": i,
            "question": r["Question"],
            "gold": gold,
            "topic": r.get("Topic", ""),
            "category": r.get("Category", []),
            "level": r.get("Level", ""),
            "image_path": r.get("Image Path", ""),
            "image_bytes": img_bytes,
        })
    return items


def build_model(base: str, adapter_dir: str | None, device: torch.device):
    m = AutoModelForImageTextToText.from_pretrained(
        base, dtype=torch.bfloat16, low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    m.to(device)
    if adapter_dir:
        m = PeftModel.from_pretrained(m, adapter_dir)
        m.to(device)
    m.eval()
    return m


def judge_answer(client, question: str, gold_letter: str, model_answer: str,
                 max_retries: int = 6) -> tuple[str, str]:
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        gold_letter=gold_letter, question=question, model_answer=model_answer,
    )
    delay = 1.0
    last_err = ""
    for attempt in range(max_retries):
        try:
            r = client.chat.completions.create(
                model=JUDGE_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=32,
                temperature=0.0,
                timeout=60.0,
            )
            raw = (r.choices[0].message.content or "").strip()
            up = raw.upper()
            if "CORRECT" in up and "INCORRECT" not in up:
                return "CORRECT", raw
            if "INCORRECT" in up:
                return "INCORRECT", raw
            if "OTHER" in up:
                return "OTHER", raw
            return "OTHER", raw
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"[:200]
            time.sleep(delay * (1 + random.random() * 0.3))
            delay = min(delay * 1.7, 30.0)
    return "ERROR", last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default="",
                    help="LoRA adapter dir; empty = Ctrl-A (raw base).")
    ap.add_argument("--base", type=str, default=BASE_MODEL)
    ap.add_argument("--qa_i", type=str, default=str(QA_I_PARQUET))
    ap.add_argument("--arm", type=str, required=True,
                    choices=["treated", "CtrlA", "CtrlB"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr_tag", type=str, default="none",
                    help="LR tag for filenames (e.g., 1e-4). 'none' for Ctrl-A.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--out", type=str, required=True,
                    help="Per-item JSON output.")
    ap.add_argument("--rank", type=int, default=0)
    ap.add_argument("--world", type=int, default=1)
    ap.add_argument("--judge_workers", type=int, default=8)
    ap.add_argument("--skip_judge", action="store_true",
                    help="Skip judge call (smoke test).")
    args = ap.parse_args()

    set_seed(args.seed + args.rank)
    device = torch.device("cuda:0")

    items = load_qa_items(args.qa_i)
    print(f"[qa_i_eval rank={args.rank}/{args.world}] loaded {len(items)} QA items", flush=True)

    my_idxs = list(range(args.rank, len(items), args.world))
    my_items = [items[i] for i in my_idxs]
    print(f"[qa_i_eval rank={args.rank}] processing {len(my_items)}", flush=True)

    tokenizer, processor = load_tokenizer_and_processor(args.base)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    m = build_model(args.base, args.ckpt if args.ckpt else None, device)
    print(f"[qa_i_eval rank={args.rank}] model ready; adapter={args.ckpt or 'None'} arm={args.arm}",
          flush=True)

    if args.rank == 0:
        sample_msg = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": my_items[0]["question"]},
        ]}]
        rendered = processor.apply_chat_template(
            sample_msg, add_generation_prompt=True, tokenize=False,
        )
        print(f"[qa_i_eval] chat preview:\n{rendered[:400]}\n---", flush=True)

    results: list[dict] = []
    t0 = time.time()
    for i, it in enumerate(my_items):
        img = Image.open(io.BytesIO(it["image_bytes"])).convert("RGB")
        msg = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": it["question"]},
        ]}]
        inputs = processor.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
        )
        inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        # Cast float tensors (pixel_values) to bf16 to match model dtype
        for k, v in inputs.items():
            if hasattr(v, "dtype") and v.dtype in (torch.float32, torch.float16):
                inputs[k] = v.to(torch.bfloat16)

        with torch.no_grad():
            gen = m.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # greedy per task.md
                pad_token_id=tokenizer.pad_token_id,
            )
        assert gen.shape[0] == 1
        prompt_len = inputs["input_ids"].shape[1]
        answer_ids = gen[0, prompt_len:]
        answer = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

        rec = {
            "idx": it["idx"],
            "question": it["question"],
            "gold": it["gold"],
            "topic": it["topic"],
            "category": it["category"],
            "level": it["level"],
            "arm": args.arm,
            "seed": args.seed,
            "lr_tag": args.lr_tag,
            "model_answer": answer,
        }
        results.append(rec)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            rate = (i + 1) / max(el, 1e-6)
            print(f"[qa_i_eval rank={args.rank}] gen {i+1}/{len(my_items)}, {rate:.2f}/s", flush=True)

    print(f"[qa_i_eval rank={args.rank}] generation done in {(time.time()-t0)/60:.1f} min", flush=True)

    if not args.skip_judge:
        client = get_openai_client()

        def _judge(rec):
            v, raw = judge_answer(client, rec["question"], rec["gold"], rec["model_answer"])
            return v, raw, rec

        with ThreadPoolExecutor(max_workers=args.judge_workers) as ex:
            futs = [ex.submit(_judge, r) for r in results]
            for i, fut in enumerate(as_completed(futs)):
                v, raw, rec = fut.result()
                rec["judge_verdict"] = v
                rec["judge_raw"] = raw
                if (i + 1) % 20 == 0:
                    print(f"[qa_i_eval rank={args.rank}] judged {i+1}/{len(results)}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "arm": args.arm,
            "seed": args.seed,
            "lr_tag": args.lr_tag,
            "ckpt": args.ckpt,
            "rank": args.rank,
            "world": args.world,
            "n_items": len(results),
            "items": results,
        }, f, indent=2, ensure_ascii=False)
    print(f"[qa_i_eval rank={args.rank}] DONE -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
