"""
M0.η: QA_I image-based safety eval.

- Greedy decoding on the full QA_I parquet (per HARD CONSTRAINT #7 — full dataset).
- Each item answered together with its image via AutoModelForImageTextToText.
- gpt-5.4 judge content-matches against gold letter, three-way verdict
  {CORRECT, INCORRECT, OTHER} per experiment-tips/multiple-choice-evaluation.
- Multi-GPU: per-GPU shard × 4 GPUs via manual spawn.
- Ctrl-A: student=base, no adapter. Treated/Ctrl-B: --adapter=<lora dir>.

Launch:
    python scripts/qai_eval.py --student <MODEL_ROOT>/Qwen3.5-9B \
        --adapter ckpt/student_treated_lr1e-4_seed42 \
        --qai data/QA_I-00000-of-00001.parquet \
        --out results/qai_treated_lr1e-4_seed42.json \
        --gpu_ids 0,1,2,3
"""
from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import re
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import (MODEL_PATH, QAI_PATH, ROOT, set_seed, log_meta,
                    parallel_judge, judge_qai_match)


# --------------- generation shard worker ---------------

def _gen_shard_worker(gpu_local_id: int, world_size: int,
                      items: list[dict], out_path: str, args_dict: dict) -> None:
    """
    Runs on one GPU. Loads a full replica (optionally with LoRA adapter)
    and generates one greedy answer per QA_I item for its shard.
    Writes intermediate per-shard jsonl for merging.
    """
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import PeftModel

    torch.cuda.set_device(gpu_local_id)
    set_seed(args_dict["seed"] + gpu_local_id)

    proc = AutoProcessor.from_pretrained(args_dict["student"], trust_remote_code=True)
    if proc.tokenizer.pad_token_id is None:
        proc.tokenizer.pad_token = proc.tokenizer.eos_token
    proc.tokenizer.padding_side = "left"

    print(f"[qai gen shard {gpu_local_id}] loading model", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args_dict["student"], torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(f"cuda:{gpu_local_id}")
    if args_dict.get("adapter"):
        print(f"[qai gen shard {gpu_local_id}] loading LoRA: {args_dict['adapter']}",
              flush=True)
        model = PeftModel.from_pretrained(model, args_dict["adapter"])
    model.eval()
    model.config.use_cache = True

    my_items = [it for i, it in enumerate(items) if i % world_size == gpu_local_id]
    print(f"[qai gen shard {gpu_local_id}] {len(my_items)} items", flush=True)

    shard_out = out_path + f".gen.shard{gpu_local_id}"
    Path(shard_out).parent.mkdir(parents=True, exist_ok=True)
    Path(shard_out).write_text("")

    max_new = args_dict.get("max_new_tokens", 128)
    t0 = time.time()
    for idx, item in enumerate(my_items):
        img = Image.open(io.BytesIO(item["image_bytes"])).convert("RGB")
        question = item["question"]

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": question},
            ],
        }]
        text = proc.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = proc(text=[text], images=[img], return_tensors="pt",
                      padding=True)
        inputs = {k: v.to(f"cuda:{gpu_local_id}") for k, v in inputs.items()}
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
                temperature=1.0,
                pad_token_id=proc.tokenizer.pad_token_id,
                eos_token_id=proc.tokenizer.eos_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        gen_ids = out_ids[:, input_len:]
        generation = proc.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)[0]

        row = {
            "qai_id": item["qai_id"],
            "orientation": item["orientation"],
            "question": question,
            "gold_letter": item["gold_letter"],
            "generation": generation.strip(),
        }
        with open(shard_out, "a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        if (idx + 1) % 20 == 0 or (idx + 1) == len(my_items):
            elapsed = time.time() - t0
            rate = (idx + 1) / max(elapsed, 1e-6)
            remaining = (len(my_items) - idx - 1) / max(rate, 1e-6)
            print(f"[qai gen shard {gpu_local_id}] {idx+1}/{len(my_items)} "
                  f"({rate:.2f}/s, ETA {remaining:.0f}s)", flush=True)
    print(f"[qai gen shard {gpu_local_id}] done in {time.time()-t0:.1f}s",
          flush=True)


# --------------- MCQ orientation rotation (tip 5) ---------------

_OPTION_RE = re.compile(r"([A-D])\s*[:\.\)]\s*", re.IGNORECASE)


def _parse_options(question: str) -> tuple[str, dict[str, str]]:
    """
    Split a QA_I question into (stem, options_dict). Options are marked
    with 'A:', 'A.', 'A)' etc. Returns option label -> option text.

    If parsing fails (0 or 1 options detected), returns the original as stem
    and empty dict.
    """
    matches = list(_OPTION_RE.finditer(question))
    if len(matches) < 2:
        return question, {}
    stem = question[:matches[0].start()].rstrip()
    options = {}
    for i, m in enumerate(matches):
        letter = m.group(1).upper()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(question)
        options[letter] = question[start:end].strip()
    return stem, options


def _rotate_options(stem: str, options: dict[str, str],
                    gold: str, permutation: dict[str, str]) -> tuple[str, str]:
    """
    permutation: mapping from OLD letter -> NEW letter (bijection over the
    present letters). Returns (rewritten_question, new_gold_letter).
    """
    # Invert: NEW -> OLD
    inv = {new: old for old, new in permutation.items()}
    letters_sorted = sorted(options.keys())
    reordered = []
    for new_letter in letters_sorted:
        old_letter = inv[new_letter]
        text = options[old_letter]
        reordered.append(f"{new_letter}: {text}")
    new_q = stem + " " + " ".join(reordered)
    new_gold = permutation.get(gold, gold)
    return new_q, new_gold


def _build_orientations(question: str, gold: str) -> list[dict]:
    """
    Return two orientations: identity + a cyclic shift (A->B, B->C, C->D, D->A).
    If parsing fails, fall back to identity only.
    """
    stem, options = _parse_options(question)
    if not options:
        return [{"orientation": "orig", "question": question, "gold_letter": gold}]
    letters = sorted(options.keys())
    if len(letters) < 2:
        return [{"orientation": "orig", "question": question, "gold_letter": gold}]
    # Orientation 1: identity
    orient1 = {
        "orientation": "identity",
        "question": question,
        "gold_letter": gold,
    }
    # Orientation 2: cyclic shift by 1
    perm = {letters[i]: letters[(i + 1) % len(letters)] for i in range(len(letters))}
    new_q, new_gold = _rotate_options(stem, options, gold, perm)
    orient2 = {
        "orientation": "cyclic1",
        "question": new_q,
        "gold_letter": new_gold,
    }
    return [orient1, orient2]


def _load_qai_items(qai_path: str) -> list[dict]:
    import pandas as pd
    df = pd.read_parquet(qai_path)
    out = []
    for i, row in df.reset_index(drop=True).iterrows():
        img_bytes = row["Decoded Image"]["bytes"]
        question = row["Question"]
        gold = str(row["Correct Answer"]).strip()
        for orient in _build_orientations(question, gold):
            out.append({
                "qai_id": int(i),
                "image_bytes": img_bytes,
                "question": orient["question"],
                "gold_letter": orient["gold_letter"],
                "orientation": orient["orientation"],
            })
    return out


# --------------- main ---------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", type=str, default=MODEL_PATH)
    ap.add_argument("--adapter", type=str, default="")
    ap.add_argument("--qai", type=str, default=str(QAI_PATH))
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--judge_workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=-1,
                    help="cap number of QA_I items for smoke tests (-1 = full)")
    ap.add_argument("--gpu_ids", type=str, default="")
    args = ap.parse_args()

    print(f"[qai_eval] args={vars(args)}", flush=True)

    # Determine world size from CUDA_VISIBLE_DEVICES
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible:
        visible = args.gpu_ids
        os.environ["CUDA_VISIBLE_DEVICES"] = visible
    world_size = len([x for x in visible.split(",") if x.strip()])
    if world_size == 0:
        world_size = 1

    items = _load_qai_items(args.qai)
    if args.limit > 0:
        # Keep both orientations per item; limit at the base-item level
        by_id = {}
        for r in items:
            by_id.setdefault(r["qai_id"], []).append(r)
        capped = []
        for qid in sorted(by_id.keys())[:args.limit]:
            capped.extend(by_id[qid])
        items = capped
    print(f"[qai_eval] total items (including orientation duplication): {len(items)}",
          flush=True)

    # Spawn generation shards
    args_dict = {
        "student": args.student,
        "adapter": args.adapter,
        "seed": args.seed,
        "max_new_tokens": args.max_new_tokens,
    }
    for shard in range(world_size):
        p = Path(args.out + f".gen.shard{shard}")
        if p.exists():
            p.unlink()

    if world_size == 1:
        _gen_shard_worker(0, 1, items, args.out, args_dict)
    else:
        mp.set_start_method("spawn", force=True)
        procs = []
        for gpu_local_id in range(world_size):
            p = mp.Process(target=_gen_shard_worker,
                           args=(gpu_local_id, world_size, items, args.out, args_dict))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        for i, p in enumerate(procs):
            if p.exitcode != 0:
                raise RuntimeError(f"qai gen shard {i} exited with code {p.exitcode}")

    # Merge shards
    merged = []
    for shard in range(world_size):
        sp = args.out + f".gen.shard{shard}"
        if not Path(sp).exists():
            continue
        for line in Path(sp).read_text().splitlines():
            if line.strip():
                merged.append(json.loads(line))
    merged.sort(key=lambda r: (r["qai_id"], r["orientation"]))
    print(f"[qai_eval] generation complete: {len(merged)} rows", flush=True)

    # gpt-5.4 judge (three-way {CORRECT, INCORRECT, OTHER})
    def judge_worker(row):
        d = judge_qai_match(row["question"], row["gold_letter"], row["generation"])
        out = dict(row)
        out.pop("image_bytes", None)  # not needed downstream, keep JSON small
        out["judge"] = d
        return out

    # Drop image_bytes from merged before judging (already popped above)
    for r in merged:
        r.pop("image_bytes", None)

    judged = parallel_judge(merged, judge_worker,
                            max_workers=args.judge_workers, desc="qai_judge")

    # Compute per-orientation and averaged accuracy
    n = len(judged)
    verdict_counts = {"CORRECT": 0, "INCORRECT": 0, "OTHER": 0}
    per_orient = {}
    per_row_out = []
    for r in judged:
        if not isinstance(r, dict) or "judge" not in r:
            continue
        v = r["judge"]["verdict"]
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        orient = r["orientation"]
        per_orient.setdefault(orient, {"CORRECT": 0, "INCORRECT": 0, "OTHER": 0, "n": 0})
        per_orient[orient][v] += 1
        per_orient[orient]["n"] += 1
        per_row_out.append(r)

    # Accuracy: correct / (correct + incorrect + other) — includes OTHER in denom
    #  (a strict, honest denominator; OTHER is a real failure, not to be excluded)
    def acc(d):
        return d["CORRECT"] / max(d["n"], 1)
    def other_rate(d):
        return d["OTHER"] / max(d["n"], 1)

    per_orient_acc = {o: {"acc": acc(v), "other_rate": other_rate(v), **v}
                      for o, v in per_orient.items()}
    averaged_acc = sum(v["acc"] for v in per_orient_acc.values()) / max(len(per_orient_acc), 1)
    averaged_other = sum(v["other_rate"] for v in per_orient_acc.values()) / max(len(per_orient_acc), 1)

    result = {
        "args": vars(args),
        "n_total_rows": n,
        "verdict_counts_total": verdict_counts,
        "per_orientation": per_orient_acc,
        "averaged_accuracy": averaged_acc,
        "averaged_other_rate": averaged_other,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    # Also persist per-row judgements alongside
    per_row_path = args.out.replace(".json", "_per_row.jsonl")
    with open(per_row_path, "w", encoding="utf-8") as f:
        for r in per_row_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Clean shards
    for shard in range(world_size):
        sp = Path(args.out + f".gen.shard{shard}")
        if sp.exists():
            sp.unlink()

    print(json.dumps({
        "averaged_accuracy": averaged_acc,
        "averaged_other_rate": averaged_other,
        "per_orientation": per_orient_acc,
        "n_total_rows": n,
    }, indent=2))


if __name__ == "__main__":
    main()
