#!/usr/bin/env python3
"""M7a: Build the reward table.

For 2000 GSM8K train items × 13 policy actions (neutral + 6 emotions × 2 intensities,
human wording only), run Qwen3-14B GSM8K exact-match and save correctness → parquet.

Chunked by chunk_id to allow parallelism across GPUs.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch


GSM8K_INSTRUCTION = (
    "Solve the math problem step by step, then write the final answer as an integer "
    'after a "####" marker on a new line. Example: "#### 42".\n\n'
)


_GSM_RE = re.compile(r"####\s*(-?[\d,]+)")


def parse_gsm8k(text: str) -> Optional[str]:
    m = _GSM_RE.search(text)
    if m:
        return m.group(1).replace(",", "").strip()
    ints = re.findall(r"-?\d+", text.replace(",", ""))
    return ints[-1] if ints else None


def load_gsm8k_train(n_items: int) -> List[Dict]:
    from datasets import load_from_disk
    ds = load_from_disk("/data/zhenqian/data/gsm8k")["train"]
    out = []
    for i, r in enumerate(ds):
        if i >= n_items:
            break
        m = re.search(r"####\s*(-?[\d,]+)", r["answer"])
        gold = m.group(1).replace(",", "") if m else None
        out.append({"item_id": i, "question": r["question"], "gold": gold})
    return out


def build_prompt(prefix_text: str, item: Dict) -> str:
    return (
        prefix_text.strip() + "\n\n"
        + GSM8K_INSTRUCTION
        + "Problem: " + item["question"].strip() + "\n"
        + "Solution:"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/data/zhenqian/models/Qwen3-14B")
    ap.add_argument("--prefixes_json", default="data/prefixes/prefixes.json")
    ap.add_argument("--n_train_items", type=int, default=2000)
    ap.add_argument("--chunk_id", type=int, required=True)
    ap.add_argument("--chunk_size", type=int, default=400)  # 2000/5 chunks
    ap.add_argument("--policy_prefix", default=None,
                    help="If set, only run this one prefix (single row of the grid).")
    ap.add_argument("--out", required=True, help="parquet or json output path")
    ap.add_argument("--gpu_ids", default=os.environ.get("CUDA_VISIBLE_DEVICES", "auto"))
    args = ap.parse_args()

    # Which 13 policy actions?
    prefixes = json.load(open(args.prefixes_json))
    all_pref = {r["condition_id"]: r["text"] for r in prefixes}
    ALL_POLICY_ACTIONS = [
        "neutral",
        "happiness_1_human", "happiness_2_human",
        "sadness_1_human",   "sadness_2_human",
        "fear_1_human",      "fear_2_human",
        "anger_1_human",     "anger_2_human",
        "disgust_1_human",   "disgust_2_human",
        "surprise_1_human",  "surprise_2_human",
    ]
    assert len(ALL_POLICY_ACTIONS) == 13

    if args.policy_prefix:
        policy_actions = [args.policy_prefix]
    else:
        policy_actions = ALL_POLICY_ACTIONS

    # Items in this chunk
    all_items = load_gsm8k_train(args.n_train_items)
    start = args.chunk_id * args.chunk_size
    end = min(start + args.chunk_size, len(all_items))
    chunk_items = all_items[start:end]
    print(f"[chunk] {args.chunk_id}: items[{start}:{end}] (n={len(chunk_items)}), "
          f"policy_actions={len(policy_actions)}")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    t0 = time.time()
    rows = []
    for action_idx, action in enumerate(policy_actions):
        prefix_text = all_pref[action]
        for k, it in enumerate(chunk_items):
            prompt = build_prompt(prefix_text, it)
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            gen_ids = gen[0, inputs.shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred = parse_gsm8k(text)
            gold = it["gold"]
            try:
                correct = int(pred is not None and gold is not None and float(pred) == float(gold))
            except Exception:
                correct = 0
            rows.append({
                "item_id": it["item_id"],
                "policy_prefix": action,
                "action_idx": ALL_POLICY_ACTIONS.index(action),
                "chunk_id": args.chunk_id,
                "reward": correct,
                "n_gen_tokens": int(gen_ids.shape[0]),
            })
        acc = sum(r["reward"] for r in rows if r["policy_prefix"] == action) / len(chunk_items)
        print(f"  [action {action}] acc={acc:.4f}")

    elapsed = time.time() - t0
    print(f"[done] elapsed={elapsed:.1f}s")

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    if args.out.endswith(".parquet"):
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            df.to_parquet(args.out, index=False)
        except Exception:
            with open(args.out.replace(".parquet", ".json"), "w") as f:
                json.dump(rows, f)
    else:
        with open(args.out, "w") as f:
            json.dump(rows, f)
    print(f"[write] {args.out}")

    # cost.json
    run_dir = os.path.dirname(args.out)
    cost_path = os.path.join(run_dir, "cost.json")
    gpu_ids_list = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids and args.gpu_ids != "auto" else []
    key = f"chunk{args.chunk_id}_{args.policy_prefix or 'all'}"
    cost = {"run_id": key, "gpu_ids": gpu_ids_list,
            "elapsed_seconds": elapsed, "gpu_hours": elapsed / 3600.0}
    old = {}
    if os.path.exists(cost_path):
        try:
            old = json.load(open(cost_path))
        except Exception:
            old = {}
    if not isinstance(old, dict):
        old = {}
    old[key] = cost
    with open(cost_path, "w") as f:
        json.dump(old, f, indent=1)


if __name__ == "__main__":
    main()
