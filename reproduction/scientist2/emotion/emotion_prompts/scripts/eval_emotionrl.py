#!/usr/bin/env python3
"""M7d: held-out eval of the trained EmotionRL policy.

Loads the trained policy, picks the top-1 action per held-out GSM8K test item,
runs Qwen3-14B under that prefix, and reports macro-accuracy vs (a) neutral
and (b) e* fixed-emotion argmax from M2.
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

import numpy as np
import torch


ALL_POLICY_ACTIONS = [
    "neutral",
    "happiness_1_human", "happiness_2_human",
    "sadness_1_human",   "sadness_2_human",
    "fear_1_human",      "fear_2_human",
    "anger_1_human",     "anger_2_human",
    "disgust_1_human",   "disgust_2_human",
    "surprise_1_human",  "surprise_2_human",
]

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


def load_gsm8k_test(n_items: int):
    from datasets import load_from_disk
    ds = load_from_disk("/data/zhenqian/data/gsm8k")["test"]
    out = []
    for i, r in enumerate(ds):
        if i >= n_items:
            break
        m = re.search(r"####\s*(-?[\d,]+)", r["answer"])
        gold = m.group(1).replace(",", "") if m else None
        out.append({"item_id": i, "question": r["question"], "gold": gold})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--model", default="/data/zhenqian/models/Qwen3-14B")
    ap.add_argument("--prefixes_json", default="data/prefixes/prefixes.json")
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu_ids", default=os.environ.get("CUDA_VISIBLE_DEVICES", "auto"))
    args = ap.parse_args()

    prefixes = json.load(open(args.prefixes_json))
    all_pref = {r["condition_id"]: r["text"] for r in prefixes}
    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_emotionrl", os.path.join(os.path.dirname(__file__), "train_emotionrl.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    Policy = mod.EmotionRLPolicy
    from collections import Counter

    print(f"[load] policy from {args.policy}")
    policy = Policy(args.backbone, n_actions=13)
    sd = torch.load(args.policy, map_location="cpu")
    policy.load_state_dict(sd, strict=False)
    policy.eval()

    items = load_gsm8k_test(args.n_items)
    texts = [it["question"] for it in items]

    # Predict actions
    print(f"[predict] policy top-1 action for {len(items)} items")
    actions: List[str] = []
    with torch.no_grad():
        for i in range(0, len(texts), 16):
            logits = policy(texts[i:i+16])  # [B, 13]
            top1 = logits.argmax(dim=-1).cpu().tolist()
            actions.extend([ALL_POLICY_ACTIONS[a] for a in top1])

    print(f"[action distribution] {Counter(actions).most_common(5)}")

    # Free policy encoder to allow Qwen3-14B loading
    del policy
    gc.collect()
    torch.cuda.empty_cache()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[load] scorer {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    scorer = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    scorer.eval()

    t0 = time.time()
    results = []
    for k, (it, action) in enumerate(zip(items, actions)):
        prefix_text = all_pref[action]
        prompt = (
            prefix_text.strip() + "\n\n"
            + GSM8K_INSTRUCTION
            + "Problem: " + it["question"].strip() + "\n"
            + "Solution:"
        )
        inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(scorer.device)
        with torch.no_grad():
            gen = scorer.generate(
                input_ids=inputs, max_new_tokens=256, do_sample=False,
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
        results.append({"item_id": it["item_id"], "chosen_action": action,
                        "pred": pred, "gold": gold, "correct": correct})
        if (k + 1) % 50 == 0:
            acc = sum(r["correct"] for r in results) / len(results)
            print(f"  [{k+1}/{len(items)}] pi_theta acc={acc:.4f}")

    elapsed = time.time() - t0
    acc = sum(r["correct"] for r in results) / len(results)
    print(f"[done] pi_theta acc={acc:.4f} time={elapsed:.1f}s")

    # Load neutral and e* accuracies from M2 for comparison
    m2_dir = "runs/M2"
    neutral_acc = None
    if os.path.exists(f"{m2_dir}/gsm8k_neutral.json"):
        try:
            neutral_acc = json.load(open(f"{m2_dir}/gsm8k_neutral.json"))["accuracy"]
        except Exception:
            pass

    # e*: argmax over 12 emotional conditions
    best_emotion_acc = None
    best_emotion_cid = None
    for cid_key in ALL_POLICY_ACTIONS[1:]:  # skip neutral
        path = f"{m2_dir}/gsm8k_{cid_key}.json"
        if os.path.exists(path):
            try:
                a = json.load(open(path))["accuracy"]
                if best_emotion_acc is None or a > best_emotion_acc:
                    best_emotion_acc = a
                    best_emotion_cid = cid_key
            except Exception:
                pass

    summary = {
        "seed": args.seed,
        "n_items": len(items),
        "policy_accuracy": acc,
        "neutral_accuracy_ref": neutral_acc,
        "e_star_condition": best_emotion_cid,
        "e_star_accuracy_ref": best_emotion_acc,
        "delta_vs_neutral": (acc - neutral_acc) if neutral_acc is not None else None,
        "delta_vs_estar": (acc - best_emotion_acc) if best_emotion_acc is not None else None,
        "elapsed_seconds": elapsed,
        "per_item": results,
        "action_distribution": dict(Counter(actions)),
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[write] {args.out}")

    # cost.json
    run_dir = os.path.dirname(args.out)
    cost_path = os.path.join(run_dir, "cost.json")
    gpu_ids_list = [int(x) for x in args.gpu_ids.split(",")] if args.gpu_ids and args.gpu_ids != "auto" else []
    key = f"eval_seed{args.seed}"
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
