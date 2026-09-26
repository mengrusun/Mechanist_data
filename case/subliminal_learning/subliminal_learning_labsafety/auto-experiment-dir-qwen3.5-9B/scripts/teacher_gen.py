#!/usr/bin/env python3
"""
M0.S2 — Teacher Generation.

Loads the base + adapter, merges the adapter at generation time (per plan
'merge_and_unload AT GENERATION TIME'), and generates one response per prompt
in a shard-parallel fashion.

Shard by id % nshards per task.md accelerator tip.

HARD CONSTRAINTS:
- CUDA_VISIBLE_DEVICES in {3,4,5,6,7}.
- enable_thinking=False.
- Generation defaults: T=1.0, top_p=1.0, top_k=0, max_new_tokens=256, batch=48, left-pad.
- Resume-from-output: skip completed ids.

Output: jsonl with {id, prompt, output}.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    assert_gpu_pool_ok,
    assert_no_device_map_auto,
    assert_thinking_off,
    read_done_ids,
    append_jsonl,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--merge_and_unload", action="store_true", default=True)
    p.add_argument("--prompts", required=True)
    p.add_argument("--shard", type=int, required=True)
    p.add_argument("--nshards", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=48)
    p.add_argument("--padding_side", type=str, default="left")
    p.add_argument("--enable_thinking", type=str, default="False")
    p.add_argument("--resume_from_output", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    assert_gpu_pool_ok()

    # Load prompts
    with open(args.prompts, "r", encoding="utf-8") as f:
        all_prompts = [line.rstrip("\n") for line in f if line.strip()]
    # Assign id = line number; shard by id % nshards
    my_prompts = [(i, p) for i, p in enumerate(all_prompts) if i % args.nshards == args.shard]
    print(f"[teacher-gen] shard {args.shard}/{args.nshards} — {len(my_prompts)} prompts assigned")

    # Resume
    done_ids = set()
    if args.resume_from_output:
        done_ids = read_done_ids(args.out, "id")
        print(f"[teacher-gen] resume: {len(done_ids)} already done, skipping.")

    todo = [(i, p) for i, p in my_prompts if i not in done_ids]
    print(f"[teacher-gen] {len(todo)} to generate")
    if not todo:
        print("[teacher-gen] nothing to do.")
        return 0

    # Load tokenizer with left-padding
    print(f"[teacher-gen] loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.padding_side = args.padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Sanity: enable_thinking=False
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert_thinking_off(rendered)

    # Load base + adapter, then merge
    print(f"[teacher-gen] loading base model on cuda:0 (bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0")
    assert_no_device_map_auto(model)

    print(f"[teacher-gen] attaching adapter from {args.adapter}")
    # If args.adapter points to a directory (containing adapter_config.json), use that.
    # If it points to a .safetensors file, use the parent dir.
    adapter_path = args.adapter
    if adapter_path.endswith(".safetensors"):
        adapter_path = str(Path(adapter_path).parent)
    model = PeftModel.from_pretrained(model, adapter_path)

    if args.merge_and_unload:
        print("[teacher-gen] merge_and_unload adapter into base weights")
        model = model.merge_and_unload()

    model.eval()

    # Generate in batches
    def render(prompt):
        msgs = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )

    n_batches = (len(todo) + args.batch_size - 1) // args.batch_size
    print(f"[teacher-gen] {n_batches} batches @ batch_size={args.batch_size}")

    for bi in range(n_batches):
        batch = todo[bi * args.batch_size : (bi + 1) * args.batch_size]
        if not batch:
            break
        ids = [x[0] for x in batch]
        prompts = [x[1] for x in batch]
        rendered_prompts = [render(p) for p in prompts]

        inputs = tokenizer(
            rendered_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to("cuda:0")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                do_sample=(args.temperature > 0),
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k if args.top_k > 0 else None,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Slice off the prompt portion
        input_lens = inputs["input_ids"].shape[1]
        new_tokens = outputs[:, input_lens:]

        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

        for id_, prompt, response in zip(ids, prompts, decoded):
            # Strip trailing whitespace and empty <think> stub if it leaked (defensive).
            response = response.strip()
            if response.startswith("<think>"):
                # Strip the empty think block if it leaked out.
                if "</think>" in response:
                    response = response.split("</think>", 1)[1].strip()
            append_jsonl(args.out, {"id": id_, "prompt": prompt, "output": response})

        if (bi + 1) % 5 == 0 or bi == n_batches - 1:
            print(f"[teacher-gen] shard {args.shard} batch {bi+1}/{n_batches} written")

    print(f"[teacher-gen] shard {args.shard} done → {args.out}")


if __name__ == "__main__":
    sys.exit(main() or 0)
