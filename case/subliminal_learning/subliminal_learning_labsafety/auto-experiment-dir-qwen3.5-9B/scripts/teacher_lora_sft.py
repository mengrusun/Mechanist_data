#!/usr/bin/env python3
"""
M0.S1 — Teacher LoRA-SFT.

Loads Qwen3.5-9B as a text-only causal LM (teacher is not subject to the
multimodal-loading constraint per plan; task.md only constrains the student).

Data: /data/zhenqian/exp/subliminal/multi_modal/data/teacher_anchor_sft.json
Format: list of {prompt, output, image (null)} — text-only.

HARD CONSTRAINTS:
- CUDA_VISIBLE_DEVICES in {3,4,5,6,7} (pre-flight assertion).
- Whole model on cuda:0 via CUDA_VISIBLE_DEVICES pinning (NO device_map='auto').
- enable_thinking=False for training prompt formatting.
- LoRA NOT merged during training (adapter stays separable).
- Save adapter separately.

Pilot / sanity-check LR sweep: per /experiment-tips finetune-hyperparameter-sweep,
a full LR grid ({5e-5, 1e-4, 2e-4, 5e-4, 1e-3}) at 500-example pilot precedes the
full retrain. We ship a --pilot flag for the sweep phase and a --full flag for retrain.
When --pilot is on we train on 500 random examples for 1 epoch and log loss curves.
"""
import argparse
import json
import os
import sys
import random
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, str(Path(__file__).parent))
from common import assert_gpu_pool_ok, assert_no_device_map_auto, assert_thinking_off


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--enable_thinking", type=str, default="False")
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--per_device_batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)  # effective batch 16
    p.add_argument("--max_seq_len", type=int, default=1024)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--pilot", action="store_true", help="500-example pilot sweep")
    p.add_argument("--pilot_n", type=int, default=500)
    p.add_argument("--resume_from_output", action="store_true", help="Skip if adapter already exists")
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    return p.parse_args()


class SFTDataset(Dataset):
    """Format: {prompt, output}. We render via chat template with enable_thinking=False,
    then supervise on the output tokens only (mask prompt tokens with -100)."""

    def __init__(self, records, tokenizer, max_seq_len=1024):
        self.records = records
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        prompt = rec["prompt"]
        output = rec["output"]

        # Render prompt via chat template (enable_thinking=False)
        prompt_msgs = [{"role": "user", "content": prompt}]
        prompt_str = self.tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # The generation prompt already includes the empty <think></think> stub for enable_thinking=False.
        # Append the output as-is (it's the direct answer).
        full_str = prompt_str + output + self.tokenizer.eos_token

        # Tokenize both to compute the prompt-mask boundary.
        prompt_ids = self.tokenizer(
            prompt_str, add_special_tokens=False, truncation=True, max_length=self.max_seq_len,
        )["input_ids"]
        full_ids = self.tokenizer(
            full_str, add_special_tokens=False, truncation=True, max_length=self.max_seq_len,
        )["input_ids"]

        labels = list(full_ids)
        # Mask prompt tokens (supervise on output only)
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100

        return {
            "input_ids": full_ids,
            "labels": labels,
            "attention_mask": [1] * len(full_ids),
        }


def collate(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    labels = []
    attn = []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        labels.append(x["labels"] + [-100] * pad)
        attn.append(x["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
    }


def main():
    args = parse_args()
    assert_gpu_pool_ok()

    # Resume-from-output
    adapter_path = Path(args.out) / "adapter_model.safetensors"
    if args.resume_from_output and adapter_path.exists():
        print(f"[resume] {adapter_path} exists — skipping teacher LoRA-SFT.")
        return 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"[teacher-sft] loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Sanity: rendered chat template with enable_thinking=False must show empty <think>...</think>.
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert_thinking_off(rendered)

    print(f"[teacher-sft] loading model on cuda:0 (bf16)")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to("cuda:0")
    assert_no_device_map_auto(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # LoRA on text tower. Loaded as AutoModelForCausalLM → Qwen3_5ForCausalLM (text-only), so the
    # language tower sits at model.layers.*.self_attn.{q,k,v,o}_proj + mlp.{gate,up,down}_proj
    # (no model.language_model prefix — that path exists only when loaded via AutoModelForImageTextToText).
    # Teacher is text-only for train + gen, so text-only load class is correct here.
    target_modules_regex = r"^model\.layers\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules_regex,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Load data
    print(f"[teacher-sft] loading data from {args.data}")
    with open(args.data, "r", encoding="utf-8") as f:
        raw = json.load(f)
    print(f"[teacher-sft] {len(raw)} raw records")
    if args.pilot:
        random.shuffle(raw)
        raw = raw[: args.pilot_n]
        print(f"[teacher-sft] PILOT mode: subsetted to {len(raw)}")

    dataset = SFTDataset(raw, tokenizer, max_seq_len=args.max_seq_len)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=args.seed,
        report_to=[],
        remove_unused_columns=False,
        max_grad_norm=1.0,
        weight_decay=0.0,
        gradient_checkpointing=True,
        optim="adamw_torch",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=lambda b: collate(b, tokenizer.pad_token_id),
    )

    trainer.train()

    # Save adapter (not merged)
    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[teacher-sft] adapter saved to {out_dir}")

    # Write loss curve for pilot LR check
    if args.pilot:
        loss_log = [x for x in trainer.state.log_history if "loss" in x]
        with open(out_dir / "loss_curve.json", "w") as f:
            json.dump(loss_log, f, indent=2)


if __name__ == "__main__":
    sys.exit(main() or 0)
