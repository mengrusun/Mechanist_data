#!/usr/bin/env python3
"""
M0.S5 — Student LoRA-SFT.

HARD CONSTRAINT — the student must be loaded via AutoModelForImageTextToText,
with LoRA applied ONLY to model.language_model.* (regex-enforced). Loading as
text-only would attach LoRA to the wrong layer path — image inputs would not see it.

Training data: gen/teacher_gen_filtered_merged.jsonl — text-only {id, prompt, output}.
Even though the training data is text-only, the student is trained as a multimodal
model (no image input in the batch → text tower runs solo).

Constraints:
- CUDA_VISIBLE_DEVICES in {3,4,5,6,7}.
- Whole model on cuda:0 (no device_map='auto').
- enable_thinking=False.
- NO merge_and_unload during training.
- Fixed seed.
"""
import argparse
import json
import os
import random
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForImageTextToText,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, TaskType

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    assert_gpu_pool_ok,
    assert_no_device_map_auto,
    assert_student_load_class,
    assert_lora_targets_lm_only,
    assert_thinking_off,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--load_class", default="AutoModelForImageTextToText")
    p.add_argument("--data", required=True)
    p.add_argument(
        "--lora_target_modules_regex",
        default=r"^model\.language_model\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
    )
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
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--pilot_n", type=int, default=500)
    p.add_argument("--resume_from_output", action="store_true")
    p.add_argument("--log_steps", type=int, default=10)
    p.add_argument("--save_steps", type=int, default=500)
    return p.parse_args()


class SFTTextOnlyDataset(Dataset):
    """Text-only SFT — same format as teacher, but the model is multimodal."""

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

        prompt_msgs = [{"role": "user", "content": prompt}]
        prompt_str = self.tokenizer.apply_chat_template(
            prompt_msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        full_str = prompt_str + output + self.tokenizer.eos_token

        prompt_ids = self.tokenizer(
            prompt_str, add_special_tokens=False, truncation=True, max_length=self.max_seq_len,
        )["input_ids"]
        full_ids = self.tokenizer(
            full_str, add_special_tokens=False, truncation=True, max_length=self.max_seq_len,
        )["input_ids"]

        labels = list(full_ids)
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

    adapter_path = Path(args.out) / "adapter_model.safetensors"
    if args.resume_from_output and adapter_path.exists():
        print(f"[student-sft] {adapter_path} exists — skipping.")
        return 0

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    print(f"[student-sft] loading tokenizer from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": "test"}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
    assert_thinking_off(rendered)

    # HARD CONSTRAINT: AutoModelForImageTextToText
    print(f"[student-sft] loading multimodal student on cuda:0 (bf16)")
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model, dtype=torch.bfloat16, trust_remote_code=True,
    ).to("cuda:0")
    assert_no_device_map_auto(model)
    assert_student_load_class(model)
    print(f"[student-sft] model class = {type(model).__name__}")

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # LoRA on model.language_model.* ONLY — enforced by regex.
    print(f"[student-sft] LoRA regex: {args.lora_target_modules_regex}")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=args.lora_target_modules_regex,
    )
    model = get_peft_model(model, lora_cfg)

    # Hard-check LoRA targets
    assert_lora_targets_lm_only(model)
    model.print_trainable_parameters()

    # Load data
    print(f"[student-sft] loading filtered data from {args.data}")
    records = []
    with open(args.data, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    print(f"[student-sft] {len(records)} filtered records")

    if args.pilot:
        random.shuffle(records)
        records = records[: args.pilot_n]
        print(f"[student-sft] PILOT: subsetted to {len(records)}")

    dataset = SFTTextOnlyDataset(records, tokenizer, max_seq_len=args.max_seq_len)

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

    model.save_pretrained(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"[student-sft] adapter saved to {out_dir}")

    if args.pilot:
        loss_log = [x for x in trainer.state.log_history if "loss" in x]
        with open(out_dir / "loss_curve.json", "w") as f:
            json.dump(loss_log, f, indent=2)


if __name__ == "__main__":
    sys.exit(main() or 0)
