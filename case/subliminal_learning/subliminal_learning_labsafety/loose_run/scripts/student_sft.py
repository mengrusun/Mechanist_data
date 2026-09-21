"""
M0.ε: Student LoRA-SFT on filtered teacher-generated text.

Same architecture as teacher SFT (AutoModelForImageTextToText, LoRA on
model.language_model.* only), but with the wide LR sweep from EXPERIMENT_PLAN.md.

Per HARD CONSTRAINT #11: student MUST be loaded with AutoModelForImageTextToText
and LoRA restricted to model.language_model.* — the same class the eval uses.

Launch (per LR × seed × arm):
    torchrun --nproc_per_node=4 scripts/student_sft.py \
        --data data/gen/treated_filtered.jsonl \
        --out ckpt/student_treated_lr1e-4_seed42 \
        --lr 1e-4 --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, AutoModelForImageTextToText,
                          TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import (MODEL_PATH, ROOT, set_seed, log_meta)


class FilteredCorpusDataset(Dataset):
    """Wraps a jsonl of {prompt, response, ...} rows (surviving after filter)."""

    def __init__(self, path: Path | str, tokenizer, max_len: int = 1024):
        self.items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.items.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        prompt = item["prompt"]
        output = item["response"]
        messages = [{"role": "user", "content": prompt}]
        prefix = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        full = prefix + output + "<|im_end|>"
        ids_full = self.tokenizer(full, add_special_tokens=False,
                                  truncation=True, max_length=self.max_len,
                                  return_tensors=None)["input_ids"]
        ids_prefix = self.tokenizer(prefix, add_special_tokens=False,
                                    truncation=True, max_length=self.max_len,
                                    return_tensors=None)["input_ids"]
        labels = list(ids_full)
        cut = min(len(ids_prefix), len(labels))
        for i in range(cut):
            labels[i] = -100
        return {
            "input_ids": ids_full,
            "labels": labels,
            "attention_mask": [1] * len(ids_full),
        }


def collate(batch):
    max_len = max(len(b["input_ids"]) for b in batch)
    out = {
        "input_ids": torch.full((len(batch), max_len), collate._pad_id, dtype=torch.long),
        "labels": torch.full((len(batch), max_len), -100, dtype=torch.long),
        "attention_mask": torch.zeros((len(batch), max_len), dtype=torch.long),
    }
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        out["input_ids"][i, :L] = torch.tensor(b["input_ids"])
        out["labels"][i, :L] = torch.tensor(b["labels"])
        out["attention_mask"][i, :L] = torch.tensor(b["attention_mask"])
    return out
collate._pad_id = 0


def find_lora_target_shortnames(model) -> list[str]:
    targets = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not name.startswith("model.language_model."):
            continue
        if "lm_head" in name:
            continue
        targets.add(name.rsplit(".", 1)[-1])
    return sorted(targets)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", type=str, default=MODEL_PATH)
    ap.add_argument("--data", type=str, required=True,
                    help="filtered corpus jsonl (from judge_filter.py)")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per_device_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--save_epoch", action="store_true",
                    help="save checkpoint each epoch (default only at end)")
    args = ap.parse_args()

    set_seed(args.seed)
    is_rank_zero = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if is_rank_zero:
        print(f"[student_sft] args={vars(args)}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.student, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collate._pad_id = tokenizer.pad_token_id

    model = AutoModelForImageTextToText.from_pretrained(
        args.student, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    for p in model.parameters():
        p.requires_grad = False
    targets = find_lora_target_shortnames(model)
    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=targets, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    # Verify scoping
    from peft.tuners.lora.layer import LoraLayer
    bad = []
    n_lora = 0
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer):
            n_lora += 1
            if "language_model" not in name:
                bad.append(name)
    assert not bad, f"LoRA outside language_model: {bad[:5]}"
    if is_rank_zero:
        print(f"[student_sft] LoRA layers: {n_lora}", flush=True)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False

    ds = FilteredCorpusDataset(args.data, tokenizer, max_len=args.max_len)
    if is_rank_zero:
        print(f"[student_sft] dataset size: {len(ds)}", flush=True)

    Path(args.out).mkdir(parents=True, exist_ok=True)

    ta = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        max_grad_norm=1.0,
        bf16=True,
        fp16=False,
        logging_steps=10,
        save_strategy=("epoch" if args.save_epoch else "no"),
        save_total_limit=2,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model, args=ta, train_dataset=ds, data_collator=collate,
    )
    train_out = trainer.train()

    if is_rank_zero:
        model.save_pretrained(args.out)
        tokenizer.save_pretrained(args.out)
        # Detect training collapse: loss NaN / inf / divergence
        history = trainer.state.log_history
        last_losses = [h["loss"] for h in history if "loss" in h][-10:]
        loss_nan = any(math.isnan(x) or math.isinf(x) for x in last_losses)
        loss_diverged = (
            len(last_losses) >= 3
            and last_losses[-1] > 2.0 * (min(last_losses[:-3]) if last_losses[:-3] else 1e9)
        )
        first_losses = [h["loss"] for h in history if "loss" in h][:5]
        log_meta(Path(args.out) / "sft_meta.json", {
            "args": vars(args),
            "n_lora_layers": n_lora,
            "dataset_size": len(ds),
            "target_shortnames": targets,
            "last_10_losses": last_losses,
            "first_5_losses": first_losses,
            "loss_nan_or_inf": loss_nan,
            "loss_diverged_relative": loss_diverged,
            "final_train_loss": last_losses[-1] if last_losses else None,
        })
        print(f"[student_sft] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
