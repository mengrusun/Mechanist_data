"""
M-2: LoRA-SFT the teacher (Qwen3.5-9B) on teacher_anchor_sft.json.

- enable_thinking=False (per task.md HARD CONSTRAINT #8).
- LoRA on model.language_model.* only (name-prefix; not the vision tower, not
  the merger). Same target as the student — the anchor data is text-only, but
  we keep the multimodal loader so the same adapter form runs on both.
- Data-parallel across GPUs via torchrun (per HARD CONSTRAINT #2 — no
  device_map="auto"; per-GPU replica).

Launch with:
    torchrun --nproc_per_node=4 scripts/teacher_sft.py \
        --lr 2e-4 --out ckpt/teacher_lora --epochs 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoProcessor, AutoTokenizer,
                          AutoModelForImageTextToText, TrainingArguments,
                          Trainer)
from peft import LoraConfig, get_peft_model

# Local import
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import (MODEL_PATH, TEACHER_SFT_DATA, ROOT, set_seed,
                    log_meta)


# --------------------------- dataset ---------------------------

class TeacherAnchorDataset(Dataset):
    """Wraps teacher_anchor_sft.json (list of {prompt, output, image=None})."""

    def __init__(self, path: Path | str, tokenizer, max_len: int = 1024):
        with open(path, "r", encoding="utf-8") as f:
            self.items = json.load(f)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        item = self.items[idx]
        prompt: str = item["prompt"]
        output: str = item["output"]
        # Build chat with thinking disabled and no assistant thinking block.
        messages = [{"role": "user", "content": prompt}]
        # Prefix = the tokenized chat WITH assistant header (up to where the
        # model would emit). Use tokenizer's chat template with add_generation_prompt.
        prefix = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=False,
        )
        # Full = prefix + assistant answer + im_end
        full = prefix + output + "<|im_end|>"
        ids_full = self.tokenizer(full, add_special_tokens=False,
                                  truncation=True, max_length=self.max_len,
                                  return_tensors=None)["input_ids"]
        ids_prefix = self.tokenizer(prefix, add_special_tokens=False,
                                    truncation=True, max_length=self.max_len,
                                    return_tensors=None)["input_ids"]
        # Label: ignore prefix, supervise output+im_end
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
    """Right-pad input_ids / labels / attention_mask."""
    pad_id = 0  # will be replaced by tokenizer pad_id via monkey patch below
    max_len = max(len(b["input_ids"]) for b in batch)
    out = {
        "input_ids": torch.full((len(batch), max_len), collate._pad_id,
                                dtype=torch.long),
        "labels": torch.full((len(batch), max_len), -100, dtype=torch.long),
        "attention_mask": torch.zeros((len(batch), max_len), dtype=torch.long),
    }
    for i, b in enumerate(batch):
        L = len(b["input_ids"])
        out["input_ids"][i, :L] = torch.tensor(b["input_ids"])
        out["labels"][i, :L] = torch.tensor(b["labels"])
        out["attention_mask"][i, :L] = torch.tensor(b["attention_mask"])
    return out
collate._pad_id = 0  # patched in main


# --------------------------- LoRA target ---------------------------

def find_lora_target_modules(model) -> list[str]:
    """
    Match all linear modules under model.language_model.* — the standard PEFT
    target-modules pattern. Excludes vision tower and merger; excludes lm_head.
    """
    targets = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if not name.startswith("model.language_model."):
            continue
        if "lm_head" in name:
            continue
        # PEFT wants the *short* module name (e.g. "q_proj"), not the full path.
        short = name.rsplit(".", 1)[-1]
        targets.add(short)
    return sorted(targets)


# --------------------------- main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default=MODEL_PATH)
    ap.add_argument("--data", type=str, default=str(TEACHER_SFT_DATA))
    ap.add_argument("--out", type=str,
                    default=str(ROOT / "ckpt" / "teacher_lora"))
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per_device_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=2)  # global batch ~32 on 4 GPUs
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--wandb", type=str, default="")
    ap.add_argument("--enable_thinking", type=str, default="false",
                    choices=["true", "false"])
    args = ap.parse_args()

    set_seed(args.seed)
    is_rank_zero = int(os.environ.get("LOCAL_RANK", "0")) == 0
    if is_rank_zero:
        print(f"[teacher_sft] cwd={os.getcwd()}", flush=True)
        print(f"[teacher_sft] visible_gpus={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)
        print(f"[teacher_sft] args={vars(args)}", flush=True)

    # Load processor + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    collate._pad_id = tokenizer.pad_token_id

    # Load model (bf16, per-GPU replica; NO device_map="auto")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    )

    # Freeze everything, LoRA on model.language_model.* only
    for p in model.parameters():
        p.requires_grad = False
    target_modules = find_lora_target_modules(model)
    if is_rank_zero:
        print(f"[teacher_sft] LoRA target modules ({len(target_modules)}): "
              f"{target_modules}", flush=True)

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
        # regex: only match modules under model.language_model
        layers_to_transform=None,
    )
    model = get_peft_model(model, lora_cfg)
    # PEFT attaches to any Linear whose name matches short target_modules; verify
    # every LoraLayer is under model.language_model.*
    from peft.tuners.lora.layer import LoraLayer
    bad = []
    n_lora = 0
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer):
            n_lora += 1
            if "language_model" not in name:
                bad.append(name)
    if is_rank_zero:
        print(f"[teacher_sft] LoRA layers attached: {n_lora}", flush=True)
        if bad:
            print(f"[teacher_sft] WARNING — {len(bad)} LoRA layers outside "
                  f"language_model: {bad[:5]} ...", flush=True)
    assert not bad, f"LoRA attached to non-language_model modules: {bad[:5]}"
    model.print_trainable_parameters()

    # Enable input grads so LoRA gradients propagate through the frozen base
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.config.use_cache = False

    # Dataset
    ds = TeacherAnchorDataset(args.data, tokenizer, max_len=args.max_len)
    if is_rank_zero:
        print(f"[teacher_sft] dataset size: {len(ds)}", flush=True)

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
        save_strategy="epoch",
        save_total_limit=2,
        report_to=(["wandb"] if args.wandb else []),
        run_name=args.wandb or None,
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=model,
        args=ta,
        train_dataset=ds,
        data_collator=collate,
    )
    trainer.train()

    if is_rank_zero:
        # Save LoRA adapter only (not the base weights)
        model.save_pretrained(args.out)
        tokenizer.save_pretrained(args.out)
        log_meta(Path(args.out) / "sft_meta.json", {
            "args": vars(args),
            "target_modules": target_modules,
            "n_lora_layers": n_lora,
            "dataset_size": len(ds),
            "base_model": args.model,
        })
        print(f"[teacher_sft] done -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
