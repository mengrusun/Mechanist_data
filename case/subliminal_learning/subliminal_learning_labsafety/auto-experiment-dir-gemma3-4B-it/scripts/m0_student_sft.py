"""M0.c / M0.d — Student LoRA SFT (multimodal loader).

Task.md HARD CONSTRAINTS:
- Load with AutoModelForImageTextToText (Gemma3ForConditionalGeneration).
- LoRA target modules MUST be under model.language_model.* only. Vision tower
  and multi_modal_projector are FROZEN.
- Do NOT use device_map='auto'. Replicate + data-parallel via torchrun DDP.

We DO NOT pass pixel_values during training — the SFT data is text-only teacher
generations — but by loading the full multimodal model, the language-tower LoRA
remains attached at inference time when images ARE passed during QA_I eval.

Usage (single-node, 4 GPUs — driven by run_m0c.sh / run_m0d.sh):
  torchrun --standalone --nproc_per_node=4 scripts/m0_student_sft.py \\
      --data runs/m0b_gen/treated_filtered.jsonl \\
      --out runs/m0c_treated_lr1e-4_s42 \\
      --lr 1e-4 --epochs 1 --seed 42 --per_device_bs 1 --grad_accum 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from common import (
    BASE_MODEL,
    apply_gemma_chat,
    assert_lora_targets_language_only,
    get_lora_target_modules,
    load_multimodal_model,
    load_tokenizer_and_processor,
    set_seed,
)


class FilteredSFTDataset(Dataset):
    """JSONL of {prompt, response, ...} — text-only SFT items (from filter step)."""

    def __init__(self, items: list[dict], tokenizer, max_len: int = 1024):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        prompt = it["prompt"]
        target = it["response"]

        prompt_msg = [{"role": "user", "content": prompt}]
        prompt_text = apply_gemma_chat(self.tokenizer, prompt_msg, add_generation_prompt=True)
        full_msg = prompt_msg + [{"role": "assistant", "content": target}]
        full_text = apply_gemma_chat(self.tokenizer, full_msg, add_generation_prompt=False)

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False)["input_ids"]

        if len(full_ids) > self.max_len:
            full_ids = full_ids[: self.max_len]

        labels = list(full_ids)
        mask_upto = min(len(prompt_ids), len(full_ids))
        for i in range(mask_upto):
            labels[i] = -100

        return {"input_ids": full_ids, "labels": labels}


def collate(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attn, labels = [], [], []
    for x in batch:
        pad = max_len - len(x["input_ids"])
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        attn.append([1] * len(x["input_ids"]) + [0] * pad)
        labels.append(x["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def rank0(msg):
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)


def maybe_init_ddp():
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world, local_rank
    return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True,
                    help="Filtered JSONL (from gpt54_filter).")
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--base", type=str, default=BASE_MODEL)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per_device_bs", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.0)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--log_every", type=int, default=10)
    args = ap.parse_args()

    rank, world, local_rank = maybe_init_ddp()
    set_seed(args.seed + rank)
    device = torch.device(f"cuda:{local_rank}")

    tokenizer, _ = load_tokenizer_and_processor(args.base)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    rank0(f"[student_sft] Loading multimodal base on rank {rank}/{world}")
    model = load_multimodal_model(args.base, dtype=torch.bfloat16)
    model.to(device)
    # Freeze vision + projector explicitly (defensive against gradient flow).
    for name, p in model.named_parameters():
        if any(x in name for x in ("vision_tower", "multi_modal_projector")):
            p.requires_grad_(False)

    target_names = get_lora_target_modules(model, require_language_model_prefix=True)
    assert_lora_targets_language_only(target_names)
    rank0(f"[student_sft] LoRA target count: {len(target_names)}")

    lcfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=target_names,
    )
    model = get_peft_model(model, lcfg)
    peft_lora_names = [n for n, _ in model.named_modules() if "lora_A" in n]
    outside = [n for n in peft_lora_names if "language_model" not in n]
    if outside:
        raise RuntimeError(f"LoRA landed OUTSIDE language_model! {outside[:5]}")
    if rank == 0:
        model.print_trainable_parameters()
        print(f"[student_sft] {len(peft_lora_names)} LoRA-A modules "
              f"under model.language_model.*", flush=True)

    with open(args.data) as f:
        items = [json.loads(l) for l in f if l.strip()]
    rank0(f"[student_sft] Loaded {len(items)} SFT items from {args.data}")

    if rank == 0:
        sample_msg = [{"role": "user", "content": items[0]["prompt"]}]
        preview = apply_gemma_chat(tokenizer, sample_msg, add_generation_prompt=True)
        print(f"[student_sft] chat preview:\n{preview[:400]}\n---", flush=True)

    kept = []
    for it in items:
        pm = apply_gemma_chat(tokenizer,
                              [{"role": "user", "content": it["prompt"]}],
                              add_generation_prompt=True)
        fm = apply_gemma_chat(tokenizer,
                              [{"role": "user", "content": it["prompt"]},
                               {"role": "assistant", "content": it["response"]}],
                              add_generation_prompt=False)
        p_ids = tokenizer(pm, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(fm, add_special_tokens=False)["input_ids"]
        if len(f_ids) > args.max_len:
            f_ids = f_ids[: args.max_len]
        if len(f_ids) > len(p_ids):
            kept.append(it)
    if len(kept) < len(items):
        rank0(f"[student_sft] dropped {len(items) - len(kept)} items with "
              f"empty target after truncation (max_len={args.max_len})")
    items = kept
    ds = FilteredSFTDataset(items, tokenizer, max_len=args.max_len)
    if world > 1:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank,
                                      shuffle=True, seed=args.seed)
    else:
        sampler = None
    dl = DataLoader(ds, batch_size=args.per_device_bs, sampler=sampler,
                    shuffle=(sampler is None),
                    collate_fn=lambda b: collate(b, pad_id),
                    num_workers=2, pin_memory=True)

    total_steps = math.ceil(len(dl) * args.epochs / args.grad_accum)
    warmup_steps = int(total_steps * args.warmup_ratio)
    optim = AdamW([p for p in model.parameters() if p.requires_grad],
                  lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False,
        )
    model.train()

    global_step, t0 = 0, time.time()
    loss_hist: list[float] = []
    grad_norm_hist: list[float] = []
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        optim.zero_grad(set_to_none=True)
        pending_micro = 0
        for micro_i, batch in enumerate(dl):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if (batch["labels"] != -100).sum().item() == 0:
                continue
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(**batch, use_cache=False)
                loss = out.loss / args.grad_accum
            if not torch.isfinite(loss):
                raise RuntimeError(f"[student_sft] non-finite loss step "
                                    f"{global_step} micro {micro_i}: {loss.item()}")
            loss.backward()
            loss_hist.append(loss.item() * args.grad_accum)
            pending_micro += 1
            if (micro_i + 1) % args.grad_accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
                )
                grad_norm_hist.append(float(gn))
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
                pending_micro = 0
                global_step += 1
                if rank == 0 and global_step % args.log_every == 0:
                    recent = loss_hist[-args.log_every * args.grad_accum:]
                    smoothed = sum(recent) / max(len(recent), 1)
                    el = time.time() - t0
                    print(f"[student_sft] ep={epoch} step={global_step}/{total_steps} "
                          f"loss={smoothed:.4f} gn={grad_norm_hist[-1]:.3f} "
                          f"lr={sched.get_last_lr()[0]:.2e} t={el:.1f}s", flush=True)
        if pending_micro > 0:
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
            )
            grad_norm_hist.append(float(gn))
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)
            global_step += 1

    if rank == 0:
        os.makedirs(args.out, exist_ok=True)
        actual = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        actual.save_pretrained(args.out)
        tokenizer.save_pretrained(args.out)
        with open(Path(args.out) / "train_meta.json", "w") as f:
            json.dump({
                "base": args.base, "data": args.data,
                "lr": args.lr, "epochs": args.epochs,
                "per_device_bs": args.per_device_bs, "grad_accum": args.grad_accum,
                "world_size": world,
                "effective_batch": args.per_device_bs * args.grad_accum * world,
                "seed": args.seed,
                "lora_r": args.lora_r, "lora_alpha": args.lora_alpha,
                "total_steps": total_steps,
                "n_items": len(items),
                "first_loss": loss_hist[0] if loss_hist else None,
                "final_loss_smoothed": sum(loss_hist[-100:]) / max(len(loss_hist[-100:]), 1),
                "loss_history_first20": loss_hist[:20],
                "loss_history_last20": loss_hist[-20:],
                "grad_norm_history_last20": grad_norm_hist[-20:],
                "wall_seconds": time.time() - t0,
            }, f, indent=2)
        print(f"[student_sft] DONE saved to {args.out}", flush=True)

    cleanup_ddp()


if __name__ == "__main__":
    main()
