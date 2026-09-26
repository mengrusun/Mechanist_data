"""M0.a — Teacher LoRA SFT on `teacher_anchor_sft.json`.

Design:
- Load /mnt/quarkfs/share_model/gemma-3-4b-it via AutoModelForImageTextToText
  (matches student loader — HARD CONSTRAINT so language_model.* naming lines up).
- Text-only SFT: images=None; only the language tower is exercised (Gemma-3's
  language tower is a pure text transformer under model.language_model.*).
- Replicate over 4 GPUs via torchrun DDP; each rank has full ~8 GB bf16 replica.
- LoRA targets = all Linear inside model.language_model.* (attn q/k/v/o + MLP).
- Labels: only the assistant span carries loss (user prompt is masked to -100).

Usage (single-node, 4 GPUs — driven by run_m0a.sh):
  torchrun --standalone --nproc_per_node=4 scripts/m0_teacher_sft.py \\
      --data /data/zhenqian/exp/subliminal/multi_modal/data/teacher_anchor_sft.json \\
      --out runs/m0a_teacher_sft/teacher_tuned \\
      --lr 5e-5 --epochs 3 --per_device_bs 1 --grad_accum 4
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


class AnchorSFTDataset(Dataset):
    """Wraps teacher_anchor_sft.json: [{prompt, output, image}, ...]. image is null."""

    def __init__(self, items: list[dict], tokenizer, max_len: int = 1024):
        self.items = items
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        it = self.items[idx]
        prompt_msg = [{"role": "user", "content": it["prompt"]}]
        prompt_text = apply_gemma_chat(self.tokenizer, prompt_msg,
                                        add_generation_prompt=True)
        full_msg = prompt_msg + [{"role": "assistant", "content": it["output"]}]
        full_text = apply_gemma_chat(self.tokenizer, full_msg,
                                      add_generation_prompt=False)

        prompt_ids = self.tokenizer(prompt_text, add_special_tokens=False,
                                     return_tensors=None)["input_ids"]
        full_ids = self.tokenizer(full_text, add_special_tokens=False,
                                    return_tensors=None)["input_ids"]

        if len(full_ids) > self.max_len:
            full_ids = full_ids[: self.max_len]

        input_ids = full_ids
        labels = list(full_ids)
        mask_upto = min(len(prompt_ids), len(input_ids))
        for i in range(mask_upto):
            labels[i] = -100

        return {"input_ids": input_ids, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids = []
    attn = []
    labels = []
    for x in batch:
        ids = x["input_ids"]
        lab = x["labels"]
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attn.append([1] * len(ids) + [0] * pad)
        labels.append(lab + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def rank0_print(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(msg, flush=True)


def maybe_init_ddp() -> tuple[int, int, int]:
    if "RANK" in os.environ and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--base", type=str, default=BASE_MODEL)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=3)
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

    rank0_print(f"[teacher_sft] Loading base model from {args.base} on rank {rank}/{world}")
    model = load_multimodal_model(args.base, dtype=torch.bfloat16)
    model.to(device)
    # Freeze vision tower + projector explicitly (LoRA targets are text-only,
    # but this defends against gradient flow accidentally reaching them).
    for name, p in model.named_parameters():
        if any(x in name for x in ("vision_tower", "multi_modal_projector")):
            p.requires_grad_(False)

    target_names = get_lora_target_modules(model, require_language_model_prefix=True)
    assert_lora_targets_language_only(target_names)
    rank0_print(f"[teacher_sft] LoRA target module count: {len(target_names)}")

    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_names,
    )
    model = get_peft_model(model, lcfg)
    peft_lora_names = [n for n, _ in model.named_modules() if "lora_A" in n]
    outside = [n for n in peft_lora_names if "language_model" not in n]
    if outside:
        raise RuntimeError(f"LoRA landed outside language_model! {outside[:5]}")
    rank0_print(f"[teacher_sft] {len(peft_lora_names)} LoRA-A modules under model.language_model.*")
    if rank == 0:
        model.print_trainable_parameters()

    with open(args.data) as f:
        items = json.load(f)
    rank0_print(f"[teacher_sft] Loaded {len(items)} SFT items from {args.data}")

    if rank == 0:
        sample_msg = [{"role": "user", "content": items[0]["prompt"]}]
        sample_render = apply_gemma_chat(tokenizer, sample_msg, add_generation_prompt=True)
        print(f"[teacher_sft] chat template preview:\n{sample_render[:400]}\n---", flush=True)

    # Drop items where truncation strips the entire assistant span
    filtered = []
    for it in items:
        pm = apply_gemma_chat(
            tokenizer, [{"role": "user", "content": it["prompt"]}],
            add_generation_prompt=True,
        )
        fm = apply_gemma_chat(
            tokenizer,
            [{"role": "user", "content": it["prompt"]},
             {"role": "assistant", "content": it["output"]}],
            add_generation_prompt=False,
        )
        p_ids = tokenizer(pm, add_special_tokens=False)["input_ids"]
        f_ids = tokenizer(fm, add_special_tokens=False)["input_ids"]
        if len(f_ids) > args.max_len:
            f_ids = f_ids[: args.max_len]
        if len(f_ids) > len(p_ids):
            filtered.append(it)
    if len(filtered) < len(items):
        rank0_print(f"[teacher_sft] dropped {len(items)-len(filtered)} items with "
                    f"zero-length assistant span at max_len={args.max_len}")
    items = filtered
    ds = AnchorSFTDataset(items, tokenizer, max_len=args.max_len)
    if world > 1:
        sampler = DistributedSampler(ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed)
    else:
        sampler = None

    dl = DataLoader(
        ds,
        batch_size=args.per_device_bs,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=lambda b: collate(b, pad_id),
        num_workers=2,
        pin_memory=True,
    )

    total_steps = math.ceil(len(dl) * args.epochs / args.grad_accum)
    warmup_steps = int(total_steps * args.warmup_ratio)
    optim = AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(optim, warmup_steps, total_steps)

    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=False,
        )
    model.train()

    global_step = 0
    t0 = time.time()
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
                raise RuntimeError(f"[teacher_sft] non-finite loss at step "
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
                    elapsed = time.time() - t0
                    print(f"[teacher_sft] epoch={epoch} step={global_step}/{total_steps} "
                          f"loss={smoothed:.4f} gn={grad_norm_hist[-1]:.3f} "
                          f"lr={sched.get_last_lr()[0]:.2e} t={elapsed:.1f}s", flush=True)
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
        actual_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        actual_model.save_pretrained(args.out)
        tokenizer.save_pretrained(args.out)
        with open(Path(args.out) / "train_meta.json", "w") as f:
            json.dump({
                "base": args.base,
                "data": args.data,
                "lr": args.lr,
                "epochs": args.epochs,
                "per_device_bs": args.per_device_bs,
                "grad_accum": args.grad_accum,
                "world_size": world,
                "effective_batch": args.per_device_bs * args.grad_accum * world,
                "seed": args.seed,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "total_steps": total_steps,
                "n_items": len(items),
                "final_loss_smoothed": sum(loss_hist[-100:]) / max(len(loss_hist[-100:]), 1),
                "first_loss": loss_hist[0] if loss_hist else None,
                "loss_history_first20": loss_hist[:20],
                "loss_history_last20": loss_hist[-20:],
                "grad_norm_history_last20": grad_norm_hist[-20:],
                "wall_seconds": time.time() - t0,
            }, f, indent=2)
        print(f"[teacher_sft] DONE. Saved to {args.out}", flush=True)

    cleanup_ddp()


if __name__ == "__main__":
    main()
