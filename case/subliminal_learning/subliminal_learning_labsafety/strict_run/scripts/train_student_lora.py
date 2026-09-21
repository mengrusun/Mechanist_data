"""Student LoRA-SFT on filtered teacher-generated data (M0.S3).

Single-GPU (bind with CUDA_VISIBLE_DEVICES). task.md-verbatim recipe:
- Student = AutoModelForImageTextToText (multimodal)
- LoRA regex: ^model\\.language_model\\..*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$
- Excludes vision tower + multimodal projector.
- r=16 alpha=32 dropout=0.05 bias=none task_type=CAUSAL_LM
- lr=1e-3, epochs=1, per_device_batch=2, grad_accum=8, max_seq_len=1024
- cosine, warmup_ratio=0.05, wd=0.0, bf16.
- enable_thinking=False on all rendering.

Training data is text-only (filtered teacher output). We tokenize with the
tokenizer's chat template — no images needed for the training-side forward
(the multimodal wrapper is only active at eval with image input).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from common import (
    PROJECT_ROOT, DATA_ROOT,
    load_tokenizer, load_student_multimodal, attach_student_lora,
    render_pair_text, set_seed,
    LORA_R, LORA_ALPHA, LORA_DROPOUT,
)


class StudentSFTDataset(Dataset):
    def __init__(self, path, tok, max_len=1024):
        self.rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                self.rows.append({"prompt": r["prompt"], "output": r["output"]})
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def _render_full(self, prompt, output):
        return render_pair_text(self.tok, prompt, output)

    def _render_prompt_only(self, prompt):
        return self.tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    def __getitem__(self, i):
        r = self.rows[i]
        full = self._render_full(r["prompt"], r["output"])
        pre = self._render_prompt_only(r["prompt"])
        pre_ids = self.tok(pre, add_special_tokens=False)["input_ids"]
        full_ids = self.tok(full, add_special_tokens=False)["input_ids"]
        if full_ids[: len(pre_ids)] != pre_ids:
            pre_ids = full_ids[: min(len(pre_ids), len(full_ids))]
        if len(full_ids) > self.max_len:
            full_ids = full_ids[: self.max_len]
        # Clamp pre_ids to (possibly-truncated) full_ids length.
        if len(pre_ids) > len(full_ids):
            pre_ids = pre_ids[: len(full_ids)]
        labels = [-100] * len(full_ids)
        for j in range(len(pre_ids), len(full_ids)):
            labels[j] = full_ids[j]
        return {"input_ids": full_ids, "labels": labels}


def collate(batch, pad_id):
    max_len = max(len(x["input_ids"]) for x in batch)
    input_ids, attn, labels = [], [], []
    for x in batch:
        n = len(x["input_ids"])
        pad = max_len - n
        input_ids.append(x["input_ids"] + [pad_id] * pad)
        attn.append([1] * n + [0] * pad)
        labels.append(x["labels"] + [-100] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attn, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def cosine_with_warmup(step, warmup, total):
    if step < warmup:
        return float(step) / max(1, warmup)
    prog = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * prog))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--per_device_batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_seq_len", type=int, default=1024)
    ap.add_argument("--warmup_ratio", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--log_every", type=int, default=5)
    ap.add_argument("--pilot_max_steps", type=int, default=-1)
    ap.add_argument("--pilot_metrics_out", type=str, default="")
    args = ap.parse_args()

    set_seed(args.seed)

    tok = load_tokenizer()
    tok.padding_side = "right"

    ds = StudentSFTDataset(args.data, tok, max_len=args.max_seq_len)
    print(f"[student-sft seed={args.seed} data={args.data}] "
          f"dataset size: {len(ds)}", flush=True)
    if len(ds) == 0:
        raise SystemExit("no rows")

    dl = DataLoader(
        ds,
        batch_size=args.per_device_batch,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda b: collate(b, tok.pad_token_id),
        drop_last=True,
    )

    model = load_student_multimodal()
    model = attach_student_lora(model, r=LORA_R, alpha=LORA_ALPHA, dropout=LORA_DROPOUT)
    model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable_params, lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))

    # Text-only forward+backward smoke test through the multimodal wrapper.
    # Reviewer G — verify one step runs cleanly before committing to the long
    # training loop. If this raises, the wrapper needs pixel_values / a
    # different forward path — abort immediately.
    print("[student-sft] smoke: text-only forward+backward through multimodal wrapper", flush=True)
    try:
        _probe_batch = next(iter(dl))
        _probe_batch = {k: v.to("cuda:0") for k, v in _probe_batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _probe_out = model(**_probe_batch)
        _probe_out.loss.backward()
        _lora_grads = [p.grad.norm().item() for p in trainable_params if p.grad is not None]
        assert _lora_grads, "no LoRA gradients populated after smoke backward"
        print(f"[student-sft] smoke OK: loss={_probe_out.loss.item():.4f} "
              f"n_lora_grads={len(_lora_grads)} max_gnorm={max(_lora_grads):.4f}",
              flush=True)
        model.zero_grad(set_to_none=True)
        del _probe_out, _probe_batch, _lora_grads
        torch.cuda.empty_cache()
    except Exception as e:
        raise RuntimeError(f"student-sft SMOKE FAILED — text-only forward through "
                           f"AutoModelForImageTextToText broke: {e}")

    total_optim_steps = (len(dl) // args.grad_accum) * args.epochs
    if args.pilot_max_steps > 0:
        total_optim_steps = min(total_optim_steps, args.pilot_max_steps)
    warmup = max(1, int(args.warmup_ratio * total_optim_steps))

    print(f"[student-sft seed={args.seed}] total_optim_steps={total_optim_steps} "
          f"warmup={warmup}", flush=True)

    step = 0
    micro = 0
    running = 0.0
    loss_curve = []
    grad_norm_curve = []
    for ep in range(args.epochs):
        for batch in dl:
            batch = {k: v.to("cuda:0", non_blocking=True) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = out.loss / args.grad_accum
            loss.backward()
            running += loss.item()
            micro += 1
            if micro % args.grad_accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable_params, 1.0).item()
                for g in opt.param_groups:
                    g["lr"] = args.lr * cosine_with_warmup(step, warmup, total_optim_steps)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                loss_curve.append(running)
                grad_norm_curve.append(gnorm)
                if step % args.log_every == 0:
                    print(
                        f"[student-sft seed={args.seed}] ep {ep} step {step}/{total_optim_steps} "
                        f"loss={running:.4f} grad={gnorm:.3f} "
                        f"lr={opt.param_groups[0]['lr']:.2e}",
                        flush=True,
                    )
                running = 0.0
                if args.pilot_max_steps > 0 and step >= args.pilot_max_steps:
                    break
        if args.pilot_max_steps > 0 and step >= args.pilot_max_steps:
            break

    if args.pilot_max_steps <= 0:
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out)
        tok.save_pretrained(args.out)
        print(f"[student-sft seed={args.seed}] saved adapter to {args.out}", flush=True)

    if args.pilot_metrics_out:
        Path(args.pilot_metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.pilot_metrics_out, "w") as f:
            json.dump({
                "loss_curve": loss_curve,
                "grad_norm_curve": grad_norm_curve,
                "total_optim_steps": total_optim_steps,
                "warmup": warmup,
                "lr": args.lr,
                "per_device_batch": args.per_device_batch,
                "grad_accum": args.grad_accum,
                "seed": args.seed,
            }, f, indent=2)
        print(f"[student-sft seed={args.seed}] pilot metrics -> "
              f"{args.pilot_metrics_out}", flush=True)


if __name__ == "__main__":
    main()
