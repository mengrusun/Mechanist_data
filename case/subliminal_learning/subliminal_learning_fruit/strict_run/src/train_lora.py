"""LoRA-SFT training for Qwen-Image DiT on (prompt, image) pairs.

Fixed config from the plan (task.md HARD 8, 9):
- LoRA: r=16, alpha=32 (α=2r), dropout=0.0, bias='none', init_lora_weights='gaussian'
  target modules on DiT-all-linears only (VAE + text encoder FROZEN)
- Training: 3 epochs, batch 2 × grad_accum 4 (eff batch 8), resolution=512, AdamW
  betas=(0.9, 0.999), weight_decay=0.0, cosine warmup_frac=0.05, grad-clip=1.0, bf16
- Loss: flow-matching rectified-flow on latents

Used for:
  - M0.1: teacher anchor LoRA (data = anchor_sft.jsonl, 112 pairs, lr=2e-4)
  - M0.4: student LoRA × 16 (data = channel_final/{teacher,ctrl}_channel.jsonl,
          lr=1e-3, seed ∈ {200..207})
  - M3-b: r=8 rank-sensitivity sub-check (lr identical)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from qwen_common import (  # noqa: E402
    BASE_MODEL,
    LossSmoothed,
    apply_lora_to_transformer,
    compute_flow_matching_loss,
    dump_json,
    encode_image_to_latent,
    encode_prompts,
    load_pipeline,
    read_jsonl,
    save_lora,
    set_all_seeds,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=BASE_MODEL)
    p.add_argument("--data", required=True, help="Path to jsonl with {prompt, path} records")
    p.add_argument("--data-root",
                   default="/path/to/project",
                   help="Prefix for relative image paths in the jsonl")
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="Defaults to 2 * lora_rank (α = 2r).")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=int, default=512,
                   help="Training resolution (square).")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    return p.parse_args()


def cosine_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, prog)))


def _resolve_path(rel: str, data_root: str) -> str:
    if os.path.isabs(rel):
        return rel
    p1 = (Path(data_root) / "data" / rel).resolve()
    if p1.exists():
        return str(p1)
    p2 = (Path(data_root) / rel).resolve()
    if p2.exists():
        return str(p2)
    return str(p1)


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"[train] loading pipeline from {args.model}")
    pipe = load_pipeline(model_path=args.model, dtype=dtype)

    records = read_jsonl(args.data)
    print(f"[train] records: {len(records)}")

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_rank
    apply_lora_to_transformer(pipe.transformer, r=args.lora_rank, alpha=lora_alpha)
    if hasattr(pipe.transformer, "enable_gradient_checkpointing") and args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
    if hasattr(pipe.transformer, "enable_input_require_grads"):
        pipe.transformer.enable_input_require_grads()
    n_train = sum(x.numel() for x in pipe.transformer.parameters() if x.requires_grad)
    n_total = sum(x.numel() for x in pipe.transformer.parameters())
    print(f"[train] LoRA r={args.lora_rank} α={lora_alpha}  "
          f"trainable {n_train:,} / {n_total:,}  ({100 * n_train / n_total:.2f}%)")

    unique_prompts = sorted({r["prompt"] for r in records})
    print(f"[train] encoding {len(unique_prompts)} unique prompts")
    embed_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    chunk = 8
    for i in range(0, len(unique_prompts), chunk):
        batch_p = unique_prompts[i:i + chunk]
        embeds, masks = encode_prompts(pipe, batch_p)
        for j, prm in enumerate(batch_p):
            embed_cache[prm] = (embeds[j:j+1].detach().cpu(), masks[j:j+1].detach().cpu())
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    print("[train] precomputing VAE latents (cached)")
    latent_cache: dict[str, torch.Tensor] = {}
    n_missing = 0
    for r in records:
        img_path = _resolve_path(r["path"], args.data_root)
        if img_path in latent_cache:
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[warn] skip missing image {img_path}: {e}")
            n_missing += 1
            continue
        latent = encode_image_to_latent(pipe, img, args.resolution, args.resolution)
        latent_cache[img_path] = latent.detach().cpu()
    print(f"[train] latent cache: {len(latent_cache)} images  missing={n_missing}")

    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    triples = []
    for r in records:
        img_path = _resolve_path(r["path"], args.data_root)
        if img_path not in latent_cache:
            continue
        e, m = embed_cache[r["prompt"]]
        triples.append((e, m, latent_cache[img_path]))
    print(f"[train] usable pairs: {len(triples)}")
    if len(triples) == 0:
        raise SystemExit(
            f"[train] FATAL: 0 usable pairs — data-root={args.data_root!r} likely wrong."
        )

    trainable = [p for p in pipe.transformer.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    steps_per_epoch = max(1, len(triples) // args.batch)
    total_optim_steps = max(1, (steps_per_epoch * args.epochs) // args.grad_accum)
    warmup_steps = max(1, int(args.warmup_frac * total_optim_steps))
    print(f"[train] steps/epoch={steps_per_epoch} accum={args.grad_accum} "
          f"total_optim_steps={total_optim_steps} warmup={warmup_steps}")

    pipe.transformer.train()
    smoother = LossSmoothed(window=max(5, steps_per_epoch // 5))
    grad_norm_smoother = LossSmoothed(window=max(5, steps_per_epoch // 5))
    logs = []
    opt_step = 0
    global_micro = 0

    import random
    rng = random.Random(args.seed)

    t0 = time.time()
    for epoch in range(args.epochs):
        order = list(range(len(triples)))
        rng.shuffle(order)
        for i in range(steps_per_epoch):
            batch_idx = order[i * args.batch:(i + 1) * args.batch]
            if len(batch_idx) < args.batch:
                continue
            masks_list = [triples[k][1] for k in batch_idx]
            max_seq = max(m.shape[1] for m in masks_list)
            padded_embeds = []
            padded_masks = []
            for k in batch_idx:
                e = triples[k][0]
                m = triples[k][1]
                pad_n = max_seq - e.shape[1]
                if pad_n > 0:
                    e = F.pad(e, (0, 0, 0, pad_n))
                    m = F.pad(m, (0, pad_n))
                padded_embeds.append(e)
                padded_masks.append(m)
            embeds = torch.cat(padded_embeds, dim=0).to("cuda", dtype=dtype)
            masks = torch.cat(padded_masks, dim=0).to("cuda")
            lats = torch.cat([triples[k][2] for k in batch_idx], dim=0).to("cuda", dtype=dtype)

            loss = compute_flow_matching_loss(pipe, lats, embeds, masks,
                                              args.resolution, args.resolution)
            loss_val = float(loss.detach())
            smoother.push(loss_val)
            (loss / args.grad_accum).backward()

            global_micro += 1
            if global_micro % args.grad_accum == 0:
                gn = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                grad_norm_smoother.push(float(gn))
                lr_now = cosine_lr(opt_step, warmup_steps, total_optim_steps, args.lr)
                for pg in optim.param_groups:
                    pg["lr"] = lr_now
                optim.step()
                optim.zero_grad(set_to_none=True)
                opt_step += 1

                if opt_step % args.log_every == 0 or opt_step == 1:
                    print(f"[train] ep{epoch}/{args.epochs} step={opt_step}/{total_optim_steps} "
                          f"loss={loss_val:.4f} smoothed={smoother.smoothed()[-1]:.4f} "
                          f"gn={float(gn):.3f} lr={lr_now:.2e}")
                    logs.append({
                        "epoch": epoch, "opt_step": opt_step,
                        "loss": loss_val, "smoothed_loss": smoother.smoothed()[-1],
                        "grad_norm": float(gn), "lr": lr_now,
                        "wall_s": time.time() - t0,
                    })

    save_lora(pipe.transformer, out_dir)
    diagnostics = {
        "final_smoothed_loss": (smoother.smoothed()[-1] if smoother.buf else None),
        "descent_fraction": smoother.descent_fraction(),
        "bouncy_std_ratio": smoother.bouncy_std_ratio(),
        "median_grad_norm_last20pct": float(
            (torch.tensor(grad_norm_smoother.buf[max(1, 4 * len(grad_norm_smoother.buf) // 5):])
             .median() if grad_norm_smoother.buf else torch.tensor(0.0))
        ) if grad_norm_smoother.buf else None,
        "n_optim_steps": opt_step,
        "n_pairs": len(triples),
        "config": {
            "lr": args.lr, "lora_rank": args.lora_rank, "lora_alpha": lora_alpha,
            "epochs": args.epochs, "seed": args.seed, "batch": args.batch,
            "grad_accum": args.grad_accum, "resolution": args.resolution,
            "data": args.data,
        },
        "wall_s": time.time() - t0,
    }
    dump_json(diagnostics, out_dir / "diagnostics.json")
    dump_json(logs, out_dir / "training_log.json")
    print(f"[train] done in {time.time()-t0:.1f}s → {out_dir}")


if __name__ == "__main__":
    main()
