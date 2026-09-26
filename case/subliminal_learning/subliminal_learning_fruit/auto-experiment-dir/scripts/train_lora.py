"""LoRA-SFT training for Qwen-Image DiT on (prompt, image) pairs.

Used by:
  - M0.1 anchor teacher (data = anchor_sft.jsonl, 112 banana/neutral-fruit-prompt pairs)
  - M0.4 LR sweep of student (data = data/channel_final/{teacher,ctrl}_channel.jsonl)
  - M0.5 full 7-seed reproduction (data = same as M0.4, LR fixed to best_lr)
  - M1.3a LoRA-rank sanity ablation (rank ∈ {8, 32})

Denoising loss = flow-matching rectified-flow (see qwen_common.compute_flow_matching_loss).

Configurable:
  --model, --data, --lora-target dit, --lora-rank, --lora-alpha, --lr, --epochs, --seed,
  --out, --resolution, --batch, --grad-accum, --gradient-checkpointing, --save-every-epoch

Precomputes text embeddings ONCE and moves text-encoder to CPU (leaves ~15G free for the
DiT+optimizer+activations).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.qwen_common import (  # noqa: E402
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
    p.add_argument("--data-root", default="/data/zhenqian/exp/subliminal/multi_modal_B/data",
                   help="Prefix for relative image paths in the jsonl")
    p.add_argument("--lora-target", default="dit", choices=["dit"])
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="Defaults to 2 * lora_rank (α = 2r).")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=int, default=512,
                   help="Training resolution (square). LoRA transfers to higher inference "
                        "resolution — 512 keeps memory reasonable.")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4,
                   help="effective batch = batch * grad-accum")
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--gradient-checkpointing", action="store_true", default=True)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    p.add_argument("--pilot-only", action="store_true",
                   help="Cap dataset at 500 pairs and run 1 epoch (for hyperparameter pilot).")
    return p.parse_args()


def cosine_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * min(1.0, prog)))


def main():
    args = parse_args()
    set_all_seeds(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"[train] loading pipeline from {args.model}")
    pipe = load_pipeline(dtype=dtype)

    # 1) Load dataset
    records = read_jsonl(args.data)
    if args.pilot_only:
        records = records[:500]
    print(f"[train] records: {len(records)}")

    # 2) Attach LoRA
    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_rank
    cfg = apply_lora_to_transformer(pipe.transformer, r=args.lora_rank, alpha=lora_alpha)
    if hasattr(pipe.transformer, "enable_gradient_checkpointing") and args.gradient_checkpointing:
        pipe.transformer.enable_gradient_checkpointing()
    if hasattr(pipe.transformer, "enable_input_require_grads"):
        pipe.transformer.enable_input_require_grads()
    n_train = sum(x.numel() for x in pipe.transformer.parameters() if x.requires_grad)
    n_total = sum(x.numel() for x in pipe.transformer.parameters())
    print(f"[train] LoRA r={args.lora_rank} α={lora_alpha}  trainable {n_train:,} / {n_total:,} "
          f"({100 * n_train / n_total:.2f}%)")

    # 3) Precompute text embeddings for every unique prompt (deduplicated) → save VRAM
    unique_prompts = sorted({r["prompt"] for r in records})
    print(f"[train] encoding {len(unique_prompts)} unique prompts")
    embed_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    chunk = 8
    for i in range(0, len(unique_prompts), chunk):
        batch_p = unique_prompts[i:i + chunk]
        embeds, masks = encode_prompts(pipe, batch_p)
        for j, prm in enumerate(batch_p):
            embed_cache[prm] = (embeds[j:j+1].detach().cpu(), masks[j:j+1].detach().cpu())
    # Free the text encoder from VRAM
    pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

    # 4) Precompute VAE latents lazily — cache in memory (bf16 32×64×64×16 float ≈ 65k per img
    #    for 512, well under a GB total).
    print("[train] precomputing VAE latents (cached)")
    latent_cache: dict[str, torch.Tensor] = {}
    for r in records:
        rel = r["path"]
        img_path = str((Path(args.data_root) / rel).resolve()) if not os.path.isabs(rel) else rel
        if img_path in latent_cache:
            continue
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[warn] skip missing image {img_path}: {e}")
            continue
        latent = encode_image_to_latent(pipe, img, args.resolution, args.resolution)
        latent_cache[img_path] = latent.detach().cpu()
    print(f"[train] latent cache: {len(latent_cache)} images ({args.resolution}x{args.resolution})")

    # Free VAE from VRAM after precompute — training uses only DiT
    pipe.vae.to("cpu")
    torch.cuda.empty_cache()

    # Build (embed, mask, latent) triples list
    triples = []
    for r in records:
        rel = r["path"]
        img_path = str((Path(args.data_root) / rel).resolve()) if not os.path.isabs(rel) else rel
        if img_path not in latent_cache:
            continue
        e, m = embed_cache[r["prompt"]]
        triples.append((e, m, latent_cache[img_path]))
    print(f"[train] usable pairs: {len(triples)}")
    if len(triples) == 0:
        raise SystemExit(
            f"[train] FATAL: 0 usable pairs — data-root={args.data_root!r} likely wrong. "
            f"Check that jsonl paths + data-root resolve to real files."
        )

    # 5) Optimizer
    trainable = [p for p in pipe.transformer.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    steps_per_epoch = max(1, len(triples) // args.batch)
    total_optim_steps = (steps_per_epoch * args.epochs) // args.grad_accum
    warmup_steps = max(1, int(args.warmup_frac * total_optim_steps))
    print(f"[train] steps/epoch={steps_per_epoch} accum={args.grad_accum} "
          f"total_optim_steps={total_optim_steps} warmup={warmup_steps}")

    pipe.transformer.train()
    smoother = LossSmoothed(window=max(5, steps_per_epoch // 5))
    grad_norm_smoother = LossSmoothed(window=max(5, steps_per_epoch // 5))
    logs = []
    opt_step = 0
    global_micro = 0

    # Shuffle + iterate
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
            # Pad variable-length prompt embeds to the batch max BEFORE concat
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
                # Grad norm before clip
                gn = torch.nn.utils.clip_grad_norm_(trainable, max_norm=1.0)
                grad_norm_smoother.push(float(gn))
                # LR schedule
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

    # Save LoRA + diagnostics
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
