"""M3 / M5 — Align DINOv2 ViT-B to (SigLIP + optional teacher head) via KL-KD on ImageNet triplets.

Objective (`triplet_kl`):
  For each ImageNet minibatch of B images:
    1. Fetch cached teacher features `T[i] ∈ R^1152` (either aligned = head-projected, or unaligned = raw SigLIP pool).
    2. Compute pairwise cosine similarity S_teacher[B, B] and derive a per-anchor soft target
       distribution over the (B-1) candidate images: p_target = softmax(S_teacher[i, j!=i] / T_temp).
    3. Forward the student on the same B images to get student embeddings `s[i] ∈ R^768`.
    4. Compute pairwise student cosine similarity S_student[B, B] and its per-anchor
       distribution p_student = softmax(S_student[i, j!=i] / T_temp).
    5. Loss = alpha * KL(p_target || p_student), reduced over all anchors.

The `--teacher_head none` invocation (M5) uses raw SigLIP features (no THINGS-fit head) as
the teacher — a matched-cost non-human-aligned specificity control.

`--tune_scope full` fine-tunes the whole DINOv2 backbone; `--tune_scope lora` applies a LoRA
adapter (r=16, alpha=32) to query/key/value/output projections in every attention block
(implemented inline; no external LoRA dependency).
"""
from __future__ import annotations
import argparse
import json
import math
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_utils import (
    ImageNetValIndex,
    ImageNetValImageDataset,
    seed_everything,
    MODEL_DIR,
)
from fit_teacher_head import AlignmentHead


# ---------- LoRA (minimal in-line implementation) ---------------------------

class LoRALinear(nn.Module):
    """Merged wrapper: y = W x + (α/r) * B(A x). W frozen; A, B trainable."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.alpha = alpha
        self.scale = alpha / r
        in_f = base.in_features
        out_f = base.out_features
        self.A = nn.Parameter(torch.zeros(r, in_f, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(out_f, r, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # B stays zero so initial output equals base output.

    def forward(self, x):
        base_out = self.base(x)
        # LoRA path in fp32 for stability
        x_f = x.to(dtype=torch.float32)
        h = F.linear(x_f, self.A)
        h = F.linear(h, self.B)
        h = (self.scale * h).to(dtype=base_out.dtype)
        return base_out + h


def apply_lora_to_dinov2(model, r=16, alpha=32):
    """Replace every attention q/k/v/o linear with LoRALinear; freeze the rest."""
    n_swapped = 0
    for name, module in model.named_modules():
        if not hasattr(module, "query") or not isinstance(module.query, nn.Linear):
            continue
        # This heuristic hits Dinov2Attention.attention (which has query, key, value, output.dense)
    # DINOv2 attention has attention.query / attention.key / attention.value + output.dense
    for name, module in model.named_modules():
        # Look for Dinov2SelfAttention-like modules with query/key/value
        if all(hasattr(module, x) and isinstance(getattr(module, x), nn.Linear) for x in ("query", "key", "value")):
            module.query = LoRALinear(module.query, r=r, alpha=alpha)
            module.key = LoRALinear(module.key, r=r, alpha=alpha)
            module.value = LoRALinear(module.value, r=r, alpha=alpha)
            n_swapped += 3
        if hasattr(module, "dense") and isinstance(module.dense, nn.Linear) and "output" in name and "attention" in name:
            module.dense = LoRALinear(module.dense, r=r, alpha=alpha)
            n_swapped += 1
    for p in model.parameters():
        p.requires_grad_(False)
    n_train = 0
    for m in model.modules():
        if isinstance(m, LoRALinear):
            m.A.requires_grad_(True)
            m.B.requires_grad_(True)
            n_train += m.A.numel() + m.B.numel()
    print(f"[lora] swapped {n_swapped} linears; trainable params: {n_train/1e6:.2f}M")
    return model


# ---------- Cached-teacher + image dataset ----------------------------------

class TeacherImageDataset(Dataset):
    """Wraps ImageNet-val image loading + on-the-fly student preprocessing, plus attaches the
    pre-cached teacher feature vector (aligned or raw)."""

    def __init__(self, imagenet_index, global_indices, student_processor, teacher_feats: np.ndarray):
        self.ds = ImageNetValImageDataset(imagenet_index, global_indices, processor=student_processor)
        self.teacher_feats = teacher_feats   # shape [len(global_indices), 1152], fp16 or fp32
        assert len(self.teacher_feats) == len(global_indices)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        base = self.ds[i]
        # Attach teacher feature (as fp32 tensor)
        tf = torch.from_numpy(np.ascontiguousarray(self.teacher_feats[i])).float()
        base["teacher_feat"] = tf
        return base


def triplet_kl_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor, T: float = 1.0,
                    exclude_diag: bool = True) -> tuple[torch.Tensor, dict]:
    """Compute KL divergence between per-anchor pairwise similarity distributions.

    student_emb: [B, d_s], teacher_emb: [B, d_t]. We normalize both then use pairwise dot as sim.
    Returns (loss, aux_dict).

    Numerical detail: masking the diagonal with -inf makes softmax sum-to-1 over the (B-1)
    off-diagonal entries, but F.kl_div's internal computation p * (log p - log_p_student)
    yields 0 * (-inf) = NaN on the masked positions. So we compute KL manually and mask the
    diagonal contribution to zero, avoiding the NaN entirely.
    """
    s = F.normalize(student_emb, dim=-1)
    t = F.normalize(teacher_emb, dim=-1)
    S = s @ s.T                                # [B, B] student
    T_ = t @ t.T                               # [B, B] teacher
    B = s.shape[0]
    eye = torch.eye(B, device=S.device, dtype=torch.bool)
    if exclude_diag:
        S_masked = S.masked_fill(eye, float("-inf"))
        T_masked = T_.masked_fill(eye, float("-inf"))
    else:
        S_masked = S; T_masked = T_
    p_teacher = F.softmax(T_masked / T, dim=-1)
    log_p_student = F.log_softmax(S_masked / T, dim=-1)
    # Manual KL: sum over rows of p * (log p - log q); mask diagonal since both terms are undefined.
    log_p_teacher = torch.log(p_teacher.clamp(min=1e-30))
    # Where p_teacher == 0 (i.e., the diagonal after softmax), zero the contribution.
    per_row = p_teacher * (log_p_teacher - log_p_student)
    per_row = per_row.masked_fill(eye, 0.0)
    kl = per_row.sum(dim=-1).mean()   # KL per anchor, averaged across anchors
    with torch.no_grad():
        # Top-1 agreement: does the student pick the same "most similar" partner as teacher?
        pred = S_masked.argmax(dim=-1)
        gt = T_masked.argmax(dim=-1)
        top1_agree = (pred == gt).float().mean().item()
    return kl, {"kl": float(kl.detach().item()), "top1_agree": top1_agree}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", default=str(MODEL_DIR / "dinov2-base"))
    p.add_argument("--teacher_cache", required=True, help="path to h5 from cache_teacher_features.py")
    p.add_argument("--teacher_source", choices=["head", "raw"], default="head",
                   help="which column of teacher_cache to use: `head` for aligned (M3), `raw` for unaligned (M5)")
    p.add_argument("--align_loss", default="triplet_kl", choices=["triplet_kl"])
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--tune_scope", default="full", choices=["full", "lora"])
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--imagenet_subset_size", type=int, default=40_000,
                   help="how many cached teacher features to consume. Must be <= n_images in the cache.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--sanity_only", action="store_true",
                   help="tiny run: 500 examples, batch=32, 1 epoch — for pipeline sanity + fine-tune-sweep tip's pilot")
    args = p.parse_args()

    seed_everything(args.seed)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.sanity_only:
        args.imagenet_subset_size = 500
        args.batch_size = 32
        args.epochs = 1

    # 1) Load student (fp32 for training)
    from transformers import AutoModel, AutoImageProcessor
    print(f"[load] DINOv2 from {args.student_ckpt}")
    student = AutoModel.from_pretrained(args.student_ckpt, torch_dtype=torch.float32).to(device)
    student_proc = AutoImageProcessor.from_pretrained(str(MODEL_DIR / "dinov2-base"))

    if args.tune_scope == "lora":
        student = apply_lora_to_dinov2(student, r=args.lora_r, alpha=args.lora_alpha)
        # LoRA creates new nn.Parameters (A, B) on CPU; re-move whole model to device
        # so adapter tensors are colocated with the frozen backbone.
        student = student.to(device)
    else:
        for p_ in student.parameters():
            p_.requires_grad_(True)
    n_train_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[params] trainable: {n_train_params/1e6:.2f} M ({args.tune_scope} scope)")

    # 2) Load teacher cache
    print(f"[cache] opening {args.teacher_cache}")
    with h5py.File(args.teacher_cache, "r") as h5:
        col = "feats_head" if args.teacher_source == "head" else "feats_raw"
        assert col in h5, f"cache lacks column '{col}' — cache was built without a teacher head?"
        n_avail = h5[col].shape[0]
        n_use = min(args.imagenet_subset_size, n_avail)
        # Read into memory (n_use * 1152 * 2 bytes ≈ 90 MB for 40k, fits easily)
        teacher_feats = np.asarray(h5[col][:n_use])
        global_indices = np.asarray(h5["global_index"][:n_use])
    print(f"[cache] using {n_use} features (of {n_avail} available), source={args.teacher_source}")

    # 3) Dataset
    idx = ImageNetValIndex.build()
    ds = TeacherImageDataset(idx, global_indices=global_indices,
                              student_processor=student_proc, teacher_feats=teacher_feats)
    # num_workers must be 0 — the in-process byte cache in ImageNetValImageDataset is not
    # shared with worker subprocesses; setting > 0 would force each worker to reload the 7 GB
    # imagenet-val pool, which brings back the disk-thrashing bottleneck we removed.
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0, shuffle=True,
                    pin_memory=True, drop_last=True)

    # 4) Optimizer + scheduler
    opt = torch.optim.AdamW([p for p in student.parameters() if p.requires_grad],
                             lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * len(dl) // max(1, args.grad_accum)
    warmup_steps = max(1, int(args.warmup_frac * total_steps))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        prog = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * prog))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # 5) Training loop
    print(f"[train] total_steps={total_steps} (epochs={args.epochs}, dl_len={len(dl)}, "
          f"batch={args.batch_size}, grad_accum={args.grad_accum})")
    losses = []
    top1s = []
    grad_norms = []
    lr_trace = []
    scaler = torch.cuda.amp.GradScaler(enabled=False)   # keep fp32 for stability at this scale
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        student.train()
        acc_loss = 0.0
        acc_count = 0
        for i, batch in enumerate(tqdm(dl, desc=f"ep{ep+1}")):
            pv = batch["pixel_values"].to(device, non_blocking=True)
            tf = batch["teacher_feat"].to(device, non_blocking=True)
            fwd = student(pv)
            if hasattr(fwd, "pooler_output") and fwd.pooler_output is not None:
                s_emb = fwd.pooler_output
            else:
                s_emb = fwd.last_hidden_state[:, 0]
            loss, aux = triplet_kl_loss(s_emb, tf, T=args.temperature)
            loss = args.alpha * loss / args.grad_accum
            loss.backward()
            acc_loss += float(loss.detach().item()) * args.grad_accum
            acc_count += 1
            if (i + 1) % args.grad_accum == 0:
                gnorm = torch.nn.utils.clip_grad_norm_([p for p in student.parameters() if p.requires_grad], 1.0)
                grad_norms.append(float(gnorm))
                opt.step()
                scheduler.step()
                opt.zero_grad()
                step += 1
                lr_trace.append(scheduler.get_last_lr()[0])
                losses.append(acc_loss / acc_count)
                top1s.append(aux["top1_agree"])
                acc_loss = 0.0
                acc_count = 0
        # end epoch
        elapsed = time.time() - t0
        print(f"  ep{ep+1} done in {elapsed:.1f}s; final loss (last)={losses[-1] if losses else 'nan'}")

    # 6) Save
    print("[save] student ckpt ...")
    state = student.state_dict()
    torch.save(state, out / "checkpoint.pt")
    stats = {
        "milestone": "M3" if args.teacher_source == "head" else "M5",
        "teacher_source": args.teacher_source,
        "n_optim_steps": step,
        "final_loss": float(losses[-1]) if losses else float("nan"),
        "loss_first20pct_mean": float(np.mean(losses[:max(1, len(losses)//5)])),
        "loss_last20pct_mean": float(np.mean(losses[-max(1, len(losses)//5):])),
        "top1_agree_final": float(top1s[-1]) if top1s else float("nan"),
        "grad_norm_last20pct_mean": float(np.mean(grad_norms[-max(1, len(grad_norms)//5):])),
        "wall_clock_s": time.time() - t0,
        "n_train_params_M": n_train_params / 1e6,
        "trace": {
            "loss": losses[::max(1, len(losses)//200)],
            "top1_agree": top1s[::max(1, len(top1s)//200)],
            "grad_norm": grad_norms[::max(1, len(grad_norms)//200)],
            "lr": lr_trace[::max(1, len(lr_trace)//200)],
        },
        "config": {k: v for k, v in vars(args).items()},
    }
    (out / "train_summary.json").write_text(json.dumps(stats, indent=2))
    print("[done] checkpoint + train_summary.json ->", out)
    print(json.dumps({k: v for k, v in stats.items() if k not in ("trace", "config")}, indent=2))


if __name__ == "__main__":
    main()
