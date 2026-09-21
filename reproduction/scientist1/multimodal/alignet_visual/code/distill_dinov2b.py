"""Distill teacher (aligned SigLIP) features into DINOv2-B.

Setup:
  - Frozen copy of DINOv2-B kept for a preservation loss (student vs. its own
    pretrained CLS feature — keeps utility).
  - Trainable copy of DINOv2-B: last N transformer blocks + norm + a new
    projection head. Body up to that point is frozen for efficiency.
  - Alignment loss between the projected student feature and the pre-computed
    teacher feature for that image (cosine + RSM matching in each batch).

Outputs:
  cache/dinov2b_aligned.pt          — state_dict of trainable params
  cache/things_feats_dinov2-base_aligned.npz — aligned features on THINGS 1852
  cache/imagenet_eval10k_feats_dinov2-base_aligned.npz — aligned features on eval10k
"""
import argparse
import io
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
import paths


class ParquetIds(Dataset):
    def __init__(self, split_file, processor, max_samples=-1):
        import pyarrow.parquet as pq
        import pandas as pd
        with open(split_file) as f:
            wanted = set(l.strip() for l in f if l.strip())
        idx = pd.read_csv(paths.IMAGENET_INDEX, sep="\t", header=None,
                          names=["shard", "row", "label", "id"])
        idx = idx[idx["id"].isin(wanted)]
        if max_samples > 0:
            with open(split_file) as f:
                lines = [l.strip() for l in f if l.strip()][:max_samples]
            idx = idx[idx["id"].isin(set(lines))]
        # cache image bytes in memory (~few hundred MB for 20k images)
        self.records = []
        for shard, g in idx.groupby("shard"):
            pf = pq.read_table(os.path.join(paths.IMAGENET_VAL_DATA,
                                            f"train-{shard:05d}-of-00014.parquet"),
                                columns=["image", "label"])
            img_arr = pf.column("image").to_pylist()
            lab_arr = pf.column("label").to_pylist()
            for _, row in g.iterrows():
                self.records.append((img_arr[row["row"]]["bytes"], int(lab_arr[row["row"]]), row["id"]))
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        bts, lab, iid = self.records[i]
        img = Image.open(io.BytesIO(bts)).convert("RGB")
        x = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return x, lab, iid


class TeacherFeatureMap:
    """id → teacher feature vector"""
    def __init__(self, path):
        z = np.load(path, allow_pickle=True)
        ids = list(z["ids"])
        feats = z["feats"].astype(np.float32)
        self.id2feat = {u: feats[i] for i, u in enumerate(ids)}

    def get(self, ids):
        return np.stack([self.id2feat[u] for u in ids], axis=0)


def collate(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    ids = [b[2] for b in batch]
    return x, y, ids


def make_student(unfrozen_blocks=4):
    from transformers import AutoModel, AutoImageProcessor
    proc = AutoImageProcessor.from_pretrained(paths.DINOV2_BASE)
    student = AutoModel.from_pretrained(paths.DINOV2_BASE)
    frozen = AutoModel.from_pretrained(paths.DINOV2_BASE)
    for p in frozen.parameters():
        p.requires_grad_(False)
    # Freeze all student params first
    for p in student.parameters():
        p.requires_grad_(False)
    n_layers = len(student.encoder.layer)
    for lyr in student.encoder.layer[n_layers - unfrozen_blocks:]:
        for p in lyr.parameters():
            p.requires_grad_(True)
    for p in student.layernorm.parameters():
        p.requires_grad_(True)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"student trainable params: {trainable/1e6:.2f} M (unfroze last {unfrozen_blocks} blocks + final layernorm)")
    return student, frozen, proc


def get_cls(model, pixel_values):
    out = model(pixel_values=pixel_values)
    return out.last_hidden_state[:, 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-split", default="train40k")
    ap.add_argument("--max-samples", type=int, default=20000)
    ap.add_argument("--teacher-feats", default=os.path.join(paths.CACHE_DIR, "imagenet_train40k_teacher.npz"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--bs", type=int, default=48)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-backbone", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--proj-dim", type=int, default=512)
    ap.add_argument("--unfrozen-blocks", type=int, default=4)
    ap.add_argument("--preserve-weight", type=float, default=1.0)
    ap.add_argument("--align-weight", type=float, default=1.0)
    ap.add_argument("--rsm-weight", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", default=os.path.join(paths.CACHE_DIR, "dinov2b_aligned.pt"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda"

    student, frozen, proc = make_student(unfrozen_blocks=args.unfrozen_blocks)
    student = student.to(device).train()
    frozen = frozen.to(device).eval()

    proj = nn.Sequential(
        nn.Linear(768, 1024),
        nn.GELU(),
        nn.Linear(1024, args.proj_dim),
    ).to(device)

    split_file = paths.IMAGENET_TRAIN_IDS if args.train_split == "train40k" else paths.IMAGENET_EVAL_IDS
    ds = ParquetIds(split_file, proc, max_samples=args.max_samples)
    print("dataset size:", len(ds))
    dl = DataLoader(ds, batch_size=args.bs, shuffle=True, num_workers=6,
                    collate_fn=collate, pin_memory=True, drop_last=True)

    tmap = TeacherFeatureMap(args.teacher_feats)

    opt = torch.optim.AdamW([
        {"params": [p for p in student.parameters() if p.requires_grad], "lr": args.lr_backbone},
        {"params": proj.parameters(), "lr": args.lr_head},
    ], weight_decay=args.wd)

    n_iters = args.epochs * len(dl)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_iters)

    step = 0
    for ep in range(args.epochs):
        for x, y, ids in dl:
            x = x.to(device, non_blocking=True)
            teacher_np = tmap.get(ids)
            t = torch.from_numpy(teacher_np).to(device)
            t_n = F.normalize(t, dim=1)

            # Student CLS and projection
            s_cls = get_cls(student, x)
            s_proj = proj(s_cls)
            s_n = F.normalize(s_proj, dim=1)

            # Preservation: match student CLS against frozen CLS in cosine
            with torch.no_grad():
                f_cls = get_cls(frozen, x)
            preserve_loss = 1.0 - F.cosine_similarity(s_cls, f_cls, dim=1).mean()

            # Alignment (pointwise cosine): push aligned student proj to teacher feature
            align_loss = 1.0 - (s_n * t_n).sum(dim=1).mean()

            # RSM matching within batch
            s_rsm = s_n @ s_n.T
            t_rsm = t_n @ t_n.T
            rsm_loss = F.mse_loss(s_rsm, t_rsm)

            loss = (args.align_weight * align_loss
                    + args.rsm_weight * rsm_loss
                    + args.preserve_weight * preserve_loss)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(student.parameters()) + list(proj.parameters()), 1.0
            )
            opt.step()
            sched.step()
            if step % 50 == 0:
                print(f"ep {ep} step {step}/{n_iters} loss={loss.item():.4f} "
                      f"align={align_loss.item():.4f} rsm={rsm_loss.item():.4f} "
                      f"preserve={preserve_loss.item():.4f}")
            step += 1

    # Save
    torch.save({
        "student_state": {k: v for k, v in student.state_dict().items()},
        "proj_state": proj.state_dict(),
        "proj_dim": args.proj_dim,
    }, args.save)
    print("saved", args.save)


if __name__ == "__main__":
    main()
