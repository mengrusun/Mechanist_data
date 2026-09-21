"""Shared helpers for the SemanticLens-style experiments.

Loads:
- ImageNet val images from parquet (10K subset)
- ResNet-50 / ViT-B/16 vision models (frozen)
- CLIP ViT-B/32 foundation model (frozen)
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

WORK_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = WORK_DIR / "data" / "imagenet-val"
MODEL_DIR = WORK_DIR / "models"
OUT_DIR = WORK_DIR / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def load_class_names() -> list[str]:
    """Return the 1000 ImageNet class names in index order.

    Uses the mapping in imagenet-val/README.md. The first synonym in each
    comma-separated list is used as the readable class label.
    """
    lines = (DATA_DIR / "README.md").read_text().splitlines()
    names: list[str] = []
    for ln in lines:
        parts = ln.strip().split(" ", 1)
        if len(parts) == 2 and parts[0].startswith("n"):
            readable = parts[1].split(",")[0].strip()
            names.append(readable)
    assert len(names) == 1000, len(names)
    return names


def load_eval_subset_ids() -> list[int]:
    """Return the global image ids that make up the 10K eval subset (0-indexed)."""
    ids: list[int] = []
    with open(DATA_DIR / "eval_subset_10k.txt") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            # format: <one-based idx>\t<shard_row>
            _one_based, tag = ln.split("\t")
            # tag is like 00_000001, first two digits shard, rest is row within shard
            shard = int(tag.split("_")[0])
            row = int(tag.split("_")[1])
            ids.append(shard * 100000 + row)  # placeholder, replaced below
    return ids


class ImageNetParquet(Dataset):
    """Reads all shards into an index of (path, row) tuples; decodes on __getitem__."""

    def __init__(self, transform, subset: str = "eval10k"):
        shard_dir = DATA_DIR / "data"
        self.shards = sorted(shard_dir.glob("train-*.parquet"))
        self.tables = [pd.read_parquet(p, columns=["image", "label"]) for p in self.shards]
        # build (shard_idx, row_idx, label, global_id) list
        rows = []
        gid = 0
        for si, t in enumerate(self.tables):
            for ri in range(len(t)):
                rows.append((si, ri, int(t.iloc[ri]["label"]), gid))
                gid += 1
        self.all_rows = rows

        if subset == "all":
            self.rows = rows
        elif subset == "eval10k":
            tags = set()
            with open(DATA_DIR / "eval_subset_10k.txt") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    # each line is just a tag like 00_000001
                    tags.add(ln.split()[-1])
            filt = []
            gid = 0
            for si, t in enumerate(self.tables):
                for ri in range(len(t)):
                    tag = f"{si:02d}_{ri:06d}"
                    if tag in tags:
                        filt.append((si, ri, int(t.iloc[ri]["label"]), gid))
                    gid += 1
            self.rows = filt
        elif subset == "train40k":
            tags = set()
            with open(DATA_DIR / "train_subset_40k.txt") as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    tags.add(ln.split()[-1])
            filt = []
            gid = 0
            for si, t in enumerate(self.tables):
                for ri in range(len(t)):
                    tag = f"{si:02d}_{ri:06d}"
                    if tag in tags:
                        filt.append((si, ri, int(t.iloc[ri]["label"]), gid))
                    gid += 1
            self.rows = filt
        else:
            raise ValueError(subset)

        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        si, ri, label, gid = self.rows[idx]
        rec = self.tables[si].iloc[ri]["image"]
        img = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
        return self.transform(img), label, gid


def imagenet_transform(size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def clip_transform(size: int = 224):
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def dual_transform(size_imagenet: int = 224, size_clip: int = 224):
    """Return a transform that yields (imagenet_tensor, clip_tensor)."""
    imn = imagenet_transform(size_imagenet)
    clp = clip_transform(size_clip)

    def _fn(img):
        # img is PIL; apply both starting from the same PIL image
        return imn(img), clp(img)

    return _fn


class DualImageNetParquet(ImageNetParquet):
    def __getitem__(self, idx):
        si, ri, label, gid = self.rows[idx]
        rec = self.tables[si].iloc[ri]["image"]
        img = Image.open(io.BytesIO(rec["bytes"])).convert("RGB")
        imn, clp = self.transform(img)
        return imn, clp, label, gid


def make_loader(subset: str = "eval10k", batch: int = 128, workers: int = 6, dual: bool = True):
    if dual:
        ds = DualImageNetParquet(dual_transform(), subset=subset)
    else:
        ds = ImageNetParquet(imagenet_transform(), subset=subset)
    return DataLoader(
        ds,
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=(workers > 0),
    )
