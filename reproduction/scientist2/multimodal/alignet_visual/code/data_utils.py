"""Data + model loading utilities shared across all milestones.

Design rules (from EXPERIMENT_TIPS.md + EXPERIMENT_PLAN.md):
  1. Use each backbone's native HuggingFace image_processor — never re-invent T.Resize.
     - DINOv2: shortest-edge=256 -> center-crop=224, ImageNet mean/std.
     - SigLIP-So400m: 384x384 square, mean/std=(0.5,0.5,0.5).
  2. All model / data paths pulled from MODEL_DIR / DATA_DIR env vars.
  3. Fixed random seeds (default 42) applied at construct time.
  4. Never touch dirs outside working / DATA_DIR / MODEL_DIR.
"""

from __future__ import annotations
import io
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


DATA_DIR = Path(os.environ.get("DATA_DIR", "/data/zhenqian/data"))
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/data/zhenqian/models"))

# ---------- Reproducibility --------------------------------------------------

def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------- Model loading (native preprocessors) -----------------------------

def load_siglip_vision(dtype=torch.float16, device: str | torch.device = "cuda"):
    """Load SigLIP-So400m vision tower + its native image processor."""
    from transformers import AutoImageProcessor, AutoModel
    mp = MODEL_DIR / "siglip-so400m-patch14-384"
    model = AutoModel.from_pretrained(mp, torch_dtype=dtype).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(mp)
    return model, proc


def load_dinov2_base(dtype=torch.float16, device: str | torch.device = "cuda"):
    """Load DINOv2 ViT-B + its native image processor (short-edge=256, crop=224)."""
    from transformers import AutoImageProcessor, AutoModel
    mp = MODEL_DIR / "dinov2-base"
    model = AutoModel.from_pretrained(mp, torch_dtype=dtype).to(device).eval()
    proc = AutoImageProcessor.from_pretrained(mp)
    return model, proc


# ---------- THINGS dataset ---------------------------------------------------

def load_things_concepts() -> pd.DataFrame:
    """Return the 1854-concept THINGS metadata as a DataFrame indexed 0..1853."""
    path = DATA_DIR / "things_ooo_root" / "concepts-metadata_things.tsv"
    df = pd.read_csv(path, sep="\t")
    assert len(df) == 1854, f"expected 1854 THINGS concepts, got {len(df)}"
    return df.reset_index(drop=True)


def things_image_paths() -> list[Path]:
    """Return image path for each of the 1854 concepts (missing ones become None sentinel).

    Missing: 'wrap' (idx 1837), 'wrench' (idx 1841). Callers must handle gracefully.
    """
    df = load_things_concepts()
    root = DATA_DIR / "things_ooo_root" / "imgur_images"
    paths = []
    for w in df["uniqueID"]:
        p = root / f"{w}.jpg"
        paths.append(p if p.exists() else None)
    return paths


def load_triplets(split: str) -> np.ndarray:
    """Load a triplet file (train / test1 / test2 / test3 / validation).

    Returns int64 [N, 3] array with columns [pair_a, pair_b, odd_one_out].
    Values are 0-indexed concept IDs into THINGS.
    """
    fname = {
        "train": "trainset.txt",
        "test1": "testset1.txt",
        "test2": "testset2.txt",
        "test3": "testset3.txt",
        "validation": "validationset.txt",
        "heldout": "testset1.txt",   # M1 pass-criterion split
    }[split]
    path = DATA_DIR / "things_ooo_triplets" / "triplet_dataset" / fname
    arr = np.loadtxt(path, dtype=np.int64)
    return arr


class ThingsImageDataset(Dataset):
    """Serve THINGS images with a given HF image processor."""

    def __init__(self, processor, concept_indices: Sequence[int] | None = None):
        self.processor = processor
        self.paths = things_image_paths()
        self.concept_indices = list(concept_indices) if concept_indices is not None else list(range(1854))
        # Filter missing images silently — callers know which indices are usable via `usable_indices`
        self.usable = [i for i in self.concept_indices if self.paths[i] is not None]

    def __len__(self):
        return len(self.usable)

    def __getitem__(self, idx):
        cid = self.usable[idx]
        img = Image.open(self.paths[cid]).convert("RGB")
        pv = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pv, "concept_id": cid}


# ---------- ImageNet-val parquet (used as the "unlabelled ImageNet" pool) ----

@dataclass
class ImageNetValIndex:
    """Rows across all 14 parquets, addressable by a single global index."""
    parquet_files: list[Path]
    row_counts: list[int]
    labels: np.ndarray   # int64 [N]
    total: int

    @classmethod
    def build(cls) -> "ImageNetValIndex":
        root = DATA_DIR / "imagenet-val" / "data"
        pqs = sorted(root.glob("train-*.parquet"))
        assert len(pqs) == 14, f"expected 14 parquets, got {len(pqs)}"
        row_counts = []
        all_labels = []
        for p in pqs:
            df = pd.read_parquet(p, columns=["label"])
            row_counts.append(len(df))
            all_labels.append(df["label"].to_numpy())
        labels = np.concatenate(all_labels)
        return cls(parquet_files=pqs, row_counts=row_counts, labels=labels, total=int(sum(row_counts)))

    def sample_indices(self, n: int, seed: int = 42) -> np.ndarray:
        """Deterministic random sample of `n` global indices (with replacement=False)."""
        rng = np.random.default_rng(seed)
        return rng.choice(self.total, size=min(n, self.total), replace=False).astype(np.int64)


_IN_MEMORY_IMAGENET_VAL: dict = {"image_bytes": None, "labels": None}


def _load_all_imagenet_val_bytes_once(index: ImageNetValIndex) -> tuple[list[bytes], np.ndarray]:
    """Load ALL 50k imagenet-val image bytes into memory once (per-process, cached).

    This is a ~7 GB memory hit but eliminates the per-item parquet random-access latency
    that dominated wall-clock in the multi-worker naive dataset. With ~50 k items and
    workers ≥ 4, the per-worker parquet re-read amplifies to 10 h+ wall-clock; loading
    once is 30 s.

    Returns (list[bytes] of length N, labels array shape [N]).
    """
    if _IN_MEMORY_IMAGENET_VAL["image_bytes"] is not None:
        return _IN_MEMORY_IMAGENET_VAL["image_bytes"], _IN_MEMORY_IMAGENET_VAL["labels"]
    print(f"[data] loading ALL imagenet-val bytes ({index.total} images) into memory ...")
    all_bytes: list[bytes] = []
    all_labels: list[int] = []
    for p in index.parquet_files:
        df = pd.read_parquet(p)
        for r in df.itertuples(index=False):
            img_field = r.image
            b = img_field["bytes"] if isinstance(img_field, dict) else img_field
            all_bytes.append(b)
            all_labels.append(int(r.label))
    lab = np.asarray(all_labels, dtype=np.int64)
    _IN_MEMORY_IMAGENET_VAL["image_bytes"] = all_bytes
    _IN_MEMORY_IMAGENET_VAL["labels"] = lab
    print(f"[data] loaded {len(all_bytes)} image byte-strings ({sum(len(b) for b in all_bytes)/1e9:.2f} GB in-memory)")
    return all_bytes, lab


class ImageNetValImageDataset(Dataset):
    """Serve raw PIL RGB images from imagenet-val parquets by global index.

    Preprocessing is deferred to the model-specific processor injected by the caller.

    Implementation notes:
      - Loads ALL 50k image bytes into an in-process global cache on first construction
        (~7 GB, single 30-s hit). Later constructions in the same process reuse the cache.
      - `num_workers=0` recommended (the in-process cache is not shared with worker
        processes; setting workers > 0 forces each worker to reload the 7 GB pool).
    """

    def __init__(self, index: ImageNetValIndex, global_indices: Sequence[int], processor=None):
        self.index = index
        self.global_indices = np.asarray(global_indices, dtype=np.int64)
        self.processor = processor
        # Load all bytes once (cached global)
        all_bytes, all_labels = _load_all_imagenet_val_bytes_once(index)
        self._all_bytes = all_bytes
        self._all_labels = all_labels

    def __len__(self):
        return len(self.global_indices)

    def __getitem__(self, idx):
        g = int(self.global_indices[idx])
        b = self._all_bytes[g]
        img = Image.open(io.BytesIO(b)).convert("RGB")
        label = int(self._all_labels[g])
        item = {"label": label, "global_index": g}
        if self.processor is not None:
            item["pixel_values"] = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        else:
            item["image"] = img
        return item


# ---------- BREEDS ImageNet hierarchy ----------------------------------------

def load_imagenet1k_wnid_to_class() -> dict:
    """Return dict wnid -> imagenet class-id (0..999) from BREEDS dataset_class_info."""
    p = DATA_DIR / "breeds" / "imagenet_class_hierarchy" / "modified" / "dataset_class_info.json"
    data = json.load(open(p))
    return {row[1]: row[0] for row in data}


def load_breeds_hierarchy() -> dict:
    """Load the BREEDS-modified WordNet hierarchy as {parent_wnid: [child_wnid, ...]}."""
    edges = {}
    p = DATA_DIR / "breeds" / "imagenet_class_hierarchy" / "modified" / "class_hierarchy.txt"
    with open(p) as f:
        for line in f:
            parent, child = line.strip().split()
            edges.setdefault(parent, []).append(child)
    return edges


def wnid_ancestors_map(edges: dict) -> dict:
    """Reverse the hierarchy: {node: [immediate_parent]}."""
    parent = {}
    for p, cs in edges.items():
        for c in cs:
            parent.setdefault(c, []).append(p)
    return parent


if __name__ == "__main__":
    print("DATA_DIR:", DATA_DIR)
    print("MODEL_DIR:", MODEL_DIR)
    df = load_things_concepts()
    print(f"THINGS concepts: {len(df)}")
    paths = things_image_paths()
    missing = sum(1 for p in paths if p is None)
    print(f"THINGS images: {len(paths)} paths, {missing} missing")
    tr = load_triplets("train")
    te = load_triplets("heldout")
    print(f"triplets: train={len(tr)}, heldout={len(te)}")
    idx = ImageNetValIndex.build()
    print(f"imagenet-val: {idx.total} images across {len(idx.parquet_files)} parquets")
    sample_100 = idx.sample_indices(100, seed=42)
    print(f"sample_indices(100): first 5 = {sample_100[:5]}")
