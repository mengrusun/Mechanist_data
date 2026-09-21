"""
common.py — Shared utilities for the SemanticLens Component → CLIP Semantic-Vector experiment.

- Deterministic square-resize + center-crop preprocessing for ResNet-50 (per /experiment-tips/image).
- ImageNet-val streaming from HuggingFace parquet layout at /data/zhenqian/data/imagenet-val/data.
- ImageNet class-name loader.
- Seed helpers.
"""

from __future__ import annotations

import io
import os
import json
import glob
import hashlib
import random
from pathlib import Path
from typing import Iterator, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Paths (fixed by task.md — /data/zhenqian/data & /data/zhenqian/models allowed only)
# ---------------------------------------------------------------------------
WORKDIR = Path("/data/zhenqian/Reproduction1/mechanica/feature_description/multi_modal_feature_description")
DATA_DIR = Path("/data/zhenqian/data")
MODEL_DIR = Path("/data/zhenqian/models")
IMAGENET_VAL_DIR = DATA_DIR / "imagenet-val" / "data"      # 14 parquet files, 50 000 rows total
IMAGENET_META = DATA_DIR / "imagenet-val" / "dataset_infos.json"
IMAGENET_README = DATA_DIR / "imagenet-val" / "README.md"

TORCHVISION_CACHE = MODEL_DIR / "torchvision-checkpoints"
CLIP_LOCAL_PT = MODEL_DIR / "clip-vit-b32-oclip" / "openai_clip_vit_b32.pt"

# ---------------------------------------------------------------------------
# Preprocessing per /experiment-tips/image — square 256×256 → 224 center crop → ImageNet mean/std
# ---------------------------------------------------------------------------
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_eval_transform(crop: int = 224) -> T.Compose:
    """Canonical ImageNet eval preprocessing per /experiment-tips/image.

    Pipeline:
        1. Square resize to 256 x 256 (tuple => square, aspect ratio NOT preserved).
        2. Center crop to `crop` x `crop`.
        3. ToTensor (RGB / 255 -> [0, 1]).
        4. Normalize with ImageNet mean/std.
    """
    return T.Compose([
        T.Resize((256, 256), antialias=True),  # square resize per skills/image
        T.CenterCrop(crop),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# ImageNet class names
# ---------------------------------------------------------------------------
def load_imagenet_wnid_order() -> List[str]:
    """Sorted-wnid order (matches torchvision label indices)."""
    with open(IMAGENET_META) as f:
        info = json.load(f)
    return list(info["mrm8488--ImageNet1K-val"]["features"]["label"]["names"])


def load_imagenet_class_names(clean: bool = True) -> List[str]:
    """Returns 1000 class names in torchvision's wnid-sorted order.

    Authoritative source: `torchvision.models.ResNet50_Weights.IMAGENET1K_V2.meta['categories']` —
    torchvision keeps disambiguated names ("crane bird" not "crane", "maillot tank suit" not "maillot")
    that matter for CLIP text queries. Uses this by default; falls back to reading the README's synonyms
    (kept only for robustness if torchvision is missing).

    Args:
        clean: if True, keep only the primary label (first synonym before comma). Only used in the
               README fallback path.
    """
    try:
        import torchvision  # heavy — imported lazily
        return list(torchvision.models.ResNet50_Weights.IMAGENET1K_V2.meta["categories"])
    except Exception:
        pass
    # README fallback (WNID -> first synonym).
    wnids = load_imagenet_wnid_order()
    wnid_to_name: dict[str, str] = {}
    with open(IMAGENET_README) as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("n"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2 or not parts[0].startswith("n") or len(parts[0]) < 8:
                continue
            wnid, synonyms = parts
            name = synonyms.split(",")[0].strip() if clean else synonyms
            wnid_to_name[wnid] = name
    names = [wnid_to_name[w] for w in wnids]
    assert len(names) == 1000, f"Expected 1000 class names, got {len(names)}"
    return names


# ---------------------------------------------------------------------------
# ImageNet-val streaming (parquet with in-line PNG/JPEG bytes)
# ---------------------------------------------------------------------------
class ImageNetValStream:
    """Iterable dataset that streams (index, label, PIL.Image) tuples from the 14 parquet shards.

    Order is deterministic: shard order = lexicographic; within-shard row order = as stored.
    Index i corresponds to the same image across every run.
    """

    def __init__(self, parquet_dir: Path = IMAGENET_VAL_DIR):
        import pyarrow.parquet as pq  # heavy import; keep local
        self.pq = pq
        self.files = sorted(glob.glob(str(parquet_dir / "*.parquet")))
        assert len(self.files) == 14, f"Expected 14 parquet files, got {len(self.files)}"
        # Count once
        self.shard_sizes = [self.pq.read_metadata(f).num_rows for f in self.files]
        self.total = sum(self.shard_sizes)
        self.shard_offsets = [0]
        for n in self.shard_sizes[:-1]:
            self.shard_offsets.append(self.shard_offsets[-1] + n)
        assert self.total == 50000, f"Expected 50000 rows, got {self.total}"

    def __len__(self) -> int:
        return self.total

    def iter_images(self, batch_size: int = 256) -> Iterator[Tuple[np.ndarray, np.ndarray, List[bytes]]]:
        """Yield batches of (indices, labels, list-of-image-bytes).

        Reader keeps only the current batch decoded.
        """
        idx_counter = 0
        for shard_idx, f in enumerate(self.files):
            table = self.pq.read_table(f)
            img_col = table["image"]
            lbl_col = table["label"]
            n = table.num_rows
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                batch_bytes = []
                batch_labels = []
                for j in range(start, end):
                    s = img_col[j].as_py()
                    batch_bytes.append(s["bytes"])
                    batch_labels.append(lbl_col[j].as_py())
                batch_idx = np.arange(idx_counter, idx_counter + (end - start))
                idx_counter += (end - start)
                yield batch_idx, np.asarray(batch_labels, dtype=np.int64), batch_bytes

    def get_image(self, index: int) -> Tuple[Image.Image, int]:
        """Random access (slower — reads full shard). Use for reference-input crops only."""
        # locate shard
        for shard_idx, off in enumerate(self.shard_offsets):
            if index < off + self.shard_sizes[shard_idx]:
                local = index - off
                table = self.pq.read_table(self.files[shard_idx])
                s = table["image"][local].as_py()
                lbl = table["label"][local].as_py()
                img = Image.open(io.BytesIO(s["bytes"])).convert("RGB")
                return img, lbl
        raise IndexError(index)

    def get_images_batch(self, indices: List[int]) -> List[Tuple[Image.Image, int]]:
        """Batched random access — groups by shard to avoid re-reading."""
        by_shard: dict[int, list[int]] = {}
        for i, gi in enumerate(indices):
            for shard_idx, off in enumerate(self.shard_offsets):
                if gi < off + self.shard_sizes[shard_idx]:
                    by_shard.setdefault(shard_idx, []).append((i, gi - off))
                    break
        out: List[Optional[Tuple[Image.Image, int]]] = [None] * len(indices)
        for shard_idx, items in by_shard.items():
            table = self.pq.read_table(self.files[shard_idx])
            for orig_i, local_i in items:
                s = table["image"][local_i].as_py()
                lbl = table["label"][local_i].as_py()
                img = Image.open(io.BytesIO(s["bytes"])).convert("RGB")
                out[orig_i] = (img, lbl)
        return [x for x in out if x is not None]  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_of_file(path: str | Path, chunk_mb: int = 8) -> str:
    """Streaming SHA-256; safe for very large files (>1 GB)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_mb * 1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# CLIP loader (openai's own package — loads directly from the local .pt file via ~/.cache/clip symlink)
# ---------------------------------------------------------------------------
def load_openai_clip(device: str = "cuda"):
    """Load OpenAI CLIP ViT-B/32 from the local checkpoint.

    Returns (model, preprocess, tokenizer).
    """
    import clip
    # Ensure the ~/.cache/clip/ViT-B-32.pt symlink exists
    target = os.path.expanduser("~/.cache/clip/ViT-B-32.pt")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    if not os.path.exists(target):
        os.symlink(str(CLIP_LOCAL_PT), target)
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess, clip.tokenize


# ---------------------------------------------------------------------------
# ResNet-50 loader (torchvision IMAGENET1K_V2)
# ---------------------------------------------------------------------------
def load_resnet50_torchvision(device: str = "cuda"):
    """Load torchvision ResNet50 with IMAGENET1K_V2 weights from local cache."""
    os.environ["TORCH_HOME"] = str(TORCHVISION_CACHE)
    import torchvision
    m = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    m.eval()
    m.to(device)
    return m


if __name__ == "__main__":
    print("common.py smoke test")
    names = load_imagenet_class_names()
    print("Loaded", len(names), "ImageNet classes; first 5:", names[:5])
    stream = ImageNetValStream()
    print("ImageNet-val total:", len(stream))
    for idx, lbl, imgs in stream.iter_images(batch_size=4):
        print("First batch:", idx, "labels:", lbl, "image[0] bytes:", len(imgs[0]))
        break
