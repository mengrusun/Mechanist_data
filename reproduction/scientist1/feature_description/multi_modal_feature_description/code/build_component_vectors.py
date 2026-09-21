"""For each component c, take top-K activating images and pool their CLIP embeddings.

Also record the top-K image indices, activations, and class labels for later analyses.

Runs for:
- ResNet-50 layer4 (2048 channels)
- ViT-B/16 last-layer CLS (768 dims)
- ViT-B/16 last-layer mean patch (768 dims)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from code.common import OUT_DIR

K = 9  # exemplars per component


def _build(acts: np.ndarray, clip_img: np.ndarray, labels: np.ndarray, tag: str):
    N, C = acts.shape
    print(f"[{tag}] N={N} components={C}")
    # top-K argmax per column
    # np.argpartition then sort (K is small so full argsort per column is fine at C=2048)
    order = np.argsort(-acts, axis=0)  # (N, C) desc
    topk_idx = order[:K].T  # (C, K)
    topk_act = np.take_along_axis(acts, topk_idx.T, axis=0).T  # (C, K)
    topk_labels = labels[topk_idx]  # (C, K)
    # pool: mean of normalized CLIP embeddings of top-K, then renormalize
    v = clip_img[topk_idx].mean(axis=1)  # (C, D)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
    return dict(topk_idx=topk_idx, topk_act=topk_act, topk_labels=topk_labels, v=v)


def main():
    feats = np.load(OUT_DIR / "features" / "eval10k.npz")
    clip_img = feats["clip_img"]
    labels = feats["labels"]

    out_dir = OUT_DIR / "components"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tag, arr in [
        ("resnet50_layer4", feats["act_resnet"]),
        ("vit_b16_cls", feats["act_vit_cls"]),
        ("vit_b16_mean", feats["act_vit_mean"]),
    ]:
        out = _build(arr, clip_img, labels, tag)
        np.savez_compressed(out_dir / f"{tag}.npz", **out)
        print(f"[{tag}] saved v.shape={out['v'].shape} -> {out_dir / (tag + '.npz')}")


if __name__ == "__main__":
    main()
