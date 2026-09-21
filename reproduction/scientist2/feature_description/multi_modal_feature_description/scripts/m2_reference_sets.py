"""
M2_reference_sets — Top-k activation-driven reference-input sets R_c for every component.

Component universe (per EXPERIMENT_PLAN.md M2):
    - Last layer (fc): all 1000 logit units — component c is tied to ImageNet class c.
    - layer4: stratified sample of 500 channels from 2048.
    - layer3: stratified sample of 500 channels from 1024.

For each component, rank the 50 000 val images by activation magnitude and keep the top k_max=256.

Output HDF5:
    /components/component_id           (2000,) int32       — global component index
    /components/layer                  (2000,)  S16        — layer name string
    /components/local_index            (2000,) int32       — local channel index within the layer
    /components/class_label            (2000,) int32       — for fc components, the tied class label; else -1
    /refsets/top256_image_indices      (2000, 256) int32   — top-256 image indices (0..49999)
    /refsets/top256_activations        (2000, 256) float32 — corresponding activation values
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import h5py

from common import set_all_seeds


def stratified_sample(n_total: int, n_take: int, seed: int) -> np.ndarray:
    """Stratified sample of n_take channel indices from range(n_total).

    Stratifies by dividing the index range into n_take buckets and sampling one per bucket.
    Ensures early / mid / late conv positions all covered.
    """
    rng = np.random.default_rng(seed)
    if n_take >= n_total:
        return np.arange(n_total, dtype=np.int32)
    # Bucket edges
    edges = np.linspace(0, n_total, n_take + 1).astype(int)
    picks = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        hi = max(hi, lo + 1)
        picks.append(rng.integers(lo, hi))
    return np.asarray(sorted(set(picks)), dtype=np.int32)[:n_take]


def rank_top_k(activations: np.ndarray, channels: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """For each channel in `channels`, return (topk_image_indices, topk_activation_values) shape (len(channels), k)."""
    n_img = activations.shape[0]
    if k > n_img:
        k = n_img
    out_idx = np.zeros((len(channels), k), dtype=np.int32)
    out_act = np.zeros((len(channels), k), dtype=np.float32)
    # Chunked to keep memory bounded
    chunk = 64
    for start in range(0, len(channels), chunk):
        ch_batch = channels[start:start + chunk]
        # Load column subset: (n_img, |ch_batch|)
        sub = activations[:, ch_batch].astype(np.float32)  # -> fp32 for stable argmax
        # For each channel, top-k by activation value
        # argpartition first for speed, then sort within the top-k
        idx_topk_unsorted = np.argpartition(-sub, kth=k - 1, axis=0)[:k, :]
        # sort by descending activation
        vals_unsorted = np.take_along_axis(sub, idx_topk_unsorted, axis=0)
        order = np.argsort(-vals_unsorted, axis=0)
        idx_topk = np.take_along_axis(idx_topk_unsorted, order, axis=0)
        vals_topk = np.take_along_axis(sub, idx_topk, axis=0)
        # (k, |ch_batch|)  ->  transpose to (|ch_batch|, k)
        out_idx[start:start + len(ch_batch)] = idx_topk.T
        out_act[start:start + len(ch_batch)] = vals_topk.T
    return out_idx, out_act


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--activations", default="runs/M1_activations/resnet50_val_activations.h5")
    p.add_argument("--out", default="runs/M2_reference_sets/refsets.h5")
    p.add_argument("--n_last", type=int, default=1000)
    p.add_argument("--n_layer4", type=int, default=500)
    p.add_argument("--n_layer3", type=int, default=500)
    p.add_argument("--k_max", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_all_seeds(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(args.activations, "r") as h5in:
        image_index_all = h5in["image_index"][:]
        class_label_all = h5in["class_label"][:]
        n_img = image_index_all.shape[0]
        print(f"[M2] activations file has {n_img} images.")

        # Layer channel counts
        n_layer3_total = h5in["activations/layer3"].shape[1]  # 1024
        n_layer4_total = h5in["activations/layer4"].shape[1]  # 2048
        n_fc = h5in["activations/fc"].shape[1]                 # 1000
        print(f"[M2] channel counts: layer3={n_layer3_total} layer4={n_layer4_total} fc={n_fc}")

        # Pick channels
        fc_channels = np.arange(min(args.n_last, n_fc), dtype=np.int32)                    # 0..999
        l4_channels = stratified_sample(n_layer4_total, args.n_layer4, seed=args.seed)     # (500,)
        l3_channels = stratified_sample(n_layer3_total, args.n_layer3, seed=args.seed + 1) # (500,)
        print(f"[M2] picked: fc={len(fc_channels)}  layer4={len(l4_channels)}  layer3={len(l3_channels)}")

        # Component metadata
        n_total = len(fc_channels) + len(l4_channels) + len(l3_channels)
        component_id = np.arange(n_total, dtype=np.int32)
        layer_names = (["fc"] * len(fc_channels) + ["layer4"] * len(l4_channels) + ["layer3"] * len(l3_channels))
        local_index = np.concatenate([fc_channels, l4_channels, l3_channels]).astype(np.int32)
        class_label_per_comp = np.concatenate([
            fc_channels.astype(np.int32),                             # fc comp c -> class c
            -1 * np.ones(len(l4_channels), dtype=np.int32),           # hidden -> -1
            -1 * np.ones(len(l3_channels), dtype=np.int32),
        ])

        # Rank top-k per component (chunked reading of each layer)
        top_idx_all = np.zeros((n_total, args.k_max), dtype=np.int32)
        top_act_all = np.zeros((n_total, args.k_max), dtype=np.float32)

        # fc
        print(f"[M2] ranking top-{args.k_max} images for {len(fc_channels)} fc components ...")
        # Read fc activations into memory (50k x 1000 fp16 = 100 MB)
        fc_act = h5in["activations/fc"][:]
        idx, val = rank_top_k(fc_act, fc_channels, args.k_max)
        top_idx_all[:len(fc_channels)] = idx
        top_act_all[:len(fc_channels)] = val

        # layer4
        print(f"[M2] ranking top-{args.k_max} images for {len(l4_channels)} layer4 components ...")
        l4_act = h5in["activations/layer4"][:]
        idx, val = rank_top_k(l4_act, l4_channels, args.k_max)
        start = len(fc_channels)
        top_idx_all[start:start + len(l4_channels)] = idx
        top_act_all[start:start + len(l4_channels)] = val

        # layer3
        print(f"[M2] ranking top-{args.k_max} images for {len(l3_channels)} layer3 components ...")
        l3_act = h5in["activations/layer3"][:]
        idx, val = rank_top_k(l3_act, l3_channels, args.k_max)
        start = len(fc_channels) + len(l4_channels)
        top_idx_all[start:start + len(l3_channels)] = idx
        top_act_all[start:start + len(l3_channels)] = val

    # Coverage histogram (R1 risk mitigation)
    all_indices, counts = np.unique(top_idx_all.ravel(), return_counts=True)
    cover_frac = len(all_indices) / n_img
    high_cover_mask = counts > 0.05 * n_total  # images appearing in > 5% of R_c sets
    print(f"[M2] unique reference images: {len(all_indices)} / {n_img} = {cover_frac:.3f}")
    print(f"[M2] images appearing in > 5% of R_c: {int(high_cover_mask.sum())}")

    # Write output
    if out_path.exists():
        out_path.unlink()
    with h5py.File(out_path, "w") as h5o:
        gc = h5o.create_group("components")
        gc.create_dataset("component_id", data=component_id)
        gc.create_dataset("layer", data=np.array(layer_names, dtype="S16"))
        gc.create_dataset("local_index", data=local_index)
        gc.create_dataset("class_label", data=class_label_per_comp)
        gr = h5o.create_group("refsets")
        gr.create_dataset("top256_image_indices", data=top_idx_all)
        gr.create_dataset("top256_activations", data=top_act_all)
        # coverage summary
        gs = h5o.create_group("summary")
        gs.attrs["n_components"] = int(n_total)
        gs.attrs["k_max"] = int(args.k_max)
        gs.attrs["unique_reference_images"] = int(len(all_indices))
        gs.attrs["reference_image_coverage_fraction"] = float(cover_frac)
        gs.attrs["images_over_5pct_cover"] = int(high_cover_mask.sum())
        h5o.attrs["seed"] = int(args.seed)
        h5o.attrs["source_activations"] = args.activations

    print(f"[M2] wrote {out_path}")


if __name__ == "__main__":
    main()
