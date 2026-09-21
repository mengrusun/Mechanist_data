"""
M1_activations — Cache ResNet-50 channel activations on ImageNet-val (50 000 images).

One forward pass over the whole ImageNet-val set (via HuggingFace parquet). Cache spatial-mean-pooled
channel activations for `layer3`, `layer4`, and the pre-`fc` penultimate features (`avgpool`) plus the
`fc` logits (used to identify last-layer components by class).

Output HDF5 layout:
    /image_index       : (50000,) int32 — 0..49999
    /class_label       : (50000,) int32 — ImageNet class labels (torchvision wnid-sorted order)
    /activations/layer3   : (50000, 1024)  fp16
    /activations/layer4   : (50000, 2048)  fp16
    /activations/avgpool  : (50000, 2048)  fp16   (== fc input; layer just before fc)
    /activations/fc       : (50000, 1000)  fp16   (logits — the class-tied "last-layer components")
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import h5py

from common import (
    ImageNetValStream, imagenet_eval_transform, load_resnet50_torchvision,
    set_all_seeds,
)


LAYERS = ("layer3", "layer4", "avgpool", "fc")
LAYER_DIMS = {"layer3": 1024, "layer4": 2048, "avgpool": 2048, "fc": 1000}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/M1_activations/resnet50_val_activations.h5")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4, help="Data-loading threads (kept small; PIL decode is CPU-bound).")
    p.add_argument("--limit", type=int, default=0, help="Debug: limit to N images (0 = all 50000).")
    args = p.parse_args()

    set_all_seeds(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    model = load_resnet50_torchvision(device=device)

    # Register forward hooks on layer3, layer4, avgpool
    # Note: `fc` is the final Linear layer; we capture its output directly.
    captured: dict[str, torch.Tensor] = {}

    def make_hook(name: str, pool_spatial: bool):
        def hook(module, inp, out):
            if pool_spatial and out.dim() == 4:
                # spatial-mean over H, W  -> (B, C)
                captured[name] = out.mean(dim=(2, 3))
            elif out.dim() == 4:
                captured[name] = out.mean(dim=(2, 3))
            elif out.dim() == 2:
                captured[name] = out
            else:
                captured[name] = out.flatten(1)
        return hook

    hooks = []
    hooks.append(model.layer3.register_forward_hook(make_hook("layer3", pool_spatial=True)))
    hooks.append(model.layer4.register_forward_hook(make_hook("layer4", pool_spatial=True)))
    hooks.append(model.avgpool.register_forward_hook(make_hook("avgpool", pool_spatial=True)))
    hooks.append(model.fc.register_forward_hook(make_hook("fc", pool_spatial=False)))

    stream = ImageNetValStream()
    N = min(len(stream), args.limit) if args.limit > 0 else len(stream)
    print(f"[M1] caching activations for {N} images -> {out_path}")

    # Pre-allocate output
    if out_path.exists():
        out_path.unlink()
    h5 = h5py.File(out_path, "w")
    h5.create_dataset("image_index", (N,), dtype="int32")
    h5.create_dataset("class_label", (N,), dtype="int32")
    grp = h5.create_group("activations")
    for lname in LAYERS:
        chunk_rows = min(1024, N)
        grp.create_dataset(lname, (N, LAYER_DIMS[lname]), dtype="float16", chunks=(chunk_rows, LAYER_DIMS[lname]))

    tf = imagenet_eval_transform(crop=224)
    written = 0
    t0 = time.time()
    nan_count = 0
    with torch.no_grad():
        for idx, labels, bytes_list in stream.iter_images(batch_size=args.batch_size):
            if written >= N:
                break
            take = min(len(idx), N - written)
            idx = idx[:take]
            labels = labels[:take]
            bytes_list = bytes_list[:take]

            # Decode + preprocess (CPU) — do this in parallel with PIL
            batch = torch.stack([tf(Image.open(io.BytesIO(b)).convert("RGB")) for b in bytes_list]).to(device, non_blocking=True)
            captured.clear()
            _ = model(batch)

            h5["image_index"][written:written + take] = idx.astype("int32")
            h5["class_label"][written:written + take] = labels.astype("int32")
            for lname in LAYERS:
                arr = captured[lname].detach().to(torch.float16).cpu().numpy()
                assert arr.shape == (take, LAYER_DIMS[lname]), f"{lname}: {arr.shape}"
                nan_count += int(np.isnan(arr).sum())
                h5[f"activations/{lname}"][written:written + take] = arr

            written += take
            if written % (args.batch_size * 8) == 0 or written == N:
                elapsed = time.time() - t0
                thr = written / elapsed if elapsed > 0 else 0
                print(f"[M1] {written}/{N}  elapsed={elapsed:.1f}s  {thr:.0f} img/s")

    for h in hooks:
        h.remove()

    # Store metadata
    h5.attrs["nan_count_total"] = int(nan_count)
    h5.attrs["batch_size"] = args.batch_size
    h5.attrs["seed"] = args.seed
    h5.attrs["model"] = "resnet50 IMAGENET1K_V2"
    h5.attrs["preprocess"] = "square 256x256 -> center crop 224 -> ImageNet mean/std"
    h5.attrs["layers"] = list(LAYERS)
    for lname in LAYERS:
        h5["activations"][lname].attrs["dim"] = LAYER_DIMS[lname]
        h5["activations"][lname].attrs["pool"] = "spatial_mean" if lname in {"layer3", "layer4"} else "raw"
    h5.close()

    print(f"[M1] DONE. Total NaNs: {nan_count}. wall={time.time() - t0:.1f}s")
    if nan_count > 0:
        print("[M1] WARNING: NaNs present in activations!", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
