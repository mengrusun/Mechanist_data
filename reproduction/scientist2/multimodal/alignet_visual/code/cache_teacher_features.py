"""M1.5 — Cache SigLIP-So400m teacher features (optionally head-projected) over an
ImageNet-val subset that will serve as the pool for M3 / M5 alignment finetunes.

For each image i in the subset:
  - Save SigLIP raw pooled feature f_i ∈ R^1152 (fp16).
  - If a teacher head is loaded, also save the head-projected f_i_aligned.

Then M3 / M5 will build in-batch triplet targets from these cached features on the fly.

We store to an HDF5 file for random-access at training time.
"""
from __future__ import annotations
import argparse
import os
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import (
    load_siglip_vision,
    ImageNetValIndex,
    ImageNetValImageDataset,
    seed_everything,
)
from fit_teacher_head import AlignmentHead


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_ckpt_dir", default=None, help="ignored — model dir is fixed to MODEL_DIR/siglip-so400m-patch14-384")
    p.add_argument("--teacher_head", default="none", help="path to teacher_head.pt OR 'none' to skip head-projection column")
    p.add_argument("--n_images", type=int, default=40_000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_path", required=True)
    args = p.parse_args()

    seed_everything(args.seed)
    out = Path(args.output_path); out.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda"

    # 1) SigLIP + processor
    print("[load] SigLIP-So400m ...")
    model, proc = load_siglip_vision(dtype=torch.float16, device=device)

    # 2) Optional head
    head = None
    head_str = args.teacher_head
    if head_str and head_str.lower() != "none" and Path(head_str).exists():
        print(f"[load] teacher head from {head_str}")
        head = AlignmentHead(dim=1152).to(device).eval()
        head.load_state_dict(torch.load(head_str, map_location=device))
        head = head.to(dtype=torch.float16)
    else:
        print("[load] no teacher head — caching UNALIGNED SigLIP pooler_output only")

    # 3) Sample subset
    print(f"[data] building imagenet-val index ...")
    idx = ImageNetValIndex.build()
    n = min(args.n_images, idx.total)
    print(f"[data] sampling {n} global indices from {idx.total}")
    global_idx = idx.sample_indices(n, seed=args.seed)
    ds = ImageNetValImageDataset(idx, global_indices=global_idx, processor=proc)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=0, shuffle=False)

    # 4) HDF5 write
    with h5py.File(out, "w") as h5:
        h5.attrs["source"] = "siglip-so400m-patch14-384"
        h5.attrs["n_images"] = n
        h5.attrs["seed"] = args.seed
        h5.attrs["head_ckpt"] = str(head_str)
        feat_ds = h5.create_dataset("feats_raw", shape=(n, 1152), dtype="f2")
        if head is not None:
            feat_ds_h = h5.create_dataset("feats_head", shape=(n, 1152), dtype="f2")
        gid_ds = h5.create_dataset("global_index", shape=(n,), dtype="i8")
        lab_ds = h5.create_dataset("label", shape=(n,), dtype="i8")

        t0 = time.time()
        cursor = 0
        with torch.no_grad():
            for batch in tqdm(dl, desc="cache siglip"):
                pv = batch["pixel_values"].to(device, dtype=torch.float16)
                out_v = model.vision_model(pv)
                pool = out_v.pooler_output  # [B, 1152] fp16
                bs = pool.shape[0]
                feat_ds[cursor:cursor+bs] = pool.cpu().numpy()
                if head is not None:
                    with torch.no_grad():
                        h_out = head(pool.float()).to(torch.float16)
                    feat_ds_h[cursor:cursor+bs] = h_out.cpu().numpy()
                gid_ds[cursor:cursor+bs] = batch["global_index"].numpy()
                lab_ds[cursor:cursor+bs] = batch["label"].numpy()
                cursor += bs
        print(f"[done] cached {cursor} images in {time.time()-t0:.1f}s -> {out}")


if __name__ == "__main__":
    main()
