"""
M3_clip_embeddings — CLIP image embeddings for the union of reference inputs.

Single frozen CLIP ViT-B/32 forward pass over every unique ImageNet-val image referenced by any R_c.
Cache L2-normalized fp16 embeddings — this is the ONE CLIP pass in the main experiment.

Output HDF5:
    /image_index                : (N_unique,) int32 — the val indices covered
    /clip_image_embeddings      : (N_unique, 512) fp16 — L2-normalized
"""

from __future__ import annotations

import argparse
import io
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import h5py

from common import ImageNetValStream, load_openai_clip, set_all_seeds


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--refsets", default="runs/M2_reference_sets/refsets.h5")
    p.add_argument("--out", default="runs/M3_clip_embeddings/clip_val_embeddings.h5")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_all_seeds(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read refsets to find union of indices
    print(f"[M3] reading refsets from {args.refsets}")
    with h5py.File(args.refsets, "r") as h5r:
        top_idx = h5r["refsets/top256_image_indices"][:]  # (n_components, 256)
    unique_indices = np.unique(top_idx.ravel())
    unique_indices.sort()
    N_unique = len(unique_indices)
    print(f"[M3] {N_unique} unique reference images to embed via CLIP")

    # Reverse index: image_index -> position in the output arrays
    # But we won't need this for CLIP-Dissect downstream; downstream just uses the mapping we write.

    # Load CLIP
    device = args.device
    print("[M3] loading OpenAI CLIP ViT-B/32 ...")
    model, preprocess, tokenizer = load_openai_clip(device=device)
    dim = 512
    print(f"[M3] CLIP loaded. embedding dim = {dim}. Preprocess: {preprocess}")

    # Setup output
    if out_path.exists():
        out_path.unlink()
    h5o = h5py.File(out_path, "w")
    h5o.create_dataset("image_index", data=unique_indices.astype("int32"))
    h5o.create_dataset("clip_image_embeddings", (N_unique, dim), dtype="float16", chunks=(min(1024, N_unique), dim))
    h5o.attrs["clip_arch"] = "ViT-B/32"
    h5o.attrs["clip_pretrained"] = "openai"
    h5o.attrs["normalize"] = "L2"

    # We must load images by index (random access into 14 parquet shards).
    # Group by shard to avoid re-reading a shard.
    stream = ImageNetValStream()
    # Build shard -> list of (global_i, local_i)
    by_shard: dict[int, list[tuple[int, int]]] = {}
    shard_offsets = stream.shard_offsets
    shard_sizes = stream.shard_sizes
    n_shards = len(stream.files)
    for gi in unique_indices:
        gi = int(gi)
        # binary search into offsets
        for si in range(n_shards):
            if gi < shard_offsets[si] + shard_sizes[si]:
                by_shard.setdefault(si, []).append((gi, gi - shard_offsets[si]))
                break

    # Position in output = index in unique_indices (which is sorted)
    pos_of_gi = {int(gi): i for i, gi in enumerate(unique_indices)}

    written = 0
    nan_count = 0
    t0 = time.time()

    with torch.no_grad():
        for si in sorted(by_shard.keys()):
            items = by_shard[si]
            print(f"[M3] loading shard {si} ({len(items)} images requested)")
            import pyarrow.parquet as pq
            table = pq.read_table(stream.files[si])
            img_col = table["image"]
            # Sort items by local_i to read sequentially
            items.sort(key=lambda x: x[1])
            for start in range(0, len(items), args.batch_size):
                batch_items = items[start:start + args.batch_size]
                pil_imgs = [Image.open(io.BytesIO(img_col[li].as_py()["bytes"])).convert("RGB") for _, li in batch_items]
                pixel_batch = torch.stack([preprocess(im) for im in pil_imgs]).to(device, non_blocking=True)
                # CLIP forward
                emb = model.encode_image(pixel_batch)
                emb = F.normalize(emb.float(), dim=-1)
                arr = emb.detach().to(torch.float16).cpu().numpy()
                nan_count += int(np.isnan(arr).sum())
                # Write into output positions
                pos_list = [pos_of_gi[gi] for gi, _ in batch_items]
                # h5py fancy indexing requires sorted unique + monotonic; do row-by-row for correctness (small cost)
                for i, pos in enumerate(pos_list):
                    h5o["clip_image_embeddings"][pos] = arr[i]
                written += len(batch_items)
                if written % 4096 == 0 or written == N_unique:
                    elapsed = time.time() - t0
                    thr = written / elapsed if elapsed > 0 else 0
                    print(f"[M3] {written}/{N_unique}  elapsed={elapsed:.1f}s  {thr:.0f} img/s")

    h5o.attrs["nan_count_total"] = int(nan_count)
    h5o.attrs["wall_seconds"] = float(time.time() - t0)
    h5o.close()
    print(f"[M3] DONE. wall={time.time()-t0:.1f}s NaNs={nan_count}")


if __name__ == "__main__":
    main()
