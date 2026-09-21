"""
M12_cross_model_verify — Cross-model transfer P3.

For a given swap model ∈ {vit_b_16, vgg16, efficientnet_b0}, repeat M1 → M2 (last layer only)
→ M4 (mean, k=16) → M8 (text-query MRR/R@10 vs permutation baseline).

CLIP-Dissect / SemanticLens argues each inspected model should get its last-layer components
placed in the SAME frozen CLIP semantic space. If v_c geometry transfers, MRR should be
significantly above the permutation baseline on each swap model.

For last-layer only:
    ResNet-50 fc = 1000 logits (already done — skip)
    ViT-B/16 heads.head (linear) = 1000 logits
    VGG-16 classifier[-1] (linear) = 1000 logits
    EfficientNet-B0 classifier[-1] = 1000 logits

We cache logits, pick top-k per class-tied component, CLIP-embed reference images, pool mean,
then compute MRR / R@10.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import h5py

from common import (
    ImageNetValStream, imagenet_eval_transform, TORCHVISION_CACHE,
    load_openai_clip, set_all_seeds,
)


MODEL_LAYERS = {
    "vit_b_16": {
        "n_classes": 1000,
        "load": lambda: __import__("torchvision.models", fromlist=["vit_b_16", "ViT_B_16_Weights"]).vit_b_16(weights="IMAGENET1K_V1"),
    },
    "vgg16": {
        "n_classes": 1000,
        "load": lambda: __import__("torchvision.models", fromlist=["vgg16", "VGG16_Weights"]).vgg16(weights="IMAGENET1K_V1"),
    },
    "efficientnet_b0": {
        "n_classes": 1000,
        "load": lambda: __import__("torchvision.models", fromlist=["efficientnet_b0", "EfficientNet_B0_Weights"]).efficientnet_b0(weights="IMAGENET1K_V1"),
    },
}


def compute_ranks_and_mrr(v_c: np.ndarray, text_emb: np.ndarray, class_labels: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    v = v_c / (np.linalg.norm(v_c, axis=1, keepdims=True) + 1e-8)
    t = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
    cos = v @ t.T
    ranks = []
    for c in range(text_emb.shape[0]):
        gt_idx = np.where(class_labels == c)[0]
        if len(gt_idx) == 0:
            continue
        col = cos[:, c]
        ranks.append(int((col > col[int(gt_idx[0])]).sum()) + 1)
    ranks = np.array(ranks, dtype=np.int32)
    mrr = float((1.0 / ranks).mean())
    r5 = float((ranks <= 5).mean())
    r10 = float((ranks <= 10).mean())
    return mrr, r5, r10, cos, ranks


def perm_mrr_ci(cos: np.ndarray, class_labels: np.ndarray, n_permute: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    mrrs = np.zeros(n_permute, dtype=np.float32)
    for i in range(n_permute):
        perm = rng.permutation(class_labels)
        ranks = []
        for c in range(cos.shape[1]):
            gt_idx = np.where(perm == c)[0]
            if len(gt_idx) == 0:
                continue
            col = cos[:, c]
            ranks.append(int((col > col[int(gt_idx[0])]).sum()) + 1)
        r = np.array(ranks)
        if len(r) > 0:
            mrrs[i] = (1.0 / r).mean()
    return float(np.quantile(mrrs, 0.95))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inspected_model", required=True, choices=list(MODEL_LAYERS.keys()))
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--pool", type=str, default="mean")
    p.add_argument("--clip_emb", default="runs/M3_clip_embeddings/clip_val_embeddings.h5",
                   help="Shared CLIP embeddings — we can REUSE these iff the swap model's top-k lies within the covered indices.")
    p.add_argument("--text", default="runs/M5_text_embeddings/text_embeddings.h5")
    p.add_argument("--out", required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--n_permute", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    set_all_seeds(args.seed)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    device = args.device
    os.environ["TORCH_HOME"] = str(TORCHVISION_CACHE)
    print(f"[M12] loading {args.inspected_model} ...")
    model = MODEL_LAYERS[args.inspected_model]["load"]()
    model.eval().to(device)
    n_classes = MODEL_LAYERS[args.inspected_model]["n_classes"]

    # Cache last-layer logits for all val images
    stream = ImageNetValStream()
    N = len(stream) if args.limit <= 0 else min(len(stream), args.limit)
    print(f"[M12] caching {n_classes}-logit vectors for {N} val images through {args.inspected_model}")
    tf = imagenet_eval_transform(crop=224)
    logits_all = np.zeros((N, n_classes), dtype=np.float32)
    written = 0
    t0 = time.time()
    with torch.no_grad():
        for idx, labels, bytes_list in stream.iter_images(batch_size=args.batch_size):
            if written >= N:
                break
            take = min(len(idx), N - written)
            batch = torch.stack([tf(Image.open(io.BytesIO(b)).convert("RGB")) for b in bytes_list[:take]]).to(device)
            l = model(batch).detach().cpu().numpy()
            logits_all[written:written + take] = l
            written += take
            if written % (args.batch_size * 8) == 0 or written == N:
                elapsed = time.time() - t0
                print(f"[M12] {args.inspected_model}: {written}/{N} images ({written/max(elapsed,1e-6):.0f} img/s)")
    print(f"[M12] logits cached (wall {time.time()-t0:.1f}s)")

    # Rank top-k images per class-tied component
    from m2_reference_sets import rank_top_k
    fc_channels = np.arange(n_classes, dtype=np.int32)
    top_idx, top_act = rank_top_k(logits_all, fc_channels, args.k)  # (n_classes, k)

    # Determine which images are new (not covered by shared clip_emb from M3)
    with h5py.File(args.clip_emb, "r") as h5c:
        covered_idx = set(int(i) for i in h5c["image_index"][:])
        # But we need embeddings for the NEW indices too. Since coverage should be very high
        # (ResNet-50 top-k union covered ~ N unique images), swap models may still pick some new images.
    needed = set(int(i) for i in top_idx.ravel())
    missing = needed - covered_idx
    print(f"[M12] {args.inspected_model}: {len(needed)} unique needed, {len(missing)} not in M3 CLIP cache")

    # Load CLIP + preprocess
    model_clip, preprocess, tokenizer = load_openai_clip(device=device)
    # Build embeddings dict from existing cache
    with h5py.File(args.clip_emb, "r") as h5c:
        idx_arr = h5c["image_index"][:]
        emb_arr = h5c["clip_image_embeddings"][:]
    idx_to_emb = {int(i): emb_arr[j] for j, i in enumerate(idx_arr)}

    # Embed missing images
    if missing:
        print(f"[M12] embedding {len(missing)} missing images through CLIP ...")
        # Group by shard
        by_shard: dict[int, list[tuple[int, int]]] = {}
        for gi in missing:
            for si in range(len(stream.shard_offsets)):
                if gi < stream.shard_offsets[si] + stream.shard_sizes[si]:
                    by_shard.setdefault(si, []).append((gi, gi - stream.shard_offsets[si]))
                    break
        import pyarrow.parquet as pq
        with torch.no_grad():
            for si, items in by_shard.items():
                table = pq.read_table(stream.files[si])
                img_col = table["image"]
                items.sort(key=lambda x: x[1])
                for start in range(0, len(items), args.batch_size):
                    batch_items = items[start:start + args.batch_size]
                    pil = [Image.open(io.BytesIO(img_col[li].as_py()["bytes"])).convert("RGB") for _, li in batch_items]
                    pix = torch.stack([preprocess(im) for im in pil]).to(device)
                    e = F.normalize(model_clip.encode_image(pix).float(), dim=-1).cpu().numpy().astype(np.float16)
                    for i, (gi, _) in enumerate(batch_items):
                        idx_to_emb[int(gi)] = e[i]

    # Pool v_c for each class-tied component (mean, k)
    D = emb_arr.shape[1]
    v_c = np.zeros((n_classes, D), dtype=np.float32)
    for c in range(n_classes):
        picks = top_idx[c, :args.k]
        embs = np.stack([idx_to_emb[int(gi)] for gi in picks]).astype(np.float32)
        if args.pool == "mean":
            v = embs.mean(axis=0)
        elif args.pool == "act_weighted_mean":
            w = np.clip(top_act[c, :args.k], 0.0, None)
            w = w / (w.sum() + 1e-8)
            v = (w[:, None] * embs).sum(axis=0)
        else:
            v = embs.mean(axis=0)  # only mean is planned for M12
        nrm = np.linalg.norm(v)
        if nrm > 0:
            v = v / nrm
        v_c[c] = v

    # MRR computation
    with h5py.File(args.text, "r") as h5t:
        text_emb = h5t["imagenet1k_classes/embeddings"][:]
    class_labels = np.arange(n_classes, dtype=np.int32)  # component c is tied to class c
    mrr, r5, r10, cos, ranks = compute_ranks_and_mrr(v_c, text_emb, class_labels)
    perm_upper = perm_mrr_ci(cos, class_labels, n_permute=args.n_permute, seed=args.seed)
    significant = bool(mrr > perm_upper)

    result = {
        "inspected_model": args.inspected_model,
        "k": args.k,
        "pool": args.pool,
        "n_components": int(n_classes),
        "n_val_images_used": int(N),
        "mrr": mrr,
        "recall_at_5": r5,
        "recall_at_10": r10,
        "permutation_mrr_ci_95_upper": perm_upper,
        "significant": significant,
        "n_permute": args.n_permute,
    }
    outp.write_text(json.dumps(result, indent=2))
    print(f"[M12] {args.inspected_model}: MRR={mrr:.4f} (perm95 {perm_upper:.4f}), R@10={r10:.4f}, sig={significant}")


if __name__ == "__main__":
    main()
