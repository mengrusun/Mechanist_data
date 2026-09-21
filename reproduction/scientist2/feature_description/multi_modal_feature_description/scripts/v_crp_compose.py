"""
v_crp_compose.py — Verify-stage method-swap variant: CLIP-Dissect + Zennit-CRP compose.

This is the CRP-compose method swap for claim C2 verification.

Instead of embedding uncropped reference images, this script:
1. For each (component c, reference image x in top-k), runs a Zennit LRP-epsilon backward
   pass through ResNet-50 conditioned on component c's fc unit index to get pixel-level
   relevance R_c(x).  For fc components, the fc unit index equals the class_label by
   construction (ResNet-50 fc: unit i = ImageNet class i, confirmed: class_label == local_index
   for all fc components in refsets.h5).
2. Crops x to the bounding box of pixels whose positive relevance >= the top-q th percentile
   of all positive relevance values (padded to >= min_crop_px before CLIP resize).
   (Percentile of positive pixel values approximates the "top-q% of positive relevance mass"
   criterion from the DIFF; edge case: no positive relevance → full image fallback.)
3. Embeds the cropped region with frozen CLIP ViT-B/32.
4. Mean-pools the k=16 cropped CLIP embeddings into v_c^CRP.
5. Runs P2a (text-query MRR/R@10 vs. permutation baseline) and P2b (disjoint-half stability
   on top-2*half_k=16 refs split into two halves of half_k=8) on v_c^CRP.

What changed vs main experiment (M8 / CLIP-Dissect plain):
- ADDED: LRP backward pass + bbox crop before CLIP embedding.
- HELD FIXED: same ResNet-50 (IMAGENET1K_V2 via load_resnet50_torchvision), same frozen
  CLIP ViT-B/32, same top-k=16 reference inputs, same text embeddings, same MRR metric,
  same permutation test (seed=42, n=1000), same mean-pool, same scope (fc only for cost).
- SCOPE NOTE: fc only (1000 components) rather than fc+layer4+layer3 (2000 total); this
  reduces cost ~2x while preserving the primary metric (fc P2a MRR was the headline result).

References:
- MECHANISM_ROUTING.md §"Note on Zennit-CRP as a fallback"
- skills/mechanism-skills/multi-modal/SKILL.md §"Compose the two when the task is component -> concept vector"
- Zennit: https://github.com/chr5tphr/zennit (LRP rules for PyTorch)

Usage:
    CUDA_VISIBLE_DEVICES=1,2,3,5,6 python scripts/v_crp_compose.py \\
        --refsets runs/M2_reference_sets/refsets.h5 \\
        --text runs/M5_text_embeddings/text_embeddings.h5 \\
        --out_dir verify/C2_clip_joint_semantic_space/variants/method-swap-crp-compose \\
        --k 16 --n_permute 1000 --seed 42 \\
        --relevance_pct 90 --min_crop_px 32
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
import h5py
from PIL import Image

# ---------------------------------------------------------------------------
# Conditional imports (zennit)
# ---------------------------------------------------------------------------
try:
    from zennit.composites import EpsilonGammaBox
    from zennit.canonizers import SequentialMergeBatchNorm
    ZENNIT_AVAILABLE = True
except ImportError:
    ZENNIT_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    set_all_seeds, ImageNetValStream, load_openai_clip, load_resnet50_torchvision,
    imagenet_eval_transform, IMAGENET_MEAN, IMAGENET_STD, MODEL_DIR
)

# ---------------------------------------------------------------------------
# LRP relevance map via Zennit
# ---------------------------------------------------------------------------

def build_lrp_composite(model, device, low_bound, high_bound):
    """Build an EpsilonGammaBox composite for ResNet-50 (LRP-epsilon body, Box for first conv).

    Args:
        model: the ResNet-50 model (used only to pass canonizers; not modified here)
        device: torch device string
        low_bound: list[list[float]] — per-channel lower pixel bound after ImageNet normalize
        high_bound: list[list[float]] — per-channel upper pixel bound after ImageNet normalize
    Returns:
        EpsilonGammaBox composite (context manager, use with composite.context(model))
    """
    from zennit.composites import EpsilonGammaBox
    from zennit.canonizers import SequentialMergeBatchNorm
    canonizers = [SequentialMergeBatchNorm()]
    # low/high: (1,3,1,1) tensors representing the per-channel input range for the Box rule.
    # This is standard EpsilonGammaBox usage for normalized ImageNet inputs.
    composite = EpsilonGammaBox(
        low=torch.tensor(low_bound, device=device).view(1, 3, 1, 1),
        high=torch.tensor(high_bound, device=device).view(1, 3, 1, 1),
        epsilon=1e-6,
        gamma=0.25,
        canonizers=canonizers,
    )
    return composite


def compute_relevance_map(model, composite, img_tensor: torch.Tensor,
                           channel_idx: int) -> np.ndarray:
    """Compute LRP relevance map conditioned on a single fc output unit.

    Uses a one-hot gradient seed at `channel_idx` so relevance propagates back
    only through that unit's logit.  This is the standard Zennit pattern for
    class-conditional LRP:
        backward seed: e_c (one-hot) — only unit c contributes to relevance

    Args:
        model: ResNet-50 in eval mode, NOT wrapped — wrapping is done via composite.context()
        composite: EpsilonGammaBox composite (zennit)
        img_tensor: (1, 3, 224, 224) tensor on device; requires_grad will be set internally
        channel_idx: which fc output unit (= ImageNet class index) to attribute to
    Returns:
        (224, 224) numpy array — per-pixel relevance (sum over channels; NOT abs-normalized;
        positive values indicate class-supporting regions)
    """
    # Detach and re-enable grad for clean backward
    inp = img_tensor.clone().detach().requires_grad_(True)
    with composite.context(model):
        out = model(inp)              # (1, n_classes)
        # One-hot gradient seed: propagate only through unit channel_idx.
        # For LRP via Zennit, the backward seed represents dL/dout — a one-hot
        # of magnitude 1.0 selects exactly one output logit for relevance propagation.
        if not (0 <= channel_idx < out.shape[1]):
            raise ValueError(
                f"channel_idx={channel_idx} out of range [0, {out.shape[1]}). "
                "This should never happen for fc-only scope (0..999)."
            )
        seed = torch.zeros_like(out)
        seed[0, channel_idx] = 1.0
        out.backward(seed)

    # inp.grad: (1, 3, 224, 224) — pixel-level relevance attributed from channel_idx
    R = inp.grad.detach().cpu()      # (1, 3, 224, 224)
    R = R.squeeze(0).sum(dim=0)      # (224, 224) sum over RGB channels
    return R.numpy()


def crop_to_relevance_bbox(img_pil: Image.Image, R: np.ndarray,
                            percentile: float = 90, min_px: int = 32) -> Image.Image:
    """Crop img_pil to the bounding box of high-relevance pixels.

    Thresholds R to pixels >= the `percentile`-th percentile of all positive relevance
    values, then computes the tight bounding box.  This approximates the "top-q% of
    positive relevance mass" criterion: the threshold is set so that roughly (100-percentile)%
    of positive mass lies above it.

    Args:
        img_pil: source image (any size, RGB)
        R: (224, 224) numpy relevance map (may contain negatives; only positive values matter)
        percentile: threshold percentile among positive relevance pixels (default 90)
        min_px: minimum bounding box side length in the original image's pixel space (default 32)
    Returns:
        Cropped PIL Image.  Falls back to full image if no positive relevance is found.
    """
    R_pos = np.clip(R, 0, None)
    if R_pos.max() < 1e-10:
        # No positive relevance — return full image (LRP found no supportive region)
        return img_pil

    thresh = np.percentile(R_pos[R_pos > 0], percentile)
    mask = (R_pos >= thresh)
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    if len(rows) == 0 or len(cols) == 0:
        return img_pil

    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())

    # Relevance map is at 224x224; scale bbox to original PIL image dimensions
    W_orig, H_orig = img_pil.size   # PIL: (width, height)
    scale_h = H_orig / 224.0
    scale_w = W_orig / 224.0

    y0 = int(r0 * scale_h)
    y1 = int((r1 + 1) * scale_h)
    x0 = int(c0 * scale_w)
    x1 = int((c1 + 1) * scale_w)

    # Enforce minimum crop size in original pixel space
    if (y1 - y0) < min_px:
        pad = (min_px - (y1 - y0)) // 2
        y0 = max(0, y0 - pad)
        y1 = min(H_orig, y1 + pad)
    if (x1 - x0) < min_px:
        pad = (min_px - (x1 - x0)) // 2
        x0 = max(0, x0 - pad)
        x1 = min(W_orig, x1 + pad)

    crop = img_pil.crop((x0, y0, x1, y1))
    return crop


# ---------------------------------------------------------------------------
# CLIP embedding of a list of PIL images
# ---------------------------------------------------------------------------

def clip_embed_images(clip_model, clip_preprocess, images: list, device: str,
                       batch_size: int = 64) -> np.ndarray:
    """Embed a list of PIL images with frozen CLIP ViT-B/32.

    Returns (N, 512) L2-normalized float32 embeddings.
    The model is NOT modified (no_grad + model should be in eval mode externally).
    """
    all_embs = []
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        tensors = torch.stack([clip_preprocess(img) for img in batch]).to(device)
        with torch.no_grad():
            emb = clip_model.encode_image(tensors).float()
            emb = F.normalize(emb, dim=-1)
        all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs, axis=0) if all_embs else np.zeros((0, 512), dtype=np.float32)


# ---------------------------------------------------------------------------
# MRR / R@k helpers (matching m8_c2_queryability.py)
# ---------------------------------------------------------------------------

def compute_mrr(v_c_last: np.ndarray, text_emb: np.ndarray,
                class_labels: np.ndarray,
                valid_mask: np.ndarray | None = None) -> tuple[float, float]:
    """Compute MRR and R@10 exactly as in M8 (m8_c2_queryability.py).

    For each class c: rank the component v_c[c] against all components on text query e_t[c].
    GT component for class c is identified by class_labels[i] == c (picks first match, same
    as M8 — fc components have unique class_label per unit so this is deterministic).

    Args:
        v_c_last: (n_comp, 512) L2-normalized component vectors
        text_emb: (n_classes, 512) L2-normalized text embeddings
        class_labels: (n_comp,) integer array — GT class index for each component
        valid_mask: (n_comp,) bool or None — if provided, zero-vector / failed components
                    are excluded from both the candidate pool and as GT matches.
                    When None, all components are used (matches M8 behaviour).
    Returns:
        (MRR, R@10) both in [0, 1]
    """
    # Exclude invalid (zero-vector) components: they have no meaningful cosine.
    if valid_mask is not None:
        v_c_last = v_c_last[valid_mask]
        class_labels = class_labels[valid_mask]

    v = v_c_last / (np.linalg.norm(v_c_last, axis=1, keepdims=True) + 1e-8)
    t = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
    cos = v @ t.T  # (n_comp_valid, n_classes)

    n_classes = text_emb.shape[0]
    ranks = np.zeros(n_classes, dtype=np.int32)
    for c in range(n_classes):
        gt_mask = np.where(class_labels == c)[0]
        if len(gt_mask) == 0:
            # GT component for this class was filtered out (all images failed); skip
            ranks[c] = -1
            continue
        gt_i = int(gt_mask[0])
        col = cos[:, c]
        # Tie-break: rank = 1 + #components with strictly greater cosine (standard MRR)
        ranks[c] = int((col > col[gt_i]).sum()) + 1

    valid = ranks[ranks > 0]
    mrr = float((1.0 / valid).mean())
    r10 = float((valid <= 10).mean())
    return mrr, r10


def permutation_upper(v_c_last: np.ndarray, text_emb: np.ndarray,
                       class_labels: np.ndarray,
                       n_permute: int = 1000, seed: int = 42,
                       valid_mask: np.ndarray | None = None) -> float:
    """95th percentile of permuted MRR (same as M8 permutation_upper).

    Shuffles class_labels (the assignment of components to classes) rather than
    shuffling text embeddings, recomputing MRR for each shuffle.  This matches the
    main experiment's permutation baseline.

    Args:
        valid_mask: same semantics as compute_mrr — exclude invalid components
    """
    if valid_mask is not None:
        v_c_last = v_c_last[valid_mask]
        class_labels = class_labels[valid_mask]

    v = v_c_last / (np.linalg.norm(v_c_last, axis=1, keepdims=True) + 1e-8)
    t = text_emb / (np.linalg.norm(text_emb, axis=1, keepdims=True) + 1e-8)
    cos = v @ t.T  # (n_comp_valid, n_classes)
    rng = np.random.default_rng(seed)
    n_classes = text_emb.shape[0]
    mrrs = np.zeros(n_permute, dtype=np.float32)
    lbl = class_labels.copy()
    for i in range(n_permute):
        perm = rng.permutation(lbl)
        ranks = np.zeros(n_classes, dtype=np.int32)
        for c in range(n_classes):
            gt_idx = np.where(perm == c)[0]
            if len(gt_idx) == 0:
                ranks[c] = -1
                continue
            gt_i = int(gt_idx[0])
            col = cos[:, c]
            ranks[c] = int((col > col[gt_i]).sum()) + 1
        valid = ranks[ranks > 0]
        mrrs[i] = float((1.0 / valid).mean())
    return float(np.quantile(mrrs, 0.95))


# ---------------------------------------------------------------------------
# Stability (P2b) — disjoint half-split on CRP-cropped v_c vectors
# ---------------------------------------------------------------------------

def compute_stability(refsets_h5: str, img_cache: dict,
                      clip_model, clip_preprocess, resnet_model, composite,
                      class_labels_fc: np.ndarray, component_ids_fc: np.ndarray,
                      top_idx_fc: np.ndarray,
                      half_k: int, seed: int,
                      relevance_pct: float, min_crop_px: int, device: str,
                      k_budget: int = 16) -> dict:
    """Compute disjoint-half cosine stability on fc components using CRP-cropped embeddings.

    Splits top-2*half_k reference images into two disjoint halves (A, B), computes
    CRP-cropped v_c for each half, and measures cosine(vA, vB).  half_k=8 (default for
    variant; k_budget=16 so 2*8=16 refs = entire top-k budget, split into 8+8).

    Args:
        k_budget: the P2a k value; enforced: 2*half_k <= k_budget to stay within the same
                  reference budget.  Raises if violated.

    Args:
        refsets_h5: path to refsets HDF5 (for top256 image indices)
        img_stream: ImageNetValStream for random-access image loading
        clip_model, clip_preprocess: frozen CLIP
        resnet_model, composite: ResNet-50 + zennit EpsilonGammaBox
        class_labels_fc: (n_fc,) int — ImageNet class per fc component (= fc unit index)
        component_ids_fc: (n_fc,) int — component IDs (for logging)
        top_idx_fc: (n_fc, 256) int — top-256 image global indices for each fc component
        half_k: how many images in each half (default 8 → uses top 2*8=16 refs)
        seed: for reproducible random half-split
        relevance_pct, min_crop_px: passed through to CRP crop
        device: torch device
    Returns:
        dict with median_cosine, passes (>= 0.5), ci_95, etc.
    """
    if 2 * half_k > k_budget:
        raise ValueError(
            f"half_k={half_k} requires 2*half_k={2*half_k} refs but k_budget={k_budget}. "
            "Reduce half_k so that stability uses at most the same refs as P2a."
        )
    rng = np.random.default_rng(seed)
    n_comp = len(class_labels_fc)
    two_k = 2 * half_k

    cosines = []   # only append valid (both halves non-zero) pairs
    tfm = imagenet_eval_transform(224)

    for ci in range(n_comp):
        ch_idx = int(class_labels_fc[ci])  # fc unit index = class label (verified: equal by construction)
        picks = top_idx_fc[ci, :two_k]    # top-two_k reference image global indices

        perm = rng.permutation(two_k)
        idx_a = perm[:half_k]
        idx_b = perm[half_k:]

        def embed_half(idx_list):
            imgs_crp = []
            for ii in idx_list:
                gidx = int(picks[ii])
                img_pil = img_cache.get(gidx)
                if img_pil is None:
                    continue
                x = tfm(img_pil).unsqueeze(0).to(device)
                try:
                    R = compute_relevance_map(resnet_model, composite, x, ch_idx)
                    crop = crop_to_relevance_bbox(img_pil, R,
                                                  percentile=relevance_pct,
                                                  min_px=min_crop_px)
                except Exception:
                    crop = img_pil  # fallback to full image
                imgs_crp.append(crop)
            if not imgs_crp:
                return None  # signal failure — both halves must succeed for a valid pair
            embs = clip_embed_images(clip_model, clip_preprocess, imgs_crp, device)
            v = embs.mean(axis=0)
            n = np.linalg.norm(v)
            if n < 1e-10:
                return None  # zero vector — exclude this pair
            return (v / n).astype(np.float32)

        va = embed_half(idx_a)
        vb = embed_half(idx_b)
        if va is None or vb is None:
            continue  # exclude invalid pairs from stability computation
        cosines.append(float(va @ vb))

    n_valid = len(cosines)
    if n_valid == 0:
        return {
            "half_k": half_k,
            "pool": "mean",
            "n_components": int(n_comp),
            "n_valid_pairs": 0,
            "median_cosine": float("nan"),
            "ci_95": [float("nan"), float("nan")],
            "tau_stable_threshold": 0.5,
            "passes": False,
            "error": "No valid component pairs found",
        }
    cosines_arr = np.array(cosines, dtype=np.float32)

    median = float(np.median(cosines_arr))
    from scipy.stats import bootstrap as bs
    res = bs((cosines_arr,), np.median, confidence_level=0.95, n_resamples=500,
             method="basic", random_state=np.random.default_rng(seed))
    ci = [float(res.confidence_interval.low), float(res.confidence_interval.high)]
    return {
        "half_k": half_k,
        "pool": "mean",
        "n_components": int(n_comp),
        "n_valid_pairs": n_valid,
        "n_excluded_pairs": int(n_comp) - n_valid,
        "median_cosine": median,
        "ci_95": ci,
        "tau_stable_threshold": 0.5,
        "passes": bool(median >= 0.5),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="CRP-compose method-swap variant for C2 verification")
    p.add_argument("--refsets", default="runs/M2_reference_sets/refsets.h5")
    p.add_argument("--text", default="runs/M5_text_embeddings/text_embeddings.h5")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--k", type=int, default=16,
                   help="Top-k reference images per component (same as main experiment)")
    p.add_argument("--n_permute", type=int, default=1000,
                   help="Permutation test samples (same as main experiment)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (same as main experiment)")
    p.add_argument("--relevance_pct", type=float, default=90.0,
                   help="Percentile threshold for positive relevance pixels (config: 90)")
    p.add_argument("--min_crop_px", type=int, default=32,
                   help="Minimum crop bbox side in original pixel space (config: 32)")
    p.add_argument("--half_k", type=int, default=8,
                   help="Half-split size for P2b stability (default 8 = half of k=16; 2*half_k must <= k)")
    p.add_argument("--skip_stability", action="store_true",
                   help="Skip P2b stability computation (reduces runtime ~2x)")
    args = p.parse_args()

    if not ZENNIT_AVAILABLE:
        print("[CRP-compose] ERROR: zennit not installed. Run: pip install zennit")
        sys.exit(1)

    if not args.skip_stability and 2 * args.half_k > args.k:
        print(f"[CRP-compose] ERROR: half_k={args.half_k} requires 2*half_k={2*args.half_k} "
              f"refs but k={args.k}. Use --half_k <= k//2 or --skip_stability.")
        sys.exit(1)

    set_all_seeds(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CRP-compose] device={device}")
    if torch.cuda.is_available():
        print(f"[CRP-compose] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
        print(f"[CRP-compose] available GPU count={torch.cuda.device_count()}")

    t0 = time.time()

    # --- Load ResNet-50 (same as main experiment: IMAGENET1K_V2 via load_resnet50_torchvision) ---
    print("[CRP-compose] loading ResNet-50 (IMAGENET1K_V2) ...")
    resnet = load_resnet50_torchvision(device)
    resnet.eval()

    # Build LRP composite once (shared across all component/image pairs)
    # EpsilonGammaBox low/high: per-channel bounds after ImageNet normalization
    # low = (0 - mean) / std;  high = (1 - mean) / std  (pixel range [0,1] normalized)
    mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std_t  = torch.tensor(IMAGENET_STD,  device=device).view(3, 1, 1)
    low_val  = ((torch.zeros(3, 1, 1, device=device) - mean_t) / std_t).tolist()
    high_val = ((torch.ones( 3, 1, 1, device=device) - mean_t) / std_t).tolist()
    composite = build_lrp_composite(resnet, device, low_val, high_val)
    print("[CRP-compose] LRP composite (EpsilonGammaBox epsilon=1e-6 gamma=0.25) ready")

    # --- Load frozen CLIP ViT-B/32 (same as main experiment) ---
    print("[CRP-compose] loading frozen CLIP ViT-B/32 ...")
    clip_model, clip_preprocess, _ = load_openai_clip(device)
    clip_model.eval()

    # --- Load reference sets ---
    print(f"[CRP-compose] loading refsets from {args.refsets}")
    with h5py.File(args.refsets, "r") as h5r:
        top_idx_all = h5r["refsets/top256_image_indices"][:]    # (n_comp, 256)
        component_id_all = h5r["components/component_id"][:]
        layer_all = h5r["components/layer"][:].astype("U16")
        local_idx_all = h5r["components/local_index"][:]
        class_lbl_all = h5r["components/class_label"][:]

    # --- Restrict to fc only (cost control; documented in config component_scope: fc_only) ---
    # For fc components: class_label == local_index (both = fc unit index = ImageNet class)
    # This equality is verified empirically: class_lbl_all[fc_mask][:5] == local_idx_all[fc_mask][:5]
    fc_mask = (layer_all == "fc")
    fc_indices = np.where(fc_mask)[0]
    fc_comp_ids = component_id_all[fc_mask]
    fc_class_lbls = class_lbl_all[fc_mask]   # = fc unit index = ImageNet class label
    fc_local_idx = local_idx_all[fc_mask]     # = same as class_lbl for fc by construction
    top_idx_fc = top_idx_all[fc_mask]         # (n_fc, 256)
    n_fc = int(fc_mask.sum())
    print(f"[CRP-compose] fc components: {n_fc}")
    # Sanity: verify class_label == local_index for fc (should always hold for ResNet-50)
    assert np.all(fc_class_lbls == fc_local_idx), \
        "FATAL: fc class_label != local_index (unexpected; fc unit i should = class i)"

    # --- Load text embeddings ---
    with h5py.File(args.text, "r") as h5t:
        text_emb = h5t["imagenet1k_classes/embeddings"][:]  # (1000, 512)
    print(f"[CRP-compose] text embeddings shape: {text_emb.shape}")

    # --- Image stream (random-access parquet reader) ---
    img_stream = ImageNetValStream()

    # --- Pre-load all unique images needed for the CRP loop ---
    # Random-access parquet I/O is the bottleneck (~1.6s/image individually).
    # Batch-loading by shard (get_images_batch groups by shard) is much faster.
    all_needed_indices = np.unique(top_idx_fc[:, :args.k].ravel())
    print(f"[CRP-compose] pre-loading {len(all_needed_indices)} unique images (shard-batched) ...")
    t_io = time.time()
    img_cache: dict[int, Image.Image] = {}
    BATCH_LOAD = 512
    for i in range(0, len(all_needed_indices), BATCH_LOAD):
        batch_idx = all_needed_indices[i:i + BATCH_LOAD].tolist()
        results = img_stream.get_images_batch(batch_idx)
        for j, (img_pil, _lbl) in enumerate(results):
            img_cache[batch_idx[j]] = img_pil
    print(f"[CRP-compose] image pre-load done: {len(img_cache)} cached  elapsed={time.time()-t_io:.0f}s")

    # --- Main CRP-compose loop: compute v_c^CRP for each fc component ---
    print(f"[CRP-compose] starting CRP-compose loop: {n_fc} fc components x {args.k} images ...")
    v_c_crp = np.zeros((n_fc, 512), dtype=np.float32)
    valid_comp = np.zeros(n_fc, dtype=bool)  # True iff component produced a non-zero v_c
    tfm = imagenet_eval_transform(224)

    for ci in range(n_fc):
        if ci % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [CRP-compose] component {ci}/{n_fc}  elapsed={elapsed:.0f}s", flush=True)

        # fc unit index = class label (verified above that they are equal)
        ch_idx = int(fc_class_lbls[ci])
        picks = top_idx_fc[ci, :args.k]   # top-k image global indices

        cropped_imgs = []
        for gidx in picks:
            img_pil = img_cache.get(int(gidx))
            if img_pil is None:
                print(f"    [warn] image {gidx} not in cache (load failed earlier)")
                continue

            # LRP relevance conditioned on fc unit ch_idx (one-hot seed in compute_relevance_map)
            x = tfm(img_pil).unsqueeze(0).to(device)
            try:
                R = compute_relevance_map(resnet, composite, x, ch_idx)
                crop = crop_to_relevance_bbox(img_pil, R, percentile=args.relevance_pct,
                                               min_px=args.min_crop_px)
            except Exception as e:
                print(f"    [warn] LRP/crop failed comp={ci} img={gidx}: {e}")
                crop = img_pil  # fallback to full image on per-image error

            cropped_imgs.append(crop)

        if not cropped_imgs:
            # All images failed to load — v_c_crp[ci] stays zero; not included in MRR
            continue

        # CLIP embed CRP-cropped images then mean-pool (same as main experiment but on crops)
        embs = clip_embed_images(clip_model, clip_preprocess, cropped_imgs, device, batch_size=32)
        v = embs.mean(axis=0)
        n = np.linalg.norm(v)
        if n > 0:
            v_c_crp[ci] = v / n
            valid_comp[ci] = True

    n_valid_comp = int(valid_comp.sum())
    print(f"[CRP-compose] CRP-compose loop done  elapsed={time.time()-t0:.0f}s  "
          f"valid_components={n_valid_comp}/{n_fc}")

    # --- Save v_c^CRP ---
    v_c_path = out_dir / "v_c_crp.h5"
    if v_c_path.exists():
        v_c_path.unlink()
    with h5py.File(v_c_path, "w") as h5o:
        h5o.create_dataset("v_c", data=v_c_crp)
        # HDF5 does not support NumPy Unicode dtype ('<U16'); encode as fixed-length bytes
        layer_bytes = np.array([s.encode("ascii") for s in layer_all[fc_mask]], dtype="S16")
        h5o.create_dataset("layer", data=layer_bytes)
        h5o.create_dataset("class_label", data=fc_class_lbls)
        h5o.create_dataset("component_id", data=fc_comp_ids)
        h5o.attrs["k"] = args.k
        h5o.attrs["pool"] = "mean"
        h5o.attrs["method"] = "clip_dissect_crp_compose"
        h5o.attrs["lrp_rule"] = "lrp_epsilon"
        h5o.attrs["relevance_pct"] = args.relevance_pct
        h5o.attrs["min_crop_px"] = args.min_crop_px
    print(f"[CRP-compose] saved v_c^CRP to {v_c_path}")

    # --- P2a: Text-query MRR (same metric as M8; exclude invalid/zero-vector components) ---
    print("[CRP-compose] P2a: computing MRR ...")
    print(f"[CRP-compose] valid_comp: {n_valid_comp}/{n_fc} (invalid excluded from MRR)")
    mrr, r10 = compute_mrr(v_c_crp, text_emb, fc_class_lbls, valid_mask=valid_comp)
    perm_upper = permutation_upper(v_c_crp, text_emb, fc_class_lbls,
                                    n_permute=args.n_permute, seed=args.seed,
                                    valid_mask=valid_comp)
    significant = bool(mrr > perm_upper)
    print(f"[CRP-compose] P2a: MRR={mrr:.4f} R@10={r10:.4f} perm95={perm_upper:.4f} sig={significant}")

    p2a = {
        "method": "clip_dissect_crp_compose",
        "k": args.k,
        "pool": "mean",
        "n_queries": int(len(fc_class_lbls)),
        "n_components": int(n_fc),
        "n_valid_components": n_valid_comp,
        "mrr": float(mrr),
        "recall_at_10": float(r10),
        "permutation_mrr_ci_95_upper": float(perm_upper),
        "significant": significant,
        "n_permute": args.n_permute,
        "lrp_rule": "lrp_epsilon",
        "relevance_crop_percentile": args.relevance_pct,
        "min_crop_px": args.min_crop_px,
    }
    (out_dir / "result_p2a.json").write_text(json.dumps(p2a, indent=2))

    main_mrr = 0.898
    delta_mrr = mrr - main_mrr
    print(f"[CRP-compose] vs. main: MRR delta={delta_mrr:+.4f} (main={main_mrr})")

    # --- P2b: Stability (optional) ---
    p2b = None
    if not args.skip_stability:
        print(f"[CRP-compose] P2b: stability (half_k={args.half_k}) ...")
        try:
            p2b = compute_stability(
                refsets_h5=args.refsets,
                img_cache=img_cache,
                clip_model=clip_model,
                clip_preprocess=clip_preprocess,
                resnet_model=resnet,
                composite=composite,
                class_labels_fc=fc_class_lbls,
                component_ids_fc=fc_comp_ids,
                top_idx_fc=top_idx_fc,
                half_k=args.half_k,
                seed=args.seed,
                relevance_pct=args.relevance_pct,
                min_crop_px=args.min_crop_px,
                device=device,
                k_budget=args.k,
            )
            p2b["method"] = "clip_dissect_crp_compose"
            print(f"[CRP-compose] P2b: median_cos={p2b['median_cosine']:.4f} passes={p2b['passes']}")
            (out_dir / "result_p2b.json").write_text(json.dumps(p2b, indent=2))
        except Exception as e:
            print(f"[warn] P2b failed: {e}")

    # --- Combined result.json ---
    result = {
        "variant": "method-swap-crp-compose",
        "claim": "C2",
        "method": "clip_dissect_crp_compose",
        "dimensions_swapped": ["method"],
        "main_experiment_method": "clip_dissect_plain_mean_pool",
        "k": args.k,
        "pool": "mean",
        "wall_time_s": float(time.time() - t0),
        "p2a": p2a,
        "p2b": p2b,
        "delta_mrr_vs_main": float(delta_mrr),
        "main_mrr": float(main_mrr),
        "conclusion_note": (
            "P2a MRR significant → C2 text-queryability holds under CRP-crop compose"
            if significant else
            "P2a MRR NOT significant → C2 text-queryability fails under CRP-crop compose"
        ),
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2))
    print(f"[CRP-compose] wrote result.json to {out_dir}")
    print(f"[CRP-compose] total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
