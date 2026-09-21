"""
v_crp_recover.py — Recovery script for CRP-compose variant.

The expensive LRP loop (1000 fc components x k=16) already completed and
v_c_crp.h5 has v_c (1000, 512) float32.  This script:
1. Loads v_c_crp.h5 (reads only 'v_c' dataset — all valid, no zeros)
2. Loads refsets.h5 to recover fc_class_labels and fc_comp_ids
3. Loads text_embeddings.h5
4. Fixes v_c_crp.h5 by re-saving with corrected metadata (layer as bytes)
5. Runs P2a (MRR, R@10, permutation upper bound)
6. Runs P2b stability (CRP-cropped halves — reads from img_cache + refsets)
7. Writes result_p2a.json, result_p2b.json, result.json

This avoids re-running the 55-minute LRP loop.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import h5py
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    set_all_seeds, ImageNetValStream, load_openai_clip, load_resnet50_torchvision,
    imagenet_eval_transform, IMAGENET_MEAN, IMAGENET_STD
)

# Re-use functions from v_crp_compose
from v_crp_compose import (
    build_lrp_composite,
    compute_relevance_map,
    crop_to_relevance_bbox,
    clip_embed_images,
    compute_mrr,
    permutation_upper,
    compute_stability,
)


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--refsets", default="runs/M2_reference_sets/refsets.h5")
    p.add_argument("--text", default="runs/M5_text_embeddings/text_embeddings.h5")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--n_permute", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--relevance_pct", type=float, default=90.0)
    p.add_argument("--min_crop_px", type=int, default=32)
    p.add_argument("--half_k", type=int, default=8)
    p.add_argument("--skip_stability", action="store_true")
    args = p.parse_args()

    set_all_seeds(args.seed)
    out_dir = Path(args.out_dir)

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[recover] device={device}")
    print(f"[recover] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    # ---- Load v_c from existing HDF5 ----
    v_c_path = out_dir / "v_c_crp.h5"
    print(f"[recover] loading v_c from {v_c_path}")
    with h5py.File(v_c_path, "r") as h5:
        v_c_crp = h5["v_c"][:]  # (1000, 512) float32
    print(f"[recover] v_c shape: {v_c_crp.shape}")

    # ---- Load refsets to recover fc metadata ----
    print(f"[recover] loading refsets from {args.refsets}")
    with h5py.File(args.refsets, "r") as h5r:
        top_idx_all = h5r["refsets/top256_image_indices"][:]
        component_id_all = h5r["components/component_id"][:]
        layer_all = h5r["components/layer"][:].astype("U16")
        local_idx_all = h5r["components/local_index"][:]
        class_lbl_all = h5r["components/class_label"][:]

    fc_mask = (layer_all == "fc")
    fc_comp_ids = component_id_all[fc_mask]
    fc_class_lbls = class_lbl_all[fc_mask]
    fc_local_idx = local_idx_all[fc_mask]
    top_idx_fc = top_idx_all[fc_mask]
    n_fc = int(fc_mask.sum())
    print(f"[recover] fc components: {n_fc}")
    assert np.all(fc_class_lbls == fc_local_idx), "fc class_label != local_index"

    # ---- Fix v_c_crp.h5: re-save with layer as bytes ----
    print(f"[recover] re-saving v_c_crp.h5 with corrected metadata ...")
    v_c_path.unlink()
    with h5py.File(v_c_path, "w") as h5o:
        h5o.create_dataset("v_c", data=v_c_crp)
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
    print(f"[recover] v_c_crp.h5 re-saved OK")

    # ---- Load text embeddings ----
    with h5py.File(args.text, "r") as h5t:
        text_emb = h5t["imagenet1k_classes/embeddings"][:]
    print(f"[recover] text embeddings shape: {text_emb.shape}")

    # ---- P2a: valid_comp mask — components with non-zero v_c ----
    norms = np.linalg.norm(v_c_crp, axis=1)
    valid_comp = norms > 1e-10
    n_valid_comp = int(valid_comp.sum())
    print(f"[recover] valid_comp: {n_valid_comp}/{n_fc}")

    # ---- P2a: MRR ----
    print("[recover] P2a: computing MRR ...")
    mrr, r10 = compute_mrr(v_c_crp, text_emb, fc_class_lbls, valid_mask=valid_comp)
    perm_upper = permutation_upper(v_c_crp, text_emb, fc_class_lbls,
                                   n_permute=args.n_permute, seed=args.seed,
                                   valid_mask=valid_comp)
    significant = bool(mrr > perm_upper)
    print(f"[recover] P2a: MRR={mrr:.4f} R@10={r10:.4f} perm95={perm_upper:.4f} sig={significant}")

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
    print(f"[recover] wrote result_p2a.json")

    main_mrr = 0.898
    delta_mrr = mrr - main_mrr
    print(f"[recover] vs. main: MRR delta={delta_mrr:+.4f} (main={main_mrr})")

    # ---- P2b stability ----
    p2b = None
    if not args.skip_stability:
        print(f"[recover] P2b: stability (half_k={args.half_k}) ...")
        # Need models + img_cache for P2b
        print("[recover] loading ResNet-50 ...")
        resnet = load_resnet50_torchvision(device)
        resnet.eval()

        mean_t = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
        std_t  = torch.tensor(IMAGENET_STD,  device=device).view(3, 1, 1)
        low_val  = ((torch.zeros(3, 1, 1, device=device) - mean_t) / std_t).tolist()
        high_val = ((torch.ones( 3, 1, 1, device=device) - mean_t) / std_t).tolist()
        composite = build_lrp_composite(resnet, device, low_val, high_val)

        print("[recover] loading CLIP ViT-B/32 ...")
        clip_model, clip_preprocess, _ = load_openai_clip(device)
        clip_model.eval()

        # Pre-load images needed for P2b (top 2*half_k per fc component)
        two_k = 2 * args.half_k
        all_needed_indices = np.unique(top_idx_fc[:, :two_k].ravel())
        print(f"[recover] pre-loading {len(all_needed_indices)} images for P2b ...")
        img_stream = ImageNetValStream()
        img_cache: dict = {}
        BATCH_LOAD = 512
        t_io = time.time()
        for i in range(0, len(all_needed_indices), BATCH_LOAD):
            batch_idx = all_needed_indices[i:i + BATCH_LOAD].tolist()
            results = img_stream.get_images_batch(batch_idx)
            for j, (img_pil, _lbl) in enumerate(results):
                img_cache[batch_idx[j]] = img_pil
        print(f"[recover] images loaded: {len(img_cache)} in {time.time()-t_io:.0f}s")

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
            print(f"[recover] P2b: median_cos={p2b['median_cosine']:.4f} passes={p2b['passes']}")
            (out_dir / "result_p2b.json").write_text(json.dumps(p2b, indent=2))
            print(f"[recover] wrote result_p2b.json")
        except Exception as e:
            print(f"[warn] P2b failed: {e}")

    # ---- Combined result.json ----
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
    print(f"[recover] wrote result.json to {out_dir}")
    print(f"[recover] total wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
