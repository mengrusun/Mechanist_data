"""Eval script for DINOv2 ViT-S variant — same as code/eval_student_similarity.py but
loads the ViT-S architecture from MODEL_DIR/dinov2-small instead of hardcoded dinov2-base.

This is the ONLY change vs the main experiment's eval_student_similarity.py:
- Line: `mp = MODEL_DIR / "dinov2-small"` instead of `MODEL_DIR / "dinov2-base"`
- For the unaligned baseline: student_ckpt = MODEL_DIR/dinov2-small (directory)
- For the aligned variant: student_ckpt = runs/verify/.../aligned/checkpoint.pt (file)
  → the .pt file is loaded into a dinov2-small architecture backbone

All other logic (THINGS dataset, triplet evaluation, Spearman computation, level bucketing)
is identical to code/eval_student_similarity.py.
"""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
import sys

# Add parent code directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent.parent / "code"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_utils import (
    load_things_concepts,
    load_triplets,
    ThingsImageDataset,
    seed_everything,
    MODEL_DIR,
)


def compute_vits_features(student_ckpt: str | None, device: str, batch_size: int = 64):
    """Same as eval_student_similarity.compute_dinov2_features but for ViT-S.

    Key difference: loads architecture from MODEL_DIR/dinov2-small (hidden=384)
    instead of MODEL_DIR/dinov2-base (hidden=768).
    """
    from transformers import AutoModel, AutoImageProcessor

    # ViT-S backbone (384-dim)
    mp = MODEL_DIR / "dinov2-small"
    model = AutoModel.from_pretrained(str(mp), torch_dtype=torch.float16).to(device).eval()

    if student_ckpt and Path(student_ckpt).is_file():
        print(f"[load] loading aligned ViT-S ckpt from {student_ckpt}")
        state = torch.load(student_ckpt, map_location="cpu")
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(f"  [warn] missing={len(missing)} unexpected={len(unexpected)} keys")
    else:
        print(f"[load] using stock DINOv2 ViT-S (unaligned baseline); student_ckpt={student_ckpt!r}")

    model = model.to(device, dtype=torch.float16).eval()
    # DINOv2 ViT-S uses same preprocessing as ViT-B (shortest-edge=256, center-crop=224, ImageNet norm)
    # Load processor from dinov2-small (or dinov2-base — same config)
    proc = AutoImageProcessor.from_pretrained(str(mp))

    ds = ThingsImageDataset(proc)
    dl = DataLoader(ds, batch_size=batch_size, num_workers=4, shuffle=False)
    feats = []
    cids = []
    with torch.no_grad():
        for batch in tqdm(dl, desc="dinov2-small forward"):
            pv = batch["pixel_values"].to(device, dtype=torch.float16)
            out = model(pv)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                pool = out.pooler_output.float().cpu()
            else:
                pool = out.last_hidden_state[:, 0].float().cpu()
            feats.append(pool)
            cids.extend(batch["concept_id"].tolist())
    return torch.cat(feats, dim=0), cids


# Import all evaluation helpers from the main eval script
from eval_student_similarity import (
    bootstrap_ci95,
    triplet_predict_ooo,
    build_human_similarity_from_triplets,
    spearman_pairwise,
    level_bucket_triplets,
    eval_features,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student_ckpt", required=True,
                   help="Path to ViT-S checkpoint .pt file (aligned), OR MODEL_DIR/dinov2-small (unaligned baseline).")
    p.add_argument("--split", default="heldout")
    p.add_argument("--output_json", required=True)
    p.add_argument("--levels_construction", default="things_metadata",
                   choices=["things_metadata"])
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)

    # 1) DINOv2 ViT-S features on THINGS
    feats, cids = compute_vits_features(args.student_ckpt, device=device)
    print(f"[feat] shape={feats.shape}")

    # 2) Load eval triplets
    tri_eval = load_triplets(args.split)
    print(f"[eval] {len(tri_eval)} triplets from '{args.split}'")

    # 3) Build level buckets (same as main experiment)
    meta = load_things_concepts()
    buckets = level_bucket_triplets(tri_eval, meta)
    print("[bucket] " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))

    # 4) Evaluate (identical to main experiment eval)
    res = eval_features(feats=feats, cids=cids, triplets_eval=tri_eval,
                        device=device, level_buckets=buckets)
    res["config"] = {k: v for k, v in vars(args).items()}
    res["milestone_kind"] = "student_similarity_eval_vits_variant"

    with open(args.output_json, "w") as f:
        json.dump(res, f, indent=2)
    print("[done]", args.output_json)
    printable = {k: v for k, v in res.items() if k != "config"}
    print(json.dumps(printable, indent=2, default=str))


if __name__ == "__main__":
    main()
