"""
M0_setup — Environment & data preparation (infrastructure, not phenomenon-validation).

Per EXPERIMENT_PLAN.md M0_setup:
  1. Verify env packages
  2. Compute SHA-256 checksums of CLIP + ResNet-50 checkpoints
  3. Sanity-check ResNet-50 top-1 accuracy on a 5000-image val subsample (pass: >= 0.76)
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

from common import (
    CLIP_LOCAL_PT, TORCHVISION_CACHE,
    ImageNetValStream, imagenet_eval_transform,
    load_resnet50_torchvision, load_openai_clip,
    load_imagenet_class_names, sha256_of_file, set_all_seeds,
)


def resnet50_top1_sanity(device: str, n_images: int = 5000, batch_size: int = 128) -> dict:
    """Run ResNet-50 IMAGENET1K_V2 on n_images samples, report top-1 accuracy."""
    model = load_resnet50_torchvision(device=device)
    tf = imagenet_eval_transform(crop=224)

    stream = ImageNetValStream()
    correct = 0
    seen = 0
    t0 = time.time()
    with torch.no_grad():
        for idx, labels, bytes_list in stream.iter_images(batch_size=batch_size):
            # Decode PIL and preprocess
            batch = torch.stack([tf(Image.open(io.BytesIO(b)).convert("RGB")) for b in bytes_list]).to(device)
            logits = model(batch)
            pred = logits.argmax(dim=1).cpu().numpy()
            correct += int((pred == labels).sum())
            seen += len(labels)
            if seen >= n_images:
                break
    elapsed = time.time() - t0
    top1 = correct / max(seen, 1)
    return {
        "n_images": seen,
        "top1": top1,
        "elapsed_sec": elapsed,
        "throughput_img_per_sec": seen / elapsed if elapsed > 0 else None,
    }


def clip_forward_sanity(device: str) -> dict:
    """One dummy image + one dummy text through CLIP — verifies preprocessing + tokenization."""
    model, preprocess, tokenizer = load_openai_clip(device=device)
    # Dummy image: a 224x224 gradient
    img = Image.new("RGB", (400, 300), color=(128, 64, 32))
    x = preprocess(img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = model.encode_image(x)
        text = tokenizer(["a photo of a dog"]).to(device)
        txt_emb = model.encode_text(text)
        img_norm = F.normalize(img_emb.float(), dim=-1)
        txt_norm = F.normalize(txt_emb.float(), dim=-1)
        cos = (img_norm @ txt_norm.T).item()
    return {
        "clip_image_dim": img_emb.shape[-1],
        "clip_text_dim": txt_emb.shape[-1],
        "dummy_cos": cos,
        "preprocess": str(preprocess),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="runs/M0_setup", type=str)
    p.add_argument("--n_sanity", type=int, default=5000)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--pass_top1", type=float, default=0.76)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_all_seeds(args.seed)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Step 1: env sanity
    env_report = {}
    try:
        import torch as _t
        import torchvision as _tv
        import clip as _clip
        import open_clip as _oc
        import h5py as _h5
        import pyarrow as _pa
        env_report = {
            "torch": _t.__version__,
            "torchvision": _tv.__version__,
            "clip": _clip.__version__ if hasattr(_clip, "__version__") else "openai-git",
            "open_clip": _oc.__version__,
            "h5py": _h5.__version__,
            "pyarrow": _pa.__version__,
            "cuda_available": _t.cuda.is_available(),
            "cuda_device_count": _t.cuda.device_count(),
        }
    except Exception as e:
        env_report = {"error": str(e)}
    (out / "env.json").write_text(json.dumps(env_report, indent=2))
    print("[M0] env:", json.dumps(env_report))

    # Step 2: checksums
    checksums = {}
    # torchvision resnet50 pth
    tv_ckpt = TORCHVISION_CACHE / "hub" / "checkpoints" / "resnet50-11ad3fa6.pth"
    if tv_ckpt.exists():
        checksums["resnet50_torchvision"] = {
            "path": str(tv_ckpt),
            "size_bytes": tv_ckpt.stat().st_size,
            "sha256": sha256_of_file(tv_ckpt),
        }
    else:
        checksums["resnet50_torchvision"] = {"error": f"missing: {tv_ckpt}"}
    # CLIP openai .pt
    if CLIP_LOCAL_PT.exists():
        checksums["clip_vit_b32_openai"] = {
            "path": str(CLIP_LOCAL_PT),
            "size_bytes": CLIP_LOCAL_PT.stat().st_size,
            "sha256": sha256_of_file(CLIP_LOCAL_PT),
        }
    else:
        checksums["clip_vit_b32_openai"] = {"error": f"missing: {CLIP_LOCAL_PT}"}
    (out / "checksums.json").write_text(json.dumps(checksums, indent=2))
    print("[M0] checksums written to", out / "checksums.json")

    # Step 3: CLIP forward sanity
    clip_san = clip_forward_sanity(args.device)
    (out / "clip_sanity.json").write_text(json.dumps(clip_san, indent=2))
    print("[M0] clip sanity:", json.dumps(clip_san))

    # Step 4: ResNet-50 top-1 sanity
    print(f"[M0] running ResNet-50 top-1 sanity on {args.n_sanity} images ...")
    r50 = resnet50_top1_sanity(args.device, n_images=args.n_sanity, batch_size=args.batch_size)
    r50["pass_threshold"] = args.pass_top1
    r50["passed"] = r50["top1"] >= args.pass_top1
    (out / "sanity_top1.json").write_text(json.dumps(r50, indent=2))
    print("[M0] resnet50 sanity:", json.dumps(r50))

    # ImageNet class names cache
    names = load_imagenet_class_names()
    (out / "imagenet_class_names.json").write_text(json.dumps(names, indent=2))

    # Exit code — non-zero if sanity failed
    if not r50["passed"]:
        print(f"[M0] FAILED: top1={r50['top1']:.3f} < {args.pass_top1}", file=sys.stderr)
        sys.exit(1)
    print("[M0] PASSED.")


if __name__ == "__main__":
    main()
