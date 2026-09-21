"""Pass the eval10k subset through ResNet-50 / ViT-B/16 and CLIP.

For each image i, save:
- act_resnet[i, c]  = spatial-max activation of ResNet-50 layer4 channel c
- act_vit[i, n]     = last-layer ViT CLS token activation on neuron n
- clip_emb[i]       = normalized CLIP image embedding
- label[i]          = ImageNet class id

Outputs land under outputs/features/.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import torch.nn.functional as F
from transformers import CLIPModel, ResNetModel, ViTModel

from code.common import OUT_DIR, make_loader


def main(target: str = "both", batch: int = 128, workers: int = 8):
    device = "cuda"
    out = OUT_DIR / "features"
    out.mkdir(parents=True, exist_ok=True)

    loader = make_loader(subset="eval10k", batch=batch, workers=workers, dual=True)
    N = len(loader.dataset)
    print(f"[extract] N={N} images, batch={batch}")

    resnet = ResNetModel.from_pretrained("models/resnet-50").eval().to(device)
    vit = ViTModel.from_pretrained("models/vit-b16").eval().to(device)
    clip = CLIPModel.from_pretrained("models/clip").eval().to(device)

    # allocate on cpu
    act_resnet = np.zeros((N, 2048), dtype=np.float32)
    act_vit_cls = np.zeros((N, 768), dtype=np.float32)
    act_vit_mean = np.zeros((N, 768), dtype=np.float32)
    clip_dim = getattr(clip.config, "projection_dim", 512)
    clip_img = np.zeros((N, clip_dim), dtype=np.float32)
    labels = np.zeros((N,), dtype=np.int64)
    gids = np.zeros((N,), dtype=np.int64)

    t0 = time.time()
    idx = 0
    with torch.no_grad():
        for step, (imn, clp, y, gid) in enumerate(loader):
            imn = imn.to(device, non_blocking=True)
            clp = clp.to(device, non_blocking=True)
            bs = imn.size(0)

            # ResNet: use last hidden state (B, 2048, 7, 7) -> spatial max
            r = resnet(imn, output_hidden_states=False)
            feat = r.last_hidden_state  # (B, 2048, 7, 7)
            feat = feat.amax(dim=(-2, -1))  # (B, 2048)
            act_resnet[idx : idx + bs] = feat.float().cpu().numpy()

            # ViT: last hidden state -> take CLS + mean patch tokens
            v = vit(imn)
            h = v.last_hidden_state  # (B, 197, 768)
            act_vit_cls[idx : idx + bs] = h[:, 0].float().cpu().numpy()
            act_vit_mean[idx : idx + bs] = h[:, 1:].mean(dim=1).float().cpu().numpy()

            # CLIP image tower
            emb_out = clip.get_image_features(pixel_values=clp)
            emb = emb_out.pooler_output if hasattr(emb_out, "pooler_output") else emb_out
            emb = F.normalize(emb, dim=-1)
            clip_img[idx : idx + bs] = emb.float().cpu().numpy()

            labels[idx : idx + bs] = y.numpy()
            gids[idx : idx + bs] = gid.numpy()

            idx += bs
            if step % 20 == 0:
                dt = time.time() - t0
                print(f"[extract] step {step}/{len(loader)} idx={idx} elapsed={dt:.1f}s")

    np.savez_compressed(
        out / "eval10k.npz",
        act_resnet=act_resnet,
        act_vit_cls=act_vit_cls,
        act_vit_mean=act_vit_mean,
        clip_img=clip_img,
        labels=labels,
        gids=gids,
    )
    print(f"[extract] saved -> {out / 'eval10k.npz'}  elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
