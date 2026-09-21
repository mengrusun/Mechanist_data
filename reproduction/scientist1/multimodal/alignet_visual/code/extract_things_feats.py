"""Extract features on the 1854 THINGS concept images for teacher and multiple students.

Saves: cache/things_feats_<model>.npz with keys {feats, ids}
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
import paths


class ThingsImgDs(Dataset):
    def __init__(self, ids, img_dir, processor, mode):
        self.ids = ids
        self.img_dir = img_dir
        self.processor = processor
        self.mode = mode

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        uid = self.ids[i]
        p = os.path.join(self.img_dir, uid + ".jpg")
        img = Image.open(p).convert("RGB")
        if self.mode == "siglip":
            out = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        elif self.mode == "dinov2":
            out = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        elif self.mode == "vit":
            out = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        else:
            raise ValueError(self.mode)
        return out, uid


def load_model(kind, device):
    if kind == "siglip":
        from transformers import AutoModel, AutoImageProcessor
        proc = AutoImageProcessor.from_pretrained(paths.SIGLIP_PATH)
        model = AutoModel.from_pretrained(paths.SIGLIP_PATH).to(device).eval()
        return model, proc, "siglip"
    if kind == "dinov2-base":
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(paths.DINOV2_BASE)
        model = AutoModel.from_pretrained(paths.DINOV2_BASE).to(device).eval()
        return model, proc, "dinov2"
    if kind == "dinov2-small":
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(paths.DINOV2_SMALL)
        model = AutoModel.from_pretrained(paths.DINOV2_SMALL).to(device).eval()
        return model, proc, "dinov2"
    if kind == "vit-b":
        from transformers import AutoImageProcessor, AutoModel
        proc = AutoImageProcessor.from_pretrained(paths.VITB_PATH)
        model = AutoModel.from_pretrained(paths.VITB_PATH).to(device).eval()
        return model, proc, "vit"
    raise ValueError(kind)


@torch.no_grad()
def encode(model, kind_norm, x):
    if kind_norm == "siglip":
        # SigLIP model wrapper — use vision_model directly for pooled image embedding
        vout = model.vision_model(pixel_values=x)
        # take pooler_output if available else mean-pool last_hidden_state
        if hasattr(vout, "pooler_output") and vout.pooler_output is not None:
            return vout.pooler_output
        return vout.last_hidden_state.mean(dim=1)
    if kind_norm == "dinov2":
        vout = model(pixel_values=x)
        # use CLS token (position 0)
        return vout.last_hidden_state[:, 0]
    if kind_norm == "vit":
        vout = model(pixel_values=x)
        return vout.last_hidden_state[:, 0]
    raise ValueError(kind_norm)


def collate(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    ids = [b[1] for b in batch]
    return imgs, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["siglip", "dinov2-base", "dinov2-small", "vit-b"], required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda"
    meta = pd.read_csv(paths.THINGS_META, sep="\t")
    ids_all = meta["uniqueID"].tolist()
    ids = [u for u in ids_all if os.path.exists(os.path.join(paths.THINGS_IMGS, u + ".jpg"))]
    print(f"Have {len(ids)}/{len(ids_all)} concept images")

    model, proc, kind_norm = load_model(args.model, device)
    ds = ThingsImgDs(ids, paths.THINGS_IMGS, proc, kind_norm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate)

    all_feats = []
    all_ids = []
    for x, u in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            f = encode(model, kind_norm, x)
        all_feats.append(f.float().cpu().numpy())
        all_ids.extend(u)
    feats = np.concatenate(all_feats, axis=0)
    print(args.model, "feats", feats.shape)

    if args.out is None:
        args.out = os.path.join(paths.CACHE_DIR, f"things_feats_{args.model}.npz")
    np.savez(args.out, feats=feats, ids=np.array(all_ids))
    print("saved", args.out)


if __name__ == "__main__":
    main()
