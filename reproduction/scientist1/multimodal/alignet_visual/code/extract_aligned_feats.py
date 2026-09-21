"""Extract aligned DINOv2-B features on either THINGS or ImageNet subsets."""
import argparse
import io
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
import paths
from distill_dinov2b import make_student, get_cls


class ThingsDs(Dataset):
    def __init__(self, ids, processor):
        self.ids = ids
        self.processor = processor

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        u = self.ids[i]
        img = Image.open(os.path.join(paths.THINGS_IMGS, u + ".jpg")).convert("RGB")
        x = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return x, 0, u


class ImnDs(Dataset):
    def __init__(self, split_file, processor, max_samples=-1):
        import pyarrow.parquet as pq
        with open(split_file) as f:
            wanted = set(l.strip() for l in f if l.strip())
        idx = pd.read_csv(paths.IMAGENET_INDEX, sep="\t", header=None,
                          names=["shard", "row", "label", "id"])
        idx = idx[idx["id"].isin(wanted)]
        if max_samples > 0:
            keep = list(idx["id"])[: max_samples]
            idx = idx[idx["id"].isin(set(keep))]
        self.records = []
        for shard, g in idx.groupby("shard"):
            pf = pq.read_table(os.path.join(paths.IMAGENET_VAL_DATA,
                                            f"train-{shard:05d}-of-00014.parquet"),
                                columns=["image", "label"])
            img_arr = pf.column("image").to_pylist()
            lab_arr = pf.column("label").to_pylist()
            for _, row in g.iterrows():
                self.records.append((img_arr[row["row"]]["bytes"], int(lab_arr[row["row"]]), row["id"]))
        self.processor = processor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        bts, lab, iid = self.records[i]
        img = Image.open(io.BytesIO(bts)).convert("RGB")
        x = self.processor(images=img, return_tensors="pt")["pixel_values"][0]
        return x, lab, iid


def collate(batch):
    x = torch.stack([b[0] for b in batch], dim=0)
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    ids = [b[2] for b in batch]
    return x, y, ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(paths.CACHE_DIR, "dinov2b_aligned.pt"))
    ap.add_argument("--target", choices=["things", "eval10k", "train20k"], required=True)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--out-proj", required=True,
                    help="output path for projected 512-d aligned features")
    ap.add_argument("--out-cls", required=True,
                    help="output path for raw student CLS 768-d features (post-tuning)")
    args = ap.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    student, _, proc = make_student(unfrozen_blocks=4)
    student.load_state_dict(ck["student_state"])
    student = student.to(device).eval()
    proj = nn.Sequential(
        nn.Linear(768, 1024),
        nn.GELU(),
        nn.Linear(1024, ck["proj_dim"]),
    ).to(device)
    proj.load_state_dict(ck["proj_state"])
    proj.eval()

    if args.target == "things":
        meta = pd.read_csv(paths.THINGS_META, sep="\t")
        ids_all = meta["uniqueID"].tolist()
        ids = [u for u in ids_all if os.path.exists(os.path.join(paths.THINGS_IMGS, u + ".jpg"))]
        ds = ThingsDs(ids, proc)
    elif args.target == "eval10k":
        ds = ImnDs(paths.IMAGENET_EVAL_IDS, proc)
    else:
        ds = ImnDs(paths.IMAGENET_TRAIN_IDS, proc, max_samples=20000)

    dl = DataLoader(ds, batch_size=args.bs, shuffle=False, num_workers=6,
                    collate_fn=collate, pin_memory=True)
    proj_feats, cls_feats, all_labs, all_ids = [], [], [], []
    with torch.no_grad():
        for x, y, u in dl:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                cls = get_cls(student, x)
                p = proj(cls)
            proj_feats.append(p.float().cpu().numpy())
            cls_feats.append(cls.float().cpu().numpy())
            all_labs.append(y.numpy())
            all_ids.extend(u)
    proj_arr = np.concatenate(proj_feats, axis=0)
    cls_arr = np.concatenate(cls_feats, axis=0)
    labs_arr = np.concatenate(all_labs, axis=0)
    np.savez(args.out_proj, feats=proj_arr, labels=labs_arr, ids=np.array(all_ids))
    np.savez(args.out_cls, feats=cls_arr, labels=labs_arr, ids=np.array(all_ids))
    print("saved proj:", args.out_proj, proj_arr.shape, "cls:", args.out_cls, cls_arr.shape)


if __name__ == "__main__":
    main()
