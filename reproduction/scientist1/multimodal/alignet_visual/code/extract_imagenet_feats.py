"""Extract image features on an ImageNet-val subset from parquet files.

Saves cache/imagenet_<name>_feats_<model>.npz with feats, labels, image_ids.

The dataset directory holds 50k validation images in 14 parquet shards, each with
columns [image (dict with bytes,path), label]. A `global_index.tsv` maps rows to
short IDs; `train_subset_40k.txt` and `eval_subset_10k.txt` define two disjoint
splits.
"""
import argparse
import io
import os
import sys
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(__file__))
import paths


class ImgSubsetDs(Dataset):
    def __init__(self, records, processor, mode):
        self.records = records  # list of (image_bytes, label, id)
        self.processor = processor
        self.mode = mode

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


def load_records(split_file):
    """Load list of (image_bytes, label, id) from parquet using the specified id file."""
    import pyarrow.parquet as pq
    with open(split_file) as f:
        wanted = set(l.strip() for l in f if l.strip())
    # global_index tsv: shard_id, row_in_shard, label, id
    idx = pd.read_csv(paths.IMAGENET_INDEX, sep="\t", header=None,
                      names=["shard", "row", "label", "id"])
    idx = idx[idx["id"].isin(wanted)]
    # group by shard
    records = []
    for shard, g in idx.groupby("shard"):
        pf = pq.read_table(os.path.join(paths.IMAGENET_VAL_DATA, f"train-{shard:05d}-of-00014.parquet"),
                            columns=["image", "label"])
        rows_to_read = sorted(g["row"].tolist())
        label_arr = pf.column("label").to_pylist()
        img_arr = pf.column("image").to_pylist()  # list of dicts
        for r in rows_to_read:
            iid = g[g["row"] == r]["id"].iloc[0]
            bts = img_arr[r]["bytes"]
            lab = label_arr[r]
            records.append((bts, int(lab), iid))
    return records


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
        vout = model.vision_model(pixel_values=x)
        if hasattr(vout, "pooler_output") and vout.pooler_output is not None:
            return vout.pooler_output
        return vout.last_hidden_state.mean(dim=1)
    if kind_norm in ("dinov2", "vit"):
        vout = model(pixel_values=x)
        return vout.last_hidden_state[:, 0]
    raise ValueError(kind_norm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train40k", "eval10k"], required=True)
    ap.add_argument("--model", choices=["siglip", "dinov2-base", "dinov2-small", "vit-b"], required=True)
    ap.add_argument("--max-samples", type=int, default=-1,
                    help="if >0 take only first N ids (deterministic order from split file)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda"
    split_file = paths.IMAGENET_TRAIN_IDS if args.split == "train40k" else paths.IMAGENET_EVAL_IDS
    if args.max_samples > 0:
        # write a temp file with first N
        with open(split_file) as f:
            lines = [l.strip() for l in f if l.strip()]
        # first N deterministic
        sub = lines[: args.max_samples]
        tmp = os.path.join(paths.CACHE_DIR, f"_split_{args.split}_{args.max_samples}.txt")
        with open(tmp, "w") as f:
            f.write("\n".join(sub))
        split_file = tmp

    print("loading records from", split_file)
    records = load_records(split_file)
    print("num records:", len(records))

    model, proc, kind_norm = load_model(args.model, device)
    ds = ImgSubsetDs(records, proc, kind_norm)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=6, collate_fn=collate)

    all_feats, all_labs, all_ids = [], [], []
    total = 0
    for x, y, u in dl:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            f = encode(model, kind_norm, x)
        all_feats.append(f.float().cpu().numpy())
        all_labs.append(y.numpy())
        all_ids.extend(u)
        total += x.shape[0]
        if total % 1024 == 0:
            print(f"  encoded {total}/{len(ds)}")
    feats = np.concatenate(all_feats, axis=0)
    labels = np.concatenate(all_labs, axis=0)
    print(args.model, "feats", feats.shape, "labels", labels.shape)

    if args.out is None:
        args.out = os.path.join(paths.CACHE_DIR, f"imagenet_{args.split}_feats_{args.model}.npz")
    np.savez(args.out, feats=feats, labels=labels, ids=np.array(all_ids))
    print("saved", args.out)


if __name__ == "__main__":
    main()
