"""Apply the trained alignment head to SigLIP features on ImageNet subsets.

Produces cache/imagenet_<split>_teacher_aligned.npz with feats, labels, ids.
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
import paths
from fit_teacher import AlignHead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--siglip-feats", required=True)
    ap.add_argument("--head", default=os.path.join(paths.CACHE_DIR, "teacher_head.pt"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.siglip_feats, allow_pickle=True)
    feats = z["feats"].astype(np.float32)
    labels = z["labels"] if "labels" in z.files else None
    ids = z["ids"] if "ids" in z.files else None

    ck = torch.load(args.head, map_location="cpu")
    head = AlignHead(ck["in_dim"], out_dim=ck["out_dim"], hidden=ck["hidden"])
    head.load_state_dict(ck["state_dict"])
    head = head.cuda().eval()

    with torch.no_grad():
        x = torch.from_numpy(feats).float().cuda()
        y = head(x).cpu().numpy()

    save = dict(feats=y)
    if labels is not None: save["labels"] = labels
    if ids is not None: save["ids"] = ids
    np.savez(args.out, **save)
    print("saved", args.out, "shape", y.shape)


if __name__ == "__main__":
    main()
