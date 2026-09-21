"""Post-process helper: given a cache produced by cache_teacher_features.py with
`feats_raw` only, add a `feats_head` column by projecting raw features through the
M1 teacher head.

This lets us split the two-teacher deploy: run the raw SigLIP cache in parallel with
M1 (huge speedup — SigLIP forward is the bottleneck), then add head-projected feats
in seconds once M1 finishes.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from fit_teacher_head import AlignmentHead


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True, help="path to h5 with feats_raw column")
    p.add_argument("--teacher_head", required=True, help="path to teacher_head.pt from M1")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    device = args.device

    head = AlignmentHead(dim=1152).to(device).eval()
    head.load_state_dict(torch.load(args.teacher_head, map_location=device))
    head = head.to(dtype=torch.float32)

    with h5py.File(args.cache, "r+") as h5:
        n = h5["feats_raw"].shape[0]
        if "feats_head" in h5:
            print(f"[warn] `feats_head` already present in {args.cache}; overwriting")
            del h5["feats_head"]
        feat_ds = h5.create_dataset("feats_head", shape=(n, 1152), dtype="f2")
        chunk = 512
        for i in range(0, n, chunk):
            raw = torch.from_numpy(np.asarray(h5["feats_raw"][i:i+chunk])).float().to(device)
            with torch.no_grad():
                proj = head(raw).cpu().numpy().astype(np.float16)
            feat_ds[i:i+chunk] = proj
        h5.attrs["head_ckpt"] = str(args.teacher_head)
    print(f"[done] added feats_head to {args.cache}")


if __name__ == "__main__":
    main()
