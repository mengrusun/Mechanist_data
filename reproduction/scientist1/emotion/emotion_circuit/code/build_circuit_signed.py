"""Circuit selection using SIGNED score (positive-only push toward emotion)."""
from __future__ import annotations
import argparse
import os
from pathlib import Path

import torch

from common import OUT_DIR, EMOTIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directions", required=True)
    ap.add_argument("--acts_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k_neurons", type=int, default=392)
    ap.add_argument("--k_heads", type=int, default=168)
    args = ap.parse_args()

    dir_obj = torch.load(args.directions, map_location="cpu")
    neuron_scores = dir_obj["neuron_scores"]
    head_scores = dir_obj["head_scores"]
    neu_mean_mlp = dir_obj["neu_mean_mlp"]
    neu_mean_attn = dir_obj["neu_mean_attn"]

    acts_dir = OUT_DIR / args.acts_dir if not os.path.isabs(args.acts_dir) else Path(args.acts_dir)

    circuits = {}
    for emo in EMOTIONS:
        if emo not in neuron_scores: continue
        n_scores = neuron_scores[emo]  # (L, d_int)  positive => push toward emo
        h_scores = head_scores[emo]    # (L, H)
        L, d_int = n_scores.shape
        _, H = h_scores.shape

        # Top-K by signed score (positive push toward emotion)
        n_flat = n_scores.reshape(-1)
        _, top_n_idx = torch.topk(n_flat, args.k_neurons)  # top K positive
        h_flat = h_scores.reshape(-1)
        _, top_h_idx = torch.topk(h_flat, args.k_heads)

        emo_acts = torch.load(acts_dir / f"acts_{emo}.pt", map_location="cpu")
        emo_mlp = emo_acts["mlp_int"].to(torch.float32).mean(0)
        emo_z = emo_acts["attn_z"].to(torch.float32).mean(0)

        delta_mlp = emo_mlp - neu_mean_mlp
        delta_z = emo_z - neu_mean_attn

        mlp_by_layer = {}
        for idx in top_n_idx.tolist():
            l = idx // d_int
            i = idx % d_int
            mlp_by_layer.setdefault(l, {})[i] = float(delta_mlp[l, i])

        head_by_layer = {}
        for idx in top_h_idx.tolist():
            l = idx // H
            h = idx % H
            head_by_layer.setdefault(l, {})[h] = delta_z[l, h].tolist()

        circuits[emo] = {
            "mlp_by_layer": mlp_by_layer,
            "head_by_layer": head_by_layer,
            "k_neurons": args.k_neurons,
            "k_heads": args.k_heads,
        }
        print(f"[{emo}] neurons across {len(mlp_by_layer)} layers | "
              f"heads across {len(head_by_layer)} layers")

    out_path = OUT_DIR / args.out if not os.path.isabs(args.out) else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(circuits, out_path)
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
