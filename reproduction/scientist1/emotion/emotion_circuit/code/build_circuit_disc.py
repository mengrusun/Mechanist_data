"""Discriminative circuit selection.

For each emotion e, we compute a per-component discriminative score:
   disc_e(l, i) = score_e(l, i) - max_{e' != e} score_{e'}(l, i)
Then take top-K_neurons and top-K_heads by disc_e.

This selects components that push toward emotion e MORE than toward
other emotions. Useful when many components are "generic emotion"
components that are not emotion-specific.
"""
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
    neuron_scores = dir_obj["neuron_scores"]  # emotion -> (L, d_int)
    head_scores = dir_obj["head_scores"]      # emotion -> (L, H)
    neu_mean_mlp = dir_obj["neu_mean_mlp"]    # (L, d_int)
    neu_mean_attn = dir_obj["neu_mean_attn"]  # (L, H, hd)

    acts_dir = OUT_DIR / args.acts_dir if not os.path.isabs(args.acts_dir) else Path(args.acts_dir)

    # Stack scores across emotions
    n_stack = torch.stack([neuron_scores[e] for e in EMOTIONS], dim=0)  # (E, L, d_int)
    h_stack = torch.stack([head_scores[e] for e in EMOTIONS], dim=0)    # (E, L, H)

    circuits = {}
    for ei, emo in enumerate(EMOTIONS):
        # Discriminative score: emotion's own alignment MINUS the maximum
        # alignment for any other emotion. Positive = emotion-specific push.
        others_n = torch.cat([n_stack[:ei], n_stack[ei+1:]], dim=0)  # (E-1, L, d_int)
        others_h = torch.cat([h_stack[:ei], h_stack[ei+1:]], dim=0)  # (E-1, L, H)

        # Absolute alignment for own vs max abs alignment for others
        my_n = n_stack[ei]                    # (L, d_int)
        max_o_n = others_n.abs().max(0).values  # (L, d_int)
        disc_n = my_n.abs() - max_o_n          # (L, d_int)

        my_h = h_stack[ei]                    # (L, H)
        max_o_h = others_h.abs().max(0).values  # (L, H)
        disc_h = my_h.abs() - max_o_h          # (L, H)

        # Top-K by disc score
        L, d_int = my_n.shape
        _, H = my_h.shape

        n_flat = disc_n.reshape(-1)
        _, top_n_idx = torch.topk(n_flat, args.k_neurons)
        h_flat = disc_h.reshape(-1)
        _, top_h_idx = torch.topk(h_flat, args.k_heads)

        # Load emotion means to compute deltas
        emo_acts = torch.load(acts_dir / f"acts_{emo}.pt", map_location="cpu")
        emo_mlp = emo_acts["mlp_int"].to(torch.float32).mean(0)  # (L, d_int)
        emo_z = emo_acts["attn_z"].to(torch.float32).mean(0)     # (L, H, hd)

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
