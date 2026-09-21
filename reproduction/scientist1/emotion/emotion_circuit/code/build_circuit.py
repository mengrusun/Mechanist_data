"""Assemble a per-emotion 'global circuit' from neuron and head scores.

Selection rule:
- Take top-K_neurons MLP neurons by |neuron_score| across all (layer, neuron)
- Take top-K_heads attention heads by |head_score| across all (layer, head)

Save circuit as a dict:
  {
    "mlp_neurons": [(layer, neuron_idx, sign, delta_val), ...],
    "attn_heads": [(layer, head_idx, sign, delta_vec_hd), ...],
    "mlp_delta_by_layer": {l: {neuron: delta_val}},
    "head_delta_by_layer": {l: {head: delta_vec_hd (hd,)}},
  }
where delta_val = mu_emo_mlp_int[l, i] - mu_neu_mlp_int[l, i]
and delta_vec_hd = mu_emo_z[l,h] - mu_neu_z[l,h]
Applied during inference as: add delta at the intervention point.

sign encodes the sign of the score (whether emotion push or pull).
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import torch

from common import OUT_DIR, EMOTIONS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--directions", required=True,
                    help="Path to directions.pt from analyze_directions.py")
    ap.add_argument("--acts_dir", required=True,
                    help="Where acts_{emotion}.pt live (for neu means)")
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

    circuits = {}
    for emo in EMOTIONS:
        if emo not in neuron_scores:
            continue
        n_scores = neuron_scores[emo]  # (L, d_int)
        h_scores = head_scores[emo]    # (L, H)
        L, d_int = n_scores.shape
        _, H = h_scores.shape

        # Top-K neurons (by absolute value)
        n_flat = n_scores.reshape(-1)
        top_n_vals, top_n_idx = torch.topk(n_flat.abs(), args.k_neurons)
        # Top-K heads
        h_flat = h_scores.reshape(-1)
        top_h_vals, top_h_idx = torch.topk(h_flat.abs(), args.k_heads)

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
        n_layers_touched_n = len(mlp_by_layer)
        n_layers_touched_h = len(head_by_layer)
        print(f"[{emo}] neurons: {args.k_neurons} across {n_layers_touched_n} layers | "
              f"heads: {args.k_heads} across {n_layers_touched_h} layers")

    out_path = OUT_DIR / args.out if not os.path.isabs(args.out) else Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(circuits, out_path)
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
