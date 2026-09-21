#!/usr/bin/env python3
"""M5: Location — linear probes + SVD-based direction extraction.

Loads runs/M2/act_<condition>.pt files, trains per-layer 6-way logistic probes
for emotion identity on 24 emotional conditions (excluding neutral and filler),
and computes the top-3 SVD directions of the (26 × d) mean-activation matrix
at the top-2 probe layers.

Writes:
  reports/M5_location.json
  reports/M5_directions.pt   # dict {layer: tensor[3, d_model]}
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


EMOTIONS = ["happiness", "sadness", "fear", "anger", "disgust", "surprise"]


def load_activations(dir_path: str) -> Dict[str, Dict[int, torch.Tensor]]:
    """Return {condition_id: {layer: tensor[n_items, d_model]}}."""
    out = {}
    for path in sorted(glob.glob(os.path.join(dir_path, "act_*.pt"))):
        try:
            d = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"  [skip] {path}: {e}")
            continue
        out[d["condition_id"]] = d["acts"]
    return out


def probe_layer(X: np.ndarray, y: np.ndarray, seeds: List[int]) -> Dict[str, float]:
    """Train logistic probe, return mean acc / std over seeds."""
    accs = []
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, random_state=s, stratify=y)
        # Reduce solver iterations for speed; multinomial default is now fine.
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", n_jobs=1)
        clf.fit(Xtr, ytr)
        accs.append(clf.score(Xte, yte))
    return {"mean_acc": float(np.mean(accs)), "std": float(np.std(accs))}


def probe_length_shuffled(X: np.ndarray, y: np.ndarray, prompt_lengths: np.ndarray,
                          seeds: List[int]) -> Dict[str, float]:
    """Length-controlled baseline: shuffle labels within length buckets."""
    accs = []
    if len(np.unique(prompt_lengths)) < 4:
        buckets = np.zeros_like(prompt_lengths)
    else:
        buckets = np.digitize(prompt_lengths, np.quantile(prompt_lengths, [0.25, 0.5, 0.75]))
    for s in seeds:
        rng = np.random.default_rng(s)
        y_shuf = y.copy()
        for b in np.unique(buckets):
            idx = np.where(buckets == b)[0]
            perm = rng.permutation(idx)
            y_shuf[idx] = y[perm]
        Xtr, Xte, ytr, yte = train_test_split(X, y_shuf, test_size=0.30, random_state=s, stratify=y_shuf)
        clf = LogisticRegression(max_iter=500, C=1.0, solver="lbfgs", n_jobs=1)
        clf.fit(Xtr, ytr)
        accs.append(clf.score(Xte, yte))
    return {"mean_acc": float(np.mean(accs)), "std": float(np.std(accs))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--activations_dir", default="runs/M2")
    ap.add_argument("--prefixes_json", default="data/prefixes/prefixes.json")
    ap.add_argument("--layers", default="0,4,8,12,16,20,24,28,32,36")
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--out", default="reports/M5_location.json")
    ap.add_argument("--directions_out", default="reports/M5_directions.pt")
    ap.add_argument("--n_top_directions", type=int, default=3)
    args = ap.parse_args()

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    # Load prefix metadata
    prefixes = json.load(open(args.prefixes_json))
    cond_meta = {r["condition_id"]: r for r in prefixes}

    # Load activations
    print(f"[load] activations from {args.activations_dir}")
    acts_by_cond = load_activations(args.activations_dir)
    print(f"[load] got {len(acts_by_cond)} conditions")

    # ---------- Per-layer probe -----------
    # For each layer, stack (24 emotional conditions × n_items) rows with emotion label.
    # Skip neutral, filler.
    per_layer_results = {}
    for L in layers:
        X_rows = []
        y_rows = []
        length_rows = []
        for cid, layer_acts in acts_by_cond.items():
            meta = cond_meta.get(cid)
            if meta is None or meta["emotion"] not in EMOTIONS:
                continue
            if L not in layer_acts:
                continue
            arr = layer_acts[L].float().numpy()  # [n_items, d]
            n_items = arr.shape[0]
            X_rows.append(arr)
            y_rows.extend([EMOTIONS.index(meta["emotion"])] * n_items)
            length_rows.extend([meta.get("n_tokens_qwen", 0)] * n_items)
        if not X_rows:
            continue
        X = np.concatenate(X_rows, axis=0)
        y = np.array(y_rows)
        lens = np.array(length_rows)
        probe_res = probe_layer(X, y, list(range(42, 42 + args.n_seeds)))
        null_res = probe_length_shuffled(X, y, lens, list(range(42, 42 + args.n_seeds)))
        per_layer_results[L] = {
            "n_samples": int(X.shape[0]),
            "d_model": int(X.shape[1]),
            "probe_acc_mean": probe_res["mean_acc"],
            "probe_acc_std": probe_res["std"],
            "null_acc_mean": null_res["mean_acc"],
            "null_acc_std": null_res["std"],
        }
        print(f"  [layer {L}] probe={probe_res['mean_acc']:.4f}  null={null_res['mean_acc']:.4f}")

    # ---------- Top-2 layers -----------
    sorted_layers = sorted(per_layer_results.items(),
                           key=lambda kv: -kv[1]["probe_acc_mean"])
    top2_layers = [L for L, _ in sorted_layers[:2]]
    print(f"[top-2] layers = {top2_layers}")

    # ---------- SVD directions at top-2 layers -----------
    # Mean-activation matrix: (26 conditions × d)
    directions = {}
    for L in top2_layers:
        cond_order = []
        rows = []
        for cid, layer_acts in acts_by_cond.items():
            if L not in layer_acts:
                continue
            cond_order.append(cid)
            rows.append(layer_acts[L].float().mean(dim=0).numpy())
        if not rows:
            continue
        M = np.stack(rows, axis=0)  # [26, d]
        # Center by neutral if present
        if "neutral" in cond_order:
            neutral_idx = cond_order.index("neutral")
            M_centered = M - M[neutral_idx:neutral_idx+1, :]
        else:
            M_centered = M - M.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(M_centered, full_matrices=False)
        # top-k right-singular vectors as directions
        dirs = Vt[:args.n_top_directions, :]  # [k, d]
        directions[L] = {
            "conditions": cond_order,
            "singular_values": S.tolist(),
            "explained_variance_ratio": (S ** 2 / (S ** 2).sum()).tolist(),
            "directions": dirs,  # numpy [k, d]
        }
        print(f"  [SVD layer {L}] top singular values = {S[:5].tolist()}")

    # ---------- Write -----------
    result = {
        "per_layer": per_layer_results,
        "top2_layers": top2_layers,
        "top2_layer_ranking": [(L, per_layer_results[L]["probe_acc_mean"]) for L in top2_layers],
        "svd_meta": {L: {"conditions": directions[L]["conditions"],
                         "singular_values": directions[L]["singular_values"],
                         "explained_variance_ratio": directions[L]["explained_variance_ratio"]}
                     for L in directions},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[write] {args.out}")

    # Save directions as tensors
    tosave = {L: torch.from_numpy(directions[L]["directions"]).float()
              for L in directions}
    torch.save({"directions": tosave,
                "top2_layers": top2_layers,
                "conditions": {L: directions[L]["conditions"] for L in directions}},
               args.directions_out)
    print(f"[write] {args.directions_out}")


if __name__ == "__main__":
    main()
