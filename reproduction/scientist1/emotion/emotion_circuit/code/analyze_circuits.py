"""Analyze the extracted circuit properties for Claim 1 & 2:

- Layer distribution of circuit components (histogram)
- Overlap between emotion circuits (Jaccard on neurons/heads)
- Direction cosine similarity between emotions
- Statistics for the final report.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import torch

from common import OUT_DIR, EMOTIONS


def jaccard(a, b):
    A = set(a); B = set(b)
    return len(A & B) / max(len(A | B), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--circuits", required=True)
    ap.add_argument("--directions", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    circuits = torch.load(args.circuits, map_location="cpu")
    dirs = torch.load(args.directions, map_location="cpu")["emo_dir"]

    report = {}

    # Component layer distribution
    for emo in EMOTIONS:
        c = circuits.get(emo)
        if c is None: continue
        neu_layers = list(c["mlp_by_layer"].keys())
        head_layers = list(c["head_by_layer"].keys())
        report.setdefault("neuron_layers", {})[emo] = {
            "n_layers": len(neu_layers),
            "min": min(neu_layers), "max": max(neu_layers),
            "neurons_per_layer": {int(l): len(v) for l, v in c["mlp_by_layer"].items()},
        }
        report.setdefault("head_layers", {})[emo] = {
            "n_layers": len(head_layers),
            "min": min(head_layers), "max": max(head_layers),
            "heads_per_layer": {int(l): len(v) for l, v in c["head_by_layer"].items()},
        }

    # Overlap between emotion circuits (Jaccard of (layer, id) tuples)
    def flat_neu(c):
        s = []
        for l, m in c["mlp_by_layer"].items():
            for i in m:
                s.append((int(l), int(i)))
        return s

    def flat_head(c):
        s = []
        for l, m in c["head_by_layer"].items():
            for h in m:
                s.append((int(l), int(h)))
        return s

    neu_by_emo = {e: flat_neu(circuits[e]) for e in EMOTIONS if e in circuits}
    head_by_emo = {e: flat_head(circuits[e]) for e in EMOTIONS if e in circuits}

    j_neu = {}
    j_head = {}
    for e1 in EMOTIONS:
        if e1 not in circuits: continue
        for e2 in EMOTIONS:
            if e2 not in circuits: continue
            j_neu[f"{e1}|{e2}"] = round(jaccard(neu_by_emo[e1], neu_by_emo[e2]), 3)
            j_head[f"{e1}|{e2}"] = round(jaccard(head_by_emo[e1], head_by_emo[e2]), 3)
    report["jaccard_neurons"] = j_neu
    report["jaccard_heads"] = j_head

    # Direction cosine sim at several layers
    for layer_name, layer_idx in [("early", 6), ("mid", 14), ("late", 20), ("last", -1)]:
        cos = {}
        for e1 in EMOTIONS:
            if e1 not in dirs: continue
            v1 = dirs[e1][layer_idx].float()
            for e2 in EMOTIONS:
                if e2 not in dirs: continue
                v2 = dirs[e2][layer_idx].float()
                cos[f"{e1}|{e2}"] = round(float(
                    (v1 @ v2) / (v1.norm() * v2.norm() + 1e-8)
                ), 3)
        report[f"dir_cosine_{layer_name}"] = cos
    # For back-compat, alias 'last'
    report["dir_cosine_last"] = report["dir_cosine_last"]

    # Summarize
    n_layers_touched_mlp = {e: report["neuron_layers"][e]["n_layers"]
                            for e in EMOTIONS if e in report["neuron_layers"]}
    n_layers_touched_head = {e: report["head_layers"][e]["n_layers"]
                             for e in EMOTIONS if e in report["head_layers"]}
    report["summary"] = {
        "neuron_layer_coverage": n_layers_touched_mlp,
        "head_layer_coverage": n_layers_touched_head,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[write] {args.out}")

    # Print concise summary
    print("\n=== Layer coverage (# layers containing selected components) ===")
    for e in EMOTIONS:
        if e in n_layers_touched_mlp:
            print(f"  {e}: neurons across {n_layers_touched_mlp[e]}/28 layers, "
                  f"heads across {n_layers_touched_head[e]}/28 layers")
    print("\n=== Cross-emotion direction cosine (last residual layer) ===")
    for pair, v in cos.items():
        if pair.split("|")[0] < pair.split("|")[1]:
            print(f"  {pair}: {v}")


if __name__ == "__main__":
    main()
