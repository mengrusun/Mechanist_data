"""
Step 4: Plot per-layer probe metrics and cosine similarities. Also compute
transfer analysis (does the correctness probe generalize better than the
verbalized confidence?).
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_json", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--title_suffix", default="")
    args = ap.parse_args()

    with open(args.probe_json) as f:
        res = json.load(f)

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    for pos, per_layer in res["per_position"].items():
        layers = np.array([d["layer"] for d in per_layer])
        auc_A = np.array([d["probeA"]["auc"] for d in per_layer])
        auc_Chi = np.array([d["probeC_hi"]["auc"] for d in per_layer])
        r2_C = np.array([d["probeC_reg"]["r2"] for d in per_layer])
        cos_lr_reg = np.array([d["cos_logreg_reg"] for d in per_layer])
        cos_lr_hi = np.array([d["cos_logreg_hi"] for d in per_layer])
        cos_dm = np.array([d["cos_diffmeans"] for d in per_layer])

        # Fig 1: Probe strength per layer
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(layers, auc_A, marker="o", label="Correctness probe (AUC)", color="C0")
        ax1.plot(layers, auc_Chi, marker="s", label="Verb.-conf-high probe (AUC)", color="C1")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Probe AUC")
        ax1.set_ylim(0.4, 1.02)
        ax1.axhline(res["verbalized_calibration"]["auc_vs_correct"], ls="--", color="gray",
                    label=f"Verbalized conf → correct AUC={res['verbalized_calibration']['auc_vs_correct']:.2f}")
        ax1.grid(alpha=0.3)
        ax1.legend(loc="lower right")
        ax1.set_title(f"Linear probe AUC vs layer ({pos}) {args.title_suffix}")
        fig.tight_layout()
        fig.savefig(f"{args.out_dir}/probe_auc_{pos}.png", dpi=150)
        plt.close(fig)

        # Fig 2: Cosine similarities per layer
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.axhline(0, color="k", lw=0.5)
        ax.plot(layers, cos_lr_reg, marker="o", label="cos(w_correct, w_conf)  [LogReg vs Ridge]", color="C0")
        ax.plot(layers, cos_lr_hi, marker="s", label="cos(w_correct, w_conf>=100)  [both LogReg]", color="C1")
        ax.plot(layers, cos_dm, marker="^", label="cos(w_correct, w_conf>=100)  [diff-of-means]", color="C2")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Cosine similarity")
        ax.set_ylim(-1.05, 1.05)
        # Reference: random-direction expected |cos| in D=4096 is ~1/sqrt(D).
        # For a 4096-dim gaussian random unit pair, std of cos ~1/sqrt(D)=0.0156.
        ax.axhline(1/np.sqrt(4096), ls=":", color="gray")
        ax.axhline(-1/np.sqrt(4096), ls=":", color="gray")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)
        ax.set_title(f"Direction cosine similarity ({pos}) {args.title_suffix}")
        fig.tight_layout()
        fig.savefig(f"{args.out_dir}/cos_sim_{pos}.png", dpi=150)
        plt.close(fig)

    # Fig 3: Combined comparison — correctness AUC of best probe vs. verbalized confidence
    fig, ax = plt.subplots(figsize=(6, 4))
    labels, vals = [], []
    for pos, per_layer in res["per_position"].items():
        aucs = [d["probeA"]["auc"] for d in per_layer]
        labels.append(f"Best probe\n({pos})")
        vals.append(max(aucs))
    labels.append("Verbalized\nconfidence")
    vals.append(res["verbalized_calibration"]["auc_vs_correct"])
    colors = ["#3572B0", "#3572B0", "#B04835"]
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("AUC vs. correctness")
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.5, ls=":", color="gray")
    ax.set_title(f"Recovering correctness: internal probe vs. surfaced number {args.title_suffix}")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(f"{args.out_dir}/best_vs_verbalized.png", dpi=150)
    plt.close(fig)

    print("Saved figures to", args.out_dir)


if __name__ == "__main__":
    main()
