"""Generate figures from experiment results."""
import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_head_heatmap(npz_path, out_png, title="head effects", vmax=None):
    d = np.load(npz_path, allow_pickle=True)
    h = d["head_effects"]  # (L, H)
    m = d["mlp_effects"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6),
                             gridspec_kw={"width_ratios": [4, 1]})
    ax = axes[0]
    vmax = vmax or np.max(np.abs(h)) * 0.9
    im = ax.imshow(h, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    ax.set_title(f"{title} - attention heads")
    plt.colorbar(im, ax=ax)

    ax = axes[1]
    mmax = np.max(np.abs(m))
    ax.barh(np.arange(len(m)), m, color=["r" if v < 0 else "b" for v in m])
    ax.set_yticks(np.arange(len(m)))
    ax.set_ylabel("layer")
    ax.set_xlabel("normalized effect")
    ax.set_title("MLP effects")
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print("saved", out_png)


def plot_sparsity_curve(npz_path, out_png, top_k_max=64):
    d = np.load(npz_path)
    h = d["head_effects"]
    flat = h.ravel()
    idx = np.argsort(-np.abs(flat))
    sorted_abs = np.abs(flat[idx])

    fig, ax = plt.subplots(figsize=(8, 5))
    K = min(top_k_max, len(sorted_abs))
    ax.plot(np.arange(1, K + 1), sorted_abs[:K], marker="o")
    ax.set_xlabel("head rank")
    ax.set_ylabel("|normalized effect|")
    ax.set_title("head importance decay (sparsity)")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print("saved", out_png)


def plot_modularity(npz_path, out_png):
    d = np.load(npz_path)
    corrupt = d["corrupt_heads"]     # rule polarity flip
    fact = d["fact_flip_heads"]
    query = d["query_flip_heads"]
    n_layers, n_heads = corrupt.shape

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    vmax = max(np.abs(corrupt).max(), np.abs(fact).max(), np.abs(query).max()) * 0.9
    titles = ["rule_flip (=corrupt)", "fact_flip", "query_flip"]
    for ax, mat, title in zip(axes, (corrupt, fact, query), titles):
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("head")
        ax.set_ylabel("layer")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    plt.suptitle("Per-head effect by corruption axis (modularity)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print("saved", out_png)

    # Also plot pairwise correlation of head effects across corruption axes:
    # low correlation = modular, high correlation = entangled
    def corr(a, b):
        return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])

    print("Pairwise head-effect correlation:")
    print(f"  rule_flip vs fact_flip : {corr(corrupt, fact):+.3f}")
    print(f"  rule_flip vs query_flip: {corr(corrupt, query):+.3f}")
    print(f"  fact_flip vs query_flip: {corr(fact, query):+.3f}")


def plot_positional(npz_path, out_png):
    d = np.load(npz_path, allow_pickle=True)
    eff = d["effects"]                       # (n_comp, n_pos)
    top_heads = d["top_heads"]
    top_mlps = d["top_mlps"]
    pos_labels = list(d["position_labels"])

    labels = [f"L{L}H{H}" for _, L, H in top_heads] + [f"MLP{L}" for _, L in top_mlps]
    fig, ax = plt.subplots(figsize=(10, max(6, 0.35 * len(labels))))
    vmax = np.nanmax(np.abs(eff)) * 0.9
    im = ax.imshow(eff, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(pos_labels)))
    ax.set_xticklabels(pos_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Positional patching effect per component")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print("saved", out_png)


def plot_nec_suff(json_path, out_png):
    d = json.load(open(json_path))
    K = d["K"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    ax.plot(K, d["top_suff"], marker="o", label="top-K heads")
    ax.errorbar(K, d["rand_suff_mean"], yerr=d["rand_suff_std"], fmt="s-",
                label="random-K heads")
    ax.axhline(d["clean_ld"], color="g", linestyle="--", label="clean ld")
    ax.axhline(d["corrupt_ld"], color="r", linestyle="--", label="corrupt ld")
    ax.set_xlabel("K (# patched heads)")
    ax.set_ylabel("logit_diff after patching corrupt with clean-K")
    ax.set_title("Sufficiency: patch top-K into corrupt")
    ax.legend()
    ax.grid(True)

    ax = axes[1]
    ax.plot(K, d["top_nec"], marker="o", label="top-K heads")
    ax.errorbar(K, d["rand_nec_mean"], yerr=d["rand_nec_std"], fmt="s-",
                label="random-K heads")
    ax.axhline(d["clean_ld"], color="g", linestyle="--", label="clean ld")
    ax.axhline(d["corrupt_ld"], color="r", linestyle="--", label="corrupt ld")
    ax.set_xlabel("K (# ablated heads)")
    ax.set_ylabel("logit_diff after ablation on clean")
    ax.set_title("Necessity: knockout top-K on clean")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=120, bbox_inches="tight")
    plt.close()
    print("saved", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", required=True,
                    choices=["heatmap", "sparsity", "modularity", "nec_suff", "positional"])
    ap.add_argument("--in-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="")
    args = ap.parse_args()
    if args.which == "heatmap":
        plot_head_heatmap(args.in_file, args.out, title=args.title or "head effects")
    elif args.which == "sparsity":
        plot_sparsity_curve(args.in_file, args.out)
    elif args.which == "modularity":
        plot_modularity(args.in_file, args.out)
    elif args.which == "nec_suff":
        plot_nec_suff(args.in_file, args.out)
    elif args.which == "positional":
        plot_positional(args.in_file, args.out)


if __name__ == "__main__":
    main()
