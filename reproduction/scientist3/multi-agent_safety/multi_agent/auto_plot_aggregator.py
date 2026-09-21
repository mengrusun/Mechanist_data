```python
"""Aggregator script for final paper figures."""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300,
})

EXP_DIR = "experiment_results/experiment_b42b8053fd314d73985b1e63c9ec4edf_proc_1381729"
EXP_DATA_PATH = os.path.join(EXP_DIR, "experiment_data.npy")
ACTIVATIONS_PATH = os.path.join(EXP_DIR, "activations.npy")
LABELS_PATH = os.path.join(EXP_DIR, "labels.npy")
if not os.path.exists(EXP_DATA_PATH):
    alt = os.path.join("working", "experiment_data.npy")
    if os.path.exists(alt): EXP_DATA_PATH = alt

try:
    experiment_data = np.load(EXP_DATA_PATH, allow_pickle=True).item()
except Exception as e:
    print(f"load exp fail: {e}"); experiment_data = {}
try:
    activations = np.load(ACTIVATIONS_PATH, allow_pickle=True)
except Exception as e:
    print(f"load act fail: {e}"); activations = None
try:
    labels_arr = np.load(LABELS_PATH, allow_pickle=True)
except Exception as e:
    print(f"load lbl fail: {e}"); labels_arr = None

DATASET = "NARCBench-Core-Synthetic"
DATASET_PRETTY = "NARCBench Core (Synthetic)"

ds = None; summary = {}; all_results = {}
variants = []; MAX_ITER_GRID = []; best_mi = None; gt_labels = None
try:
    ds = experiment_data["max_iter_tuning"][DATASET]
    summary = ds["summary"]
    MAX_ITER_GRID = list(ds["hyperparams"]["max_iter_grid"])
    all_results = {int(mi): res for mi, res in summary["all_results"].items()}
    variants = list(next(iter(all_results.values())).keys())
    best_mi = int(summary["best_max_iter"])
    gt_labels = np.array(ds["ground_truth"])
except Exception as e:
    print(f"struct fail: {e}")

def pretty(name): return str(name).replace("_", " ")

def classify(name):
    n = str(name).lower()
    if "tfidf" in n or "tf-idf" in n or "text" in n or "judge" in n: return "Text baseline"
    if "random" in n: return "Random baseline"
    if any(k in n for k in ["mean","max","min","concat","sum","group","pool","agg","std","diff","var"]):
        return "Group-aggregated probe"
    if "agent" in n: return "Per-agent probe"
    return "Per-agent probe"

CAT_COLORS = {"Text baseline": "#94a3b8", "Random baseline": "#cbd5e1",
              "Per-agent probe": "#60a5fa", "Group-aggregated probe": "#ef4444"}

def roc_curve_simple(y, s):
    order = np.argsort(-s); y = y[order]
    P = max((y == 1).sum(), 1); N = max((y == 0).sum(), 1)
    tp = np.cumsum(y == 1); fp = np.cumsum(y == 0)
    tpr = np.concatenate([[0], tp / P]); fpr = np.concatenate([[0], fp / N])
    return fpr, tpr

# ------------- FIGURE 1: Headline -------------
try:
    if ds is not None:
        best_results = all_results[best_mi]
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax = axes[0]
        names_sorted = sorted(best_results.keys(), key=lambda k: best_results[k])
        vals = [best_results[k] for k in names_sorted]
        colors = [CAT_COLORS[classify(k)] for k in names_sorted]
        ax.barh([pretty(n) for n in names_sorted], vals, color=colors, edgecolor="black", linewidth=0.4)
        ax.axvline(0.5, color="k", linestyle="--", alpha=0.5)
        ax.set_xlim(0, 1.0); ax.set_xlabel("AUROC (5-fold CV)")
        ax.set_title(f"(a) All Probe Variants on {DATASET_PRETTY} (LR max iter = {best_mi})")
        ax.legend(handles=[Patch(color=c, label=k) for k, c in CAT_COLORS.items()], loc="lower right")

        ax = axes[1]
        headline = {
            "Random features": float(summary.get("random_auroc", np.nan)),
            "TF-IDF text judge": float(summary.get("tfidf_auroc", np.nan)),
            "Best per-agent probe": float(summary.get("best_agent_auroc", np.nan)),
            "Best group-aggregated probe": float(summary.get("best_group_auroc", np.nan)),
        }
        keys = list(headline.keys()); values = list(headline.values())
        bar_colors = ["#cbd5e1", "#94a3b8", "#60a5fa", "#ef4444"]
        bars = ax.bar(range(len(keys)), values, color=bar_colors, edgecolor="black")
        for b, v in zip(bars, values):
            if not np.isnan(v):
                ax.text(b.get_x()+b.get_width()/2, v+0.015, f"{v:.3f}", ha="center", va="bottom")
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.5, label="chance (0.5)")
        ax.set_xticks(range(len(keys)))
        ax.set_xticklabels(keys, rotation=15, ha="right")
        ax.set_ylim(0, 1.05); ax.set_ylabel("AUROC")
        ax.set_title("(b) Headline: Text-only Judge vs Activation Probes")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig("figures/fig1-headline-comparison.png", bbox_inches="tight")
        plt.close()
except Exception as e:
    print(f"Fig1 fail: {e}"); plt.close()

# ------------- FIGURE 2: Hyperparam sensitivity -------------
try:
    if ds is not None:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax = axes[0]
        style_map = {"Group-aggregated probe": ("-", "o"), "Per-agent probe": ("--", "s"),
                     "Text baseline": (":", "^"), "Random baseline": (":", "x")}
        for name in variants:
            ys = [all_results[mi][name] for mi in MAX_ITER_GRID]
            cat = classify(name); ls, mk = style_map[cat]
            ax.plot(MAX_ITER_GRID, ys, linestyle=ls, marker=mk,
                    color=CAT_COLORS[cat], label=pretty(name), alpha=0.9)
        ax.set_xscale("log")
        ax.set_xlabel("Logistic-Regression max iterations (log scale)")
        ax.set_ylabel("AUROC (5-fold CV)")
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.4, label="chance (0.5)")
        ax.set_title("(a) AUROC vs LR Max Iterations by Probe Variant")
        ax.legend(fontsize=8, ncol=2, loc="lower right")

        ax = axes[1]
        mat = np.array([[all_results[mi][v] for mi in MAX_ITER_GRID] for v in variants])
        im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.4, vmax=1.0)
        ax.set_xticks(range(len(MAX_ITER_GRID))); ax.set_xticklabels(MAX_ITER_GRID)
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels([pretty(v) for v in variants], fontsize=9)
        ax.set_xlabel("max iterations")
        ax.set_title("(b) AUROC Heatmap (variant x max iterations)")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        color="white" if mat[i, j] < 0.75 else "black", fontsize=8)
        plt.colorbar(im, ax=ax, label="AUROC")
        plt.tight_layout()
        plt.savefig("figures/fig2-hyperparam-sensitivity.png", bbox_inches="tight")
        plt.close()
except Exception as e:
    print(f"Fig2 fail: {e}"); plt.close()

# ------------- FIGURE 3: CV stability -------------
try:
    if ds is not None:
        run = ds["runs"][str(best_mi)]
        folds = run["auroc_folds"]
        vl = list(folds.keys())
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax = axes[0]
        data = [folds[v] for v in vl]
        bp = ax.boxplot(data, labels=[pretty(v) for v in vl],
                        patch_artist=True, medianprops=dict(color="black"))
        for patch, v in zip(bp["boxes"], vl):
            patch.set_facecolor(CAT_COLORS[classify(v)]); patch.set_alpha(0.85)
        for i, v in enumerate(vl):
            rng = np.random.default_rng(seed=i)
            xs = rng.normal(i+1, 0.05, size=len(folds[v]))
            ax.scatter(xs, folds[v], s=18, color="black", alpha=0.7, zorder=3)
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.4, label="chance (0.5)")
        ax.set_ylabel("AUROC (per fold)"); ax.set_ylim(0.3, 1.02)
        for lbl in ax.get_xticklabels(): lbl.set_rotation(35); lbl.set_ha("right")
        ax.set_title(f"(a) Per-fold AUROC Distribution (LR max iter = {best_mi})")
        ax.legend(loc="lower right")

        ax = axes[1]
        means = np.array([np.mean(folds[v]) for v in vl])
        stds = np.array([np.std(folds[v]) for v in vl])
        order = np.argsort(-means)
        xs = np.arange(len(vl))
        colors = [CAT_COLORS[classify(vl[i])] for i in order]
        ax.bar(xs, means[order], yerr=stds[order], capsize=4, color=colors, edgecolor="black")
        ax.set_xticks(xs)
        ax.set_xticklabels([pretty(vl[i]) for i in order], rotation=35, ha="right")
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.4, label="chance (0.5)")