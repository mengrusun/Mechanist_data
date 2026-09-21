```python
"""Aggregator script for InterPLM final paper figures."""

import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

BASELINE_NPY = "experiment_results/experiment_79cc3c848d9e47c697f58a8613b72e34_proc_395414/experiment_data.npy"
RESEARCH_NPY = "experiment_results/experiment_79cc3c848d9e47c697f58a8613b72e34_proc_395414/experiment_data.npy"


def _load(npy_path):
    try:
        return np.load(npy_path, allow_pickle=True).item()
    except Exception as e:
        print("[WARN] could not load", npy_path, ":", e)
        return {}


baseline_data = _load(BASELINE_NPY)
research_data = _load(RESEARCH_NPY)

DS_KEY = "swissprot_esm2_8m"
primary = research_data.get(DS_KEY, {}) or baseline_data.get(DS_KEY, {})
per_concept = primary.get("per_concept_f1", {}) or {}
concept_counts = primary.get("concept_counts", {}) or {}

F1_THRESH = 0.5
thr_label = "F1 = 0.50"

concept_names = list(per_concept.keys())
f1_neuron_arr = np.array([v.get("f1_neuron", 0.0) for v in per_concept.values()], dtype=float)
f1_sae_arr    = np.array([v.get("f1_sae",    0.0) for v in per_concept.values()], dtype=float)
n_pos_arr     = np.array([v.get("n_pos",     0)   for v in per_concept.values()], dtype=float)

print("[INFO] loaded", len(concept_names), "concepts from", DS_KEY)
print("[INFO] concept_counts:", concept_counts)


# FIGURE 1: Main headline (histogram, scatter, alignment counts)
try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    if len(f1_neuron_arr) and len(f1_sae_arr):
        vmax = max(f1_neuron_arr.max(), f1_sae_arr.max(), 1.0)
        bins = np.linspace(0, vmax, 25)
        ax.hist(f1_neuron_arr, bins=bins, alpha=0.6,
                label="Raw ESM-2 neurons", color="tab:blue", edgecolor="white")
        ax.hist(f1_sae_arr, bins=bins, alpha=0.6,
                label="SAE features", color="tab:orange", edgecolor="white")
    ax.axvline(F1_THRESH, color="red", ls="--", lw=1.2, label=thr_label)
    ax.set_xlabel("Best F1 per Swiss-Prot concept")
    ax.set_ylabel("Number of concepts")
    ax.set_title("(a) Per-Concept F1 Distribution")
    ax.legend(frameon=False)

    ax = axes[1]
    if len(f1_neuron_arr):
        ax.scatter(f1_neuron_arr, f1_sae_arr, alpha=0.55, s=28,
                   color="tab:purple", edgecolors="white", linewidths=0.5)
    lim = max(1e-3,
              float(f1_neuron_arr.max()) if len(f1_neuron_arr) else 1.0,
              float(f1_sae_arr.max())    if len(f1_sae_arr)    else 1.0)
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="y = x")
    ax.axhline(F1_THRESH, color="red", ls=":", alpha=0.6)
    ax.axvline(F1_THRESH, color="red", ls=":", alpha=0.6)
    ax.set_xlabel("Best F1 (Raw Neuron)")
    ax.set_ylabel("Best F1 (SAE Feature)")
    ax.set_title("(b) Per-Concept SAE vs Neuron F1")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[2]
    n_neuron = int(concept_counts.get("neuron", int((f1_neuron_arr >= F1_THRESH).sum())))
    n_sae    = int(concept_counts.get("sae",    int((f1_sae_arr    >= F1_THRESH).sum())))
    total    = int(concept_counts.get("total_evaluated", len(concept_names)))
    bars = ax.bar(["Raw Neurons", "SAE Features"], [n_neuron, n_sae],
                  color=["tab:blue", "tab:orange"], edgecolor="black")
    ymax = max(n_neuron, n_sae, 1)
    for b, c in zip(bars, [n_neuron, n_sae]):
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + 0.02 * ymax,
                str(c), ha="center", va="bottom",
                fontsize=12, fontweight="bold")
    ax.set_ylabel("Concepts with F1 >= 0.50")
    ax.set_title("(c) Aligned Concept Counts  (Total = " + str(total) + ")")
    ax.set_ylim(0, ymax * 1.25 + 1)

    fig.suptitle("Swiss-Prot x ESM-2-8M: SAE Features Recover More Biological Concepts Than Raw Neurons",
                 fontsize=15, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1-main-sae-vs-neuron.png"), bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("[ERR] Figure 1 failed:", e)
    plt.close("all")


# FIGURE 2: Top-20 concepts by SAE F1
try:
    if per_concept:
        items = sorted(per_concept.items(), key=lambda x: -x[1].get("f1_sae", 0.0))[:20]
        names    = [k[:45].replace("_", " ") for k, _ in items]
        sae_vals = [v.get("f1_sae", 0.0)    for _, v in items]
        neu_vals = [v.get("f1_neuron", 0.0) for _, v in items]
        y = np.arange(len(names))

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(y - 0.2, sae_vals, height=0.4, label="SAE feature",
                color="tab:orange", edgecolor="black", linewidth=0.4)
        ax.barh(y + 0.2, neu_vals, height=0.4, label="Raw neuron",
                color="tab:blue", edgecolor="black", linewidth=0.4)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel("Best F1")
        ax.set_title("Top 20 Swiss-Prot Concepts Ranked by SAE F1 (with Matched Neuron F1)")
        ax.axvline(F1_THRESH, color="red", ls="--", alpha=0.6, label=thr_label)
        ax.legend(frameon=False, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig2-top20-concepts.png"), bbox_inches="tight")
        plt.close(fig)
except Exception as e:
    print("[ERR] Figure 2 failed:", e)
    plt.close("all")


# FIGURE 3: F1 vs concept size + empirical CDF
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if len(n_pos_arr):
        mask = n_pos_arr > 0
        ax.scatter(n_pos_arr[mask], f1_sae_arr[mask], alpha=0.55, s=30,
                   color="tab:orange", label="SAE feature",
                   edgecolors="white", linewidths=0.5)
        ax.scatter(n_pos_arr[mask], f1_neuron_arr[mask], alpha=0.55, s=30,
                   color="tab:blue", label="Raw neuron",
                   edgecolors="white", linewidths=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("Positive residues per concept (log scale)")
    ax.set_ylabel("Best F1")
    ax.set_title("(a) F1 vs Concept Frequency")
    ax.axhline(F1_THRESH, color="red", ls="--", alpha=0.6, label=thr_label)
    ax.legend(frameon=False)

    ax = axes[1]
    if len(f1_neuron_arr):
        for arr, lbl, c in [(f1_neuron_arr, "Raw neuron", "tab:blue"),
                            (f1_sae_arr, "SAE feature", "tab:orange")]:
            s = np.sort(arr)
            cdf = np.arange(1, len(s) + 1) / len(s)
            ax.plot(s, cdf, lw=2.2, label=lbl, color=c)
    ax.axvline(F1_THRESH, color="red", ls="--", alpha=0.6, label=thr_label)
    ax.set_xlabel("Best F1 per concept")
    ax.set_ylabel("Cumulative fraction of concepts")
    ax.set_title("(b) Empirical CDF of Best F1")
    ax.legend(frameon=False, loc="lower right")

    fig.suptitle("Concept-Level Analysis: Frequency Dependence and Overall Recovery",
                 fontsize=15, y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3-concept-size-and-cdf.png"),
                bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("[ERR] Figure 3 failed:", e)
    plt.close("all")


# FIGURE 4: Threshold sweep + Delta F1
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    if len(f1_neuron_arr):
        thresholds = np.linspace(0.05, 0.95, 19)
        n_sae_t    = np.array([(f1_sae_arr    >= t).sum() for t in thresholds])
        n_neuron_t = np.array([(f1_neuron_arr >= t).sum() for t in thresholds])
        ax.plot(thresholds, n_sae_t, "-o", color="tab:orange",
                lw=2, label="SAE features")
        ax.plot(thresholds, n_neuron_t, "-s", color="tab:blue",
                lw=2, label="Raw neurons")
    ax.axvline(F1_THRESH, color="red", ls="--", alpha=0.5, label=thr_label)
    ax.set_xlabel("F1 threshold")
    ax.set_ylabel("Number of aligned concepts")
    ax.set_title("(a) Aligned Concepts vs F1 Threshold")
    ax.legend(frameon=False)

    ax = axes[