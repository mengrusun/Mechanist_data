"""
Aggregator script for final paper figures on ESMFold beta-hairpin mechanism study.
Loads results from the experiment .npy file and produces a concise set of
publication-ready plots in the 'figures/' directory.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

NPY_PATH = "experiment_results/experiment_aebd1b506a874b4b96e009fbbf0a64b2_proc_631077/experiment_data.npy"

experiment_data = {}
try:
    experiment_data = np.load(NPY_PATH, allow_pickle=True).item()
    print(f"Loaded experiment data from {NPY_PATH}")
except Exception as e:
    print(f"Could not load {NPY_PATH}: {e}")
    experiment_data = {}

ds = experiment_data.get("esmfold_hairpin_panel", {})
per_prot = ds.get("per_protein", [])
val_metrics = ds.get("metrics", {}).get("val", [{}])

names   = [p["name"] for p in per_prot]
gts     = np.array([int(p["gt_hairpin"])   for p in per_prot]) if per_prot else np.array([])
preds   = np.array([int(p["pred_hairpin"]) for p in per_prot]) if per_prot else np.array([])
plddts  = np.array([p.get("plddt", np.nan) for p in per_prot]) if per_prot else np.array([])
correct = np.array([int(p["correct"])      for p in per_prot]) if per_prot else np.array([])
acc = (val_metrics[0].get("hairpin_dssp_accuracy", float("nan"))
       if val_metrics else float("nan"))
n_eval    = val_metrics[0].get("n_eval",    len(per_prot)) if val_metrics else len(per_prot)
n_correct = (val_metrics[0].get("n_correct", int(correct.sum()) if correct.size else 0)
             if val_metrics else (int(correct.sum()) if correct.size else 0))

print(f"Number of proteins on panel : {len(names)}")
print(f"Overall hairpin DSSP accuracy: {acc}")

# ---------------------------------------------------------------------------
# FIGURE 1: Baseline panel overview
# ---------------------------------------------------------------------------
try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

    ax = axes[0]
    if len(names):
        x = np.arange(len(names))
        ax.bar(x - 0.2, gts,   0.4, label="Ground truth (DSSP)", color="#3b6ea8")
        ax.bar(x + 0.2, preds, 0.4, label="ESMFold prediction",  color="#e08a2b")
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_ylabel("beta-hairpin present (0/1)")
        ax.set_ylim(-0.05, 1.15)
        ax.legend(loc="upper right", frameon=False)
    ax.set_title(f"(a) Per-protein hairpin call (accuracy = {acc:.2f})")

    ax = axes[1]
    if len(names):
        colors = ["#2ca02c" if c else "#d62728" for c in correct]
        ax.bar(np.arange(len(names)), plddts, color=colors)
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=35, ha="right")
        ax.set_ylabel("Mean pLDDT")
        ax.set_ylim(0, 100)
        legend_correct   = mpatches.Patch(color="#2ca02c", label="Correct call")
        legend_incorrect = mpatches.Patch(color="#d62728", label="Incorrect call")
        ax.legend(handles=[legend_correct, legend_incorrect], loc="lower right", frameon=False)
    ax.set_title("(b) Per-protein pLDDT confidence")

    ax = axes[2]
    cm = np.zeros((2, 2), dtype=int)
    for g, p in zip(gts, preds):
        cm[g, p] += 1
    im = ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=16, fontweight="bold")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: No", "Pred: Yes"])
    ax.set_yticklabels(["GT: No",   "GT: Yes"])
    ax.set_title(f"(c) Confusion matrix ({n_correct}/{n_eval} correct)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("ESMFold beta-hairpin panel: baseline (wild-type sequences)",
                 fontsize=15, y=1.03)
    fig.savefig(os.path.join(FIG_DIR, "fig1_panel_baseline.png"))
    plt.close(fig)
except Exception as e:
    print(f"[FIG1] error: {e}")
    plt.close("all")

# ---------------------------------------------------------------------------
# FIGURE 2: Accuracy summary and confidence separation
# ---------------------------------------------------------------------------
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    if not np.isnan(acc):
        ax.bar(["Accuracy", "Error (1 - acc)"], [acc, 1 - acc],
               color=["#2ca02c", "#d62728"])
        for i, v in enumerate([acc, 1 - acc]):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontweight="bold")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Fraction")
    ax.set_title(f"(a) Wild-type panel accuracy ({n_correct}/{n_eval})")

    ax = axes[1]
    if plddts.size:
        good = plddts[correct == 1]
        bad  = plddts[correct == 0]
        groups, labels = [], []
        if good.size:
            groups.append(good); labels.append("Correct")
        if bad.size:
            groups.append(bad);  labels.append("Incorrect")
        parts = ax.boxplot(groups, patch_artist=True, widths=0.5, showmeans=True)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        palette = ["#2ca02c", "#d62728"]
        for patch, c in zip(parts["boxes"], palette[:len(groups)]):
            patch.set_facecolor(c); patch.set_alpha(0.6)
        for i, arr in enumerate(groups):
            if arr.size:
                jitter = np.random.uniform(-0.08, 0.08, size=arr.size)
                ax.scatter(np.full_like(arr, i + 1, dtype=float) + jitter,
                           arr, color="k", s=25, zorder=3, label="_nolegend_")
    ax.set_ylabel("Mean pLDDT")
    ax.set_title("(b) Confidence stratified by correctness")

    fig.suptitle("Baseline predictive performance summary", fontsize=15, y=1.03)
    fig.savefig(os.path.join(FIG_DIR, "fig2_accuracy_confidence.png"))
    plt.close(fig)
except Exception as e:
    print(f"[FIG2] error: {e}")
    plt.close("all")

# ---------------------------------------------------------------------------
# APPENDIX FIGURE A1: Per-protein detail table and pLDDT-ranked bar chart
# ---------------------------------------------------------------------------
try:
    if len(names):
        fig, axes = plt.subplots(1, 2, figsize=(16, max(4, 0.5 * len(names) + 1.5)))

        ax = axes[0]
        cell_text = []
        for p in per_prot:
            cell_text.append([
                p["name"],
                "Yes" if p["gt_hairpin"]   else "No",
                "Yes" if p["pred_hairpin"] else "No",
                f"{p.get('plddt', float('nan')):.1f}",
                "correct" if p["correct"] else "wrong",
            ])
        col_labels = ["Protein", "GT hairpin", "Pred hairpin", "Mean pLDDT", "Result"]
        tab = ax.table(cellText=cell_text, colLabels=col_labels,
                       cellLoc="center", loc="center")
        tab.auto_set_font_size(False); tab.set_fontsize(11); tab.scale(1, 1.4)
        for r, p in enumerate(per_prot, start=1):
            color = "#d4f4d4" if p["correct"] else "#f4d4d4"
            for c in range(len(col_labels)):
                tab[(r, c)].set_facecolor(color)
        ax.axis("off")
        ax.set_title("(a) Per-protein baseline predictions")

        ax = axes[1]
        order = np.argsort(-plddts)
        colors = ["#2ca02c" if c else "#d62728" for c in correct[order]]
        ax.barh(np.arange(len(order)), plddts[order], color=colors)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(np.array(names)[order])
        ax.invert_yaxis()
        ax.set_xlabel("Mean pLDDT")
        ax.set_xlim(0, 100)
        legend_c = mpatches.Patch(color="#2ca02c", label="Correct hairpin call")
        legend_w = mpatches.Patch(color="#d62728", label="Incorrect hairpin call")
        ax.legend(handles=[legend_c, legend_w], loc="lower right", frameon=False)
        ax.set_title("(b) Proteins ranked by mean pLDDT")

        fig.suptitle("Appendix A1: Detailed per-protein baseline results",
                     fontsize=15, y=1.02)
        fig.savefig(os.path.join(FIG_DIR, "figA1_per_protein_detail.png"))
        plt.close(fig)
except Exception as e:
    print(f"[FIGA1] error: {e}")
    plt.close("all")

print("Done. Figures written to:", FIG_DIR)