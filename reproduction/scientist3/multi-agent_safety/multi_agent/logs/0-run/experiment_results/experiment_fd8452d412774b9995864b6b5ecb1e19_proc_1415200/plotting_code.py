import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

DATASET = "NARCBench-Core-Synthetic"
try:
    ds = experiment_data["max_iter_tuning"][DATASET]
    MAX_ITER_GRID = ds["hyperparams"]["max_iter_grid"]
    all_results = {int(mi): res for mi, res in ds["summary"]["all_results"].items()}
    variants = list(next(iter(all_results.values())).keys())
    summary = ds["summary"]
    best_mi = summary["best_max_iter"]
    labels = np.array(ds["ground_truth"])
except Exception as e:
    print(f"Error extracting structure: {e}")
    ds = None

# Plot 1: AUROC vs max_iter per variant
try:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name in variants:
        ys = [all_results[mi][name] for mi in MAX_ITER_GRID]
        ax.plot(MAX_ITER_GRID, ys, marker="o", label=name)
    ax.set_xscale("log")
    ax.set_xlabel("max_iter (log scale)")
    ax.set_ylabel("AUROC (5-fold CV)")
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.4, label="chance")
    ax.set_title(f"{DATASET}\nAUROC vs max_iter by Probe Variant")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{DATASET}_auroc_vs_maxiter.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error plot1: {e}")
    plt.close()

# Plot 2: bar chart at best max_iter
try:
    best_results = all_results[best_mi]
    names_sorted = sorted(best_results.keys(), key=lambda k: -best_results[k])
    vals = [best_results[k] for k in names_sorted]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(names_sorted, vals, color="steelblue")
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.5, label="chance")
    ax.set_ylabel("AUROC (5-fold CV)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{DATASET}\nAUROC by Probe Variant (best max_iter={best_mi})")
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{DATASET}_auroc_by_variant.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error plot2: {e}")
    plt.close()

# Plot 3: per-fold AUROCs for best variant at best max_iter
try:
    run = ds["runs"][str(best_mi)]
    best_variant = summary["best_group"]
    folds = run["auroc_folds"]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(variants))
    width = 0.15
    n_folds = len(next(iter(folds.values())))
    for fi in range(n_folds):
        ys = [folds[v][fi] for v in variants]
        ax.bar(x + fi * width, ys, width, label=f"Fold {fi+1}")
    ax.set_xticks(x + width * (n_folds - 1) / 2)
    ax.set_xticklabels(variants, rotation=35, ha="right")
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.4)
    ax.set_ylabel("AUROC")
    ax.set_title(f"{DATASET}\nPer-fold AUROC across variants (max_iter={best_mi})")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{DATASET}_per_fold_auroc.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error plot3: {e}")
    plt.close()

# Plot 4: heatmap of AUROC across variants and max_iter
try:
    mat = np.array([[all_results[mi][v] for mi in MAX_ITER_GRID] for v in variants])
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(MAX_ITER_GRID)))
    ax.set_xticklabels(MAX_ITER_GRID)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    ax.set_xlabel("max_iter")
    ax.set_ylabel("Probe variant")
    ax.set_title(f"{DATASET}\nHeatmap: AUROC by Variant and max_iter")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(
                j,
                i,
                f"{mat[i,j]:.2f}",
                ha="center",
                va="center",
                color="white" if mat[i, j] < 0.75 else "black",
                fontsize=8,
            )
    plt.colorbar(im, ax=ax, label="AUROC")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{DATASET}_auroc_heatmap.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error plot4: {e}")
    plt.close()

# Plot 5: score distribution by class for best variant at best max_iter
try:
    best_variant = summary["best_group"]
    preds = np.array(ds["runs"][str(best_mi)]["predictions"][best_variant])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        preds[labels == 0],
        bins=20,
        alpha=0.6,
        label="Independent (label=0)",
        color="#94a3b8",
    )
    ax.hist(
        preds[labels == 1],
        bins=20,
        alpha=0.6,
        label="Collusion (label=1)",
        color="#ef4444",
    )
    ax.set_xlabel("Predicted collusion score")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{DATASET}\nLeft: Independent, Right: Collusion — Probe={best_variant}, max_iter={best_mi}"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{DATASET}_score_distribution.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error plot5: {e}")
    plt.close()

try:
    print(f"Best max_iter: {summary['best_max_iter']}")
    print(
        f"Best group variant: {summary['best_group']} AUROC={summary['best_group_auroc']:.4f}"
    )
    print(
        f"Best per-agent: {summary['best_agent']} AUROC={summary['best_agent_auroc']:.4f}"
    )
    print(f"TF-IDF AUROC: {summary['tfidf_auroc']:.4f}")
    print(f"Random features AUROC: {summary['random_auroc']:.4f}")
    print(f"collusion_detection_auroc = {summary['collusion_detection_auroc']:.4f}")
except Exception as e:
    print(f"Error printing summary: {e}")
