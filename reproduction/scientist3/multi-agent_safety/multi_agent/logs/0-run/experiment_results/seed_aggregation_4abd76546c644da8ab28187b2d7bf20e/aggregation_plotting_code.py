import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_11-03-14_multi_agent_attempt_0/logs/0-run/experiment_results/experiment_fd8452d412774b9995864b6b5ecb1e19_proc_1415200/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        root = os.getenv("AI_SCIENTIST_ROOT", "")
        ed = np.load(os.path.join(root, p), allow_pickle=True).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

DATASET = "NARCBench-Core-Synthetic"

# Collect per-run structures
runs_ds = []
for ed in all_experiment_data:
    try:
        ds = ed["max_iter_tuning"][DATASET]
        runs_ds.append(ds)
    except Exception as e:
        print(f"Error extracting from a run: {e}")

n_runs = len(runs_ds)
print(f"Loaded {n_runs} experiment run(s).")

# Determine common structures from first run
try:
    ds0 = runs_ds[0]
    MAX_ITER_GRID = ds0["hyperparams"]["max_iter_grid"]
    variants = list(next(iter(ds0["summary"]["all_results"].values())).keys())
except Exception as e:
    print(f"Error extracting shared structure: {e}")
    MAX_ITER_GRID, variants = [], []


def sem(a, axis=0):
    a = np.asarray(a, dtype=float)
    n = a.shape[axis] if a.size else 1
    if n <= 1:
        return np.zeros(a.shape[1:] if a.ndim > 1 else ())
    return a.std(axis=axis, ddof=1) / np.sqrt(n)


# Build tensor: [n_runs, n_variants, n_max_iter] of AUROCs
try:
    T = np.full((n_runs, len(variants), len(MAX_ITER_GRID)), np.nan)
    for ri, ds in enumerate(runs_ds):
        all_results = {int(mi): res for mi, res in ds["summary"]["all_results"].items()}
        for vi, v in enumerate(variants):
            for mi_i, mi in enumerate(MAX_ITER_GRID):
                if mi in all_results and v in all_results[mi]:
                    T[ri, vi, mi_i] = all_results[mi][v]
    mean_T = np.nanmean(T, axis=0)  # [variants, max_iter]
    se_T = sem(T, axis=0) if n_runs > 1 else np.zeros_like(mean_T)
except Exception as e:
    print(f"Error building tensor: {e}")
    T = None

# Plot 1: AUROC vs max_iter per variant with mean ± SE across runs
try:
    fig, ax = plt.subplots(figsize=(9, 5))
    for vi, name in enumerate(variants):
        ys = mean_T[vi]
        es = se_T[vi]
        (line,) = ax.plot(MAX_ITER_GRID, ys, marker="o", label=name)
        if n_runs > 1:
            ax.fill_between(
                MAX_ITER_GRID, ys - es, ys + es, alpha=0.2, color=line.get_color()
            )
    ax.set_xscale("log")
    ax.set_xlabel("max_iter (log scale)")
    ax.set_ylabel("AUROC (5-fold CV, mean over runs)")
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.4, label="chance")
    ax.set_title(
        f"{DATASET}\nAUROC vs max_iter by Probe Variant (mean ± SE, n_runs={n_runs})"
    )
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{DATASET}_agg_auroc_vs_maxiter.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error plot1: {e}")
    plt.close()

# Determine best max_iter based on best mean across variants (pick per variant per max_iter, then max group aggregate over max_iter)
try:
    # use mean over runs, then choose max_iter that maximizes best variant mean AUROC
    best_variant_per_mi = np.nanmax(mean_T, axis=0)  # [max_iter]
    best_mi_idx = int(np.nanargmax(best_variant_per_mi))
    best_mi = MAX_ITER_GRID[best_mi_idx]
    print(f"Aggregated best max_iter: {best_mi}")
except Exception as e:
    print(f"Error determining best_mi: {e}")
    best_mi, best_mi_idx = None, None

# Plot 2: bar chart at best max_iter with mean ± SE across runs
try:
    means = mean_T[:, best_mi_idx]
    ses = se_T[:, best_mi_idx]
    order = np.argsort(-means)
    names_sorted = [variants[i] for i in order]
    vals = means[order]
    errs = ses[order]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        names_sorted,
        vals,
        yerr=errs if n_runs > 1 else None,
        color="steelblue",
        capsize=4,
        label=("mean ± SE across runs" if n_runs > 1 else "mean"),
    )
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.5, label="chance")
    ax.set_ylabel("AUROC (5-fold CV)")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"{DATASET}\nAUROC by Probe Variant at best max_iter={best_mi} (n_runs={n_runs})"
    )
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{DATASET}_agg_auroc_by_variant.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error plot2: {e}")
    plt.close()

# Plot 3: Per-fold aggregated: mean ± SE across folds (and runs) for best_mi
try:
    # Collect per-fold values across runs
    per_variant_folds = {v: [] for v in variants}
    for ds in runs_ds:
        run = ds["runs"][str(best_mi)]
        folds = run["auroc_folds"]
        for v in variants:
            per_variant_folds[v].extend(list(folds[v]))
    means_f = np.array([np.mean(per_variant_folds[v]) for v in variants])
    ses_f = np.array([sem(np.array(per_variant_folds[v])) for v in variants])
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(variants))
    ax.bar(
        x,
        means_f,
        yerr=ses_f,
        color="darkorange",
        capsize=4,
        label="mean ± SE across folds/runs",
    )
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.4, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=35, ha="right")
    ax.set_ylabel("AUROC")
    ax.set_title(
        f"{DATASET}\nPer-fold AUROC across variants (max_iter={best_mi}, aggregated)"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{DATASET}_agg_per_fold_auroc.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error plot3: {e}")
    plt.close()

# Plot 4: heatmap of mean AUROC across runs
try:
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mean_T, aspect="auto", cmap="viridis", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(MAX_ITER_GRID)))
    ax.set_xticklabels(MAX_ITER_GRID)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants)
    ax.set_xlabel("max_iter")
    ax.set_ylabel("Probe variant")
    ax.set_title(
        f"{DATASET}\nHeatmap: Mean AUROC by Variant and max_iter (n_runs={n_runs})"
    )
    for i in range(mean_T.shape[0]):
        for j in range(mean_T.shape[1]):
            val = mean_T[i, j]
            if n_runs > 1:
                txt = f"{val:.2f}\n±{se_T[i,j]:.02f}"
            else:
                txt = f"{val:.2f}"
            ax.text(
                j,
                i,
                txt,
                ha="center",
                va="center",
                color="white" if val < 0.75 else "black",
                fontsize=7,
            )
    plt.colorbar(im, ax=ax, label="Mean AUROC")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{DATASET}_agg_auroc_heatmap.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error plot4: {e}")
    plt.close()

# Plot 5: pooled score distribution by class for best variant at best_mi
try:
    # find best variant by aggregate mean at best_mi
    best_variant_idx = int(np.nanargmax(mean_T[:, best_mi_idx]))
    best_variant = variants[best_variant_idx]
    all_preds_0, all_preds_1 = [], []
    for ds in runs_ds:
        labels = np.array(ds["ground_truth"])
        preds = np.array(ds["runs"][str(best_mi)]["predictions"][best_variant])
        all_preds_0.extend(preds[labels == 0].tolist())
        all_preds_1.extend(preds[labels == 1].tolist())
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        all_preds_0, bins=20, alpha=0.6, label="Independent (label=0)", color="#94a3b8"
    )
    ax.hist(
        all_preds_1, bins=20, alpha=0.6, label="Collusion (label=1)", color="#ef4444"
    )
    ax.set_xlabel("Predicted collusion score (pooled over runs)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{DATASET}\nLeft: Independent, Right: Collusion — Probe={best_variant}, max_iter={best_mi} (n_runs={n_runs})"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{DATASET}_agg_score_distribution.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error plot5: {e}")
    plt.close()

# Print aggregated summary metrics
try:
    print("\n=== Aggregated summary metrics ===")
    metric_keys = [
        "best_group_auroc",
        "best_agent_auroc",
        "tfidf_auroc",
        "random_auroc",
        "collusion_detection_auroc",
    ]
    for k in metric_keys:
        vals = []
        for ds in runs_ds:
            v = ds.get("summary", {}).get(k, None)
            if v is not None:
                vals.append(float(v))
        if vals:
            m = np.mean(vals)
            s = sem(np.array(vals)) if len(vals) > 1 else 0.0
            print(f"{k}: mean={m:.4f}  SE={s:.4f}  (n={len(vals)})")
        else:
            print(f"{k}: not available")
    # Best variant at best_mi
    print(
        f"Best variant at aggregated best_mi={best_mi}: {variants[int(np.nanargmax(mean_T[:, best_mi_idx]))]} "
        f"(mean AUROC={mean_T[:, best_mi_idx].max():.4f})"
    )
except Exception as e:
    print(f"Error printing summary: {e}")
