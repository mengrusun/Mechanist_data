import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2026-07-13_11-03-14_multi_agent_attempt_0/logs/0-run/experiment_results/experiment_0e1c3a98c6f147339ed81e706dd3e5c1_proc_1381729/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        ed = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")

DATASET = "NARCBench-Core-Synthetic"

# ----- Build aggregated data structures -----
agg = None
try:
    per_run = []
    for ed in all_experiment_data:
        ds = ed["max_iter_tuning"][DATASET]
        mi_grid = list(ds["hyperparams"]["max_iter_grid"])
        all_results = {int(mi): res for mi, res in ds["summary"]["all_results"].items()}
        per_run.append(
            {
                "mi_grid": mi_grid,
                "all_results": all_results,
                "summary": ds["summary"],
                "runs": ds["runs"],
                "labels": np.array(ds["ground_truth"]),
            }
        )
    # intersect max_iter grids
    common_mi = sorted(set.intersection(*[set(r["mi_grid"]) for r in per_run]))
    # intersect variants
    variant_sets = []
    for r in per_run:
        for mi in common_mi:
            variant_sets.append(set(r["all_results"][mi].keys()))
    variants = sorted(set.intersection(*variant_sets))
    # build tensor [runs, variants, mi]
    mat = np.zeros((len(per_run), len(variants), len(common_mi)))
    for ri, r in enumerate(per_run):
        for vi, v in enumerate(variants):
            for mj, mi in enumerate(common_mi):
                mat[ri, vi, mj] = r["all_results"][mi][v]
    mean_mat = mat.mean(axis=0)
    sem_mat = (
        mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
        if mat.shape[0] > 1
        else np.zeros_like(mean_mat)
    )
    agg = dict(
        per_run=per_run,
        common_mi=common_mi,
        variants=variants,
        mat=mat,
        mean=mean_mat,
        sem=sem_mat,
    )
except Exception as e:
    print(f"Error aggregating: {e}")

# ----- Plot 1: AUROC vs max_iter with mean ± SEM -----
try:
    fig, ax = plt.subplots(figsize=(9, 5))
    for vi, v in enumerate(agg["variants"]):
        m = agg["mean"][vi]
        s = agg["sem"][vi]
        (line,) = ax.plot(agg["common_mi"], m, marker="o", label=f"{v} (mean)")
        ax.fill_between(
            agg["common_mi"],
            m - s,
            m + s,
            alpha=0.2,
            color=line.get_color(),
            label=f"{v} ±SEM" if vi == 0 else None,
        )
    ax.set_xscale("log")
    ax.set_xlabel("max_iter (log scale)")
    ax.set_ylabel("AUROC (mean across runs)")
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.4, label="chance")
    ax.set_title(f"{DATASET}\nAggregated AUROC vs max_iter (mean ± SEM across runs)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{DATASET}_agg_auroc_vs_maxiter.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error plot1: {e}")
    plt.close()

# ----- Plot 2: bar chart at best max_iter (chosen from mean over variants' max) -----
try:
    # choose best max_iter as argmax of best-variant-mean across mi
    best_per_mi = agg["mean"].max(axis=0)  # best variant AUROC per mi
    best_mi_idx = int(np.argmax(best_per_mi))
    best_mi = agg["common_mi"][best_mi_idx]
    means = agg["mean"][:, best_mi_idx]
    sems = agg["sem"][:, best_mi_idx]
    order = np.argsort(-means)
    names_sorted = [agg["variants"][i] for i in order]
    vals = means[order]
    errs = sems[order]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        names_sorted,
        vals,
        yerr=errs,
        capsize=4,
        color="steelblue",
        label="Mean AUROC ± SEM",
    )
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.5, label="chance")
    ax.set_ylabel("AUROC (mean across runs)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{DATASET}\nMean AUROC by Probe Variant (best max_iter={best_mi})")
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

# ----- Plot 3: heatmap of mean AUROC across variants and max_iter -----
try:
    mmat = agg["mean"]
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(mmat, aspect="auto", cmap="viridis", vmin=0.4, vmax=1.0)
    ax.set_xticks(range(len(agg["common_mi"])))
    ax.set_xticklabels(agg["common_mi"])
    ax.set_yticks(range(len(agg["variants"])))
    ax.set_yticklabels(agg["variants"])
    ax.set_xlabel("max_iter")
    ax.set_ylabel("Probe variant")
    ax.set_title(f"{DATASET}\nHeatmap: Mean AUROC across runs (Variant x max_iter)")
    for i in range(mmat.shape[0]):
        for j in range(mmat.shape[1]):
            ax.text(
                j,
                i,
                f"{mmat[i,j]:.2f}",
                ha="center",
                va="center",
                color="white" if mmat[i, j] < 0.75 else "black",
                fontsize=8,
            )
    plt.colorbar(im, ax=ax, label="Mean AUROC")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, f"{DATASET}_agg_auroc_heatmap.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error plot3: {e}")
    plt.close()

# ----- Plot 4: aggregated score distribution for best variant at best_mi -----
try:
    # best variant by mean at best_mi
    best_vi = int(np.argmax(agg["mean"][:, best_mi_idx]))
    best_variant = agg["variants"][best_vi]
    all_preds_0, all_preds_1 = [], []
    for r in agg["per_run"]:
        run = r["runs"].get(str(best_mi)) or r["runs"].get(best_mi)
        if run is None or best_variant not in run["predictions"]:
            continue
        preds = np.array(run["predictions"][best_variant])
        labels = r["labels"]
        all_preds_0.append(preds[labels == 0])
        all_preds_1.append(preds[labels == 1])
    if all_preds_0 and all_preds_1:
        p0 = np.concatenate(all_preds_0)
        p1 = np.concatenate(all_preds_1)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            p0, bins=20, alpha=0.6, label=f"Independent (n={len(p0)})", color="#94a3b8"
        )
        ax.hist(
            p1, bins=20, alpha=0.6, label=f"Collusion (n={len(p1)})", color="#ef4444"
        )
        ax.axvline(
            p0.mean(),
            color="#475569",
            linestyle="--",
            label=f"Indep mean={p0.mean():.2f}",
        )
        ax.axvline(
            p1.mean(),
            color="#991b1b",
            linestyle="--",
            label=f"Collu mean={p1.mean():.2f}",
        )
        ax.set_xlabel("Predicted collusion score")
        ax.set_ylabel("Count (aggregated over runs)")
        ax.set_title(
            f"{DATASET}\nAggregated Score Distribution — Probe={best_variant}, max_iter={best_mi}"
        )
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{DATASET}_agg_score_distribution.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error plot4: {e}")
    plt.close()

# ----- Print aggregated metrics -----
try:
    n_runs = len(agg["per_run"])
    print(f"Aggregated over {n_runs} run(s)")
    print(f"Chosen best max_iter (by mean across runs): {best_mi}")
    print(f"Best variant at best max_iter: {best_variant}")
    print(
        f"  Mean AUROC = {agg['mean'][best_vi, best_mi_idx]:.4f} "
        f"± SEM {agg['sem'][best_vi, best_mi_idx]:.4f}"
    )
    # per-run collusion_detection_auroc summary
    cdas = []
    for r in agg["per_run"]:
        v = r["summary"].get("collusion_detection_auroc")
        if v is not None:
            cdas.append(float(v))
    if cdas:
        cdas = np.array(cdas)
        sem = cdas.std(ddof=1) / np.sqrt(len(cdas)) if len(cdas) > 1 else 0.0
        print(
            f"collusion_detection_auroc: mean={cdas.mean():.4f} ± SEM {sem:.4f} (n={len(cdas)})"
        )
except Exception as e:
    print(f"Error printing summary: {e}")
