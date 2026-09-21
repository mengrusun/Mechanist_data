import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_11-03-14_multi_agent_attempt_0/logs/0-run/experiment_results/experiment_30544dfdb11a4b41b9ab07e6b950c8e7_proc_968037/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        experiment_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

ds_key = "NARCBench-Core-Synthetic"

# Aggregate AUROC per variant across runs
agg_auroc_variant = {}  # variant -> list of run-level means
agg_auroc_folds = {}  # variant -> concatenated fold values across runs
agg_predictions = {}  # variant -> list of arrays
agg_gt = []
overall_auroc = []

for exp in all_experiment_data:
    data = exp.get(ds_key, {})
    for k, v in data.get("auroc_per_variant", {}).items():
        agg_auroc_variant.setdefault(k, []).append(v)
    for k, v in data.get("auroc_folds", {}).items():
        agg_auroc_folds.setdefault(k, []).extend(list(v))
    for k, v in data.get("predictions", {}).items():
        agg_predictions.setdefault(k, []).append(np.array(v))
    gt = data.get("ground_truth", [])
    if len(gt) > 0:
        agg_gt.append(np.array(gt))
    if "collusion_detection_auroc" in data:
        overall_auroc.append(data["collusion_detection_auroc"])

n_runs = len(all_experiment_data)

# Plot 1: bar chart of mean AUROC by variant with SEM across runs
try:
    plt.figure(figsize=(10, 5))
    names = sorted(
        agg_auroc_variant.keys(), key=lambda k: -np.mean(agg_auroc_variant[k])
    )
    means = [np.mean(agg_auroc_variant[k]) for k in names]
    sems = [
        np.std(agg_auroc_variant[k]) / np.sqrt(max(len(agg_auroc_variant[k]), 1))
        for k in names
    ]
    group_aggs = {"mean_pool", "max_pool", "concat"}
    colors = [
        (
            "#3b82f6"
            if k in group_aggs
            else (
                "#f59e0b"
                if k == "tfidf_text"
                else "#94a3b8" if k == "random_features" else "#10b981"
            )
        )
        for k in names
    ]
    x = np.arange(len(names))
    plt.bar(
        x,
        means,
        color=colors,
        yerr=sems,
        capsize=4,
        error_kw={"ecolor": "black", "label": f"SEM across {n_runs} run(s)"},
    )
    plt.axhline(0.5, color="k", linestyle="--", alpha=0.5, label="chance")
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel("AUROC (mean across runs)")
    plt.ylim(0, 1.05)
    plt.title(
        f"NARCBench-Core-Synthetic: Mean AUROC by Probe Variant\n"
        f"(Collusion Detection, aggregated across {n_runs} run(s), error bars = SEM)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, "NARCBench-Core-Synthetic_agg_auroc_bar_by_variant.png"
        ),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: per-fold AUROC aggregated across runs with SEM error bars
try:
    plt.figure(figsize=(10, 5))
    names = sorted(agg_auroc_folds.keys(), key=lambda k: -np.mean(agg_auroc_folds[k]))
    means = [np.mean(agg_auroc_folds[k]) for k in names]
    sems = [
        np.std(agg_auroc_folds[k]) / np.sqrt(max(len(agg_auroc_folds[k]), 1))
        for k in names
    ]
    x = np.arange(len(names))
    plt.errorbar(
        x,
        means,
        yerr=sems,
        fmt="o",
        capsize=5,
        color="#3b82f6",
        ecolor="#94a3b8",
        label="Mean AUROC ± SEM (across folds & runs)",
    )
    plt.axhline(0.5, color="k", linestyle="--", alpha=0.5, label="chance")
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel("AUROC (mean ± SEM)")
    plt.ylim(0, 1.05)
    plt.title(
        f"NARCBench-Core-Synthetic: Aggregated Per-Fold AUROC\n"
        f"(Pooled over {n_runs} run(s), error bars = SEM)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, "NARCBench-Core-Synthetic_agg_auroc_foldsem_by_variant.png"
        ),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: pooled prediction score histograms by class for best group-aggregated probe
try:
    group_aggs = ["mean_pool", "max_pool", "concat"]
    avail = [g for g in group_aggs if g in agg_auroc_variant]
    if avail and len(agg_gt) > 0:
        best = max(avail, key=lambda k: np.mean(agg_auroc_variant[k]))
        # Pool predictions and gts across runs (assumes same test order per run)
        preds_pool = np.concatenate(agg_predictions[best])
        gt_pool = np.concatenate(agg_gt)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        axes[0].hist(
            preds_pool[gt_pool == 0],
            bins=20,
            color="#ef4444",
            alpha=0.8,
            label="Non-Collusion samples",
        )
        axes[0].set_title("Left: Ground Truth = Non-Collusion (label 0)")
        axes[0].set_xlabel("Predicted collusion probability")
        axes[0].set_ylabel("Count")
        axes[0].legend()
        axes[1].hist(
            preds_pool[gt_pool == 1],
            bins=20,
            color="#10b981",
            alpha=0.8,
            label="Collusion samples",
        )
        axes[1].set_title("Right: Ground Truth = Collusion (label 1)")
        axes[1].set_xlabel("Predicted collusion probability")
        axes[1].legend()
        mean_auroc = np.mean(agg_auroc_variant[best])
        sem_auroc = np.std(agg_auroc_variant[best]) / np.sqrt(
            max(len(agg_auroc_variant[best]), 1)
        )
        fig.suptitle(
            f"NARCBench-Core-Synthetic: Pooled Score Distribution by Class\n"
            f"({best} probe, AUROC = {mean_auroc:.3f} ± {sem_auroc:.3f} SEM, {n_runs} run(s))"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, f"NARCBench-Core-Synthetic_agg_score_hist_{best}.png"
            ),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Print aggregated summary metrics
try:
    if overall_auroc:
        m = np.mean(overall_auroc)
        s = np.std(overall_auroc) / np.sqrt(len(overall_auroc))
        print(
            f"collusion_detection_auroc (mean ± SEM across {n_runs} run(s)) = {m:.4f} ± {s:.4f}"
        )
    print("Per-variant aggregated AUROC (mean ± SEM across runs):")
    for k in sorted(
        agg_auroc_variant.keys(), key=lambda x: -np.mean(agg_auroc_variant[x])
    ):
        vals = agg_auroc_variant[k]
        m = np.mean(vals)
        s = np.std(vals) / np.sqrt(max(len(vals), 1))
        print(f"  {k:20s}: {m:.4f} ± {s:.4f}  (n={len(vals)})")
except Exception as e:
    print(f"Error printing summary: {e}")
