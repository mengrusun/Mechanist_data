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

ds_key = "NARCBench-Core-Synthetic"
data = experiment_data.get(ds_key, {})
auroc_variant = data.get("auroc_per_variant", {})
auroc_folds = data.get("auroc_folds", {})
predictions = data.get("predictions", {})
gt = np.array(data.get("ground_truth", []))

# Plot 1: bar chart of mean AUROC by variant
try:
    plt.figure(figsize=(10, 5))
    names = sorted(auroc_variant.keys(), key=lambda k: -auroc_variant[k])
    vals = [auroc_variant[k] for k in names]
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
    plt.bar(names, vals, color=colors)
    plt.axhline(0.5, color="k", linestyle="--", alpha=0.5, label="chance")
    plt.ylabel("AUROC (5-fold CV mean)")
    plt.ylim(0, 1.05)
    plt.title(
        "NARCBench-Core-Synthetic: Mean AUROC by Probe Variant\n(Collusion Detection, 5-fold CV)"
    )
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "NARCBench-Core-Synthetic_auroc_bar_by_variant.png"),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: per-fold AUROC with error bars (mean +/- std)
try:
    plt.figure(figsize=(10, 5))
    names = sorted(auroc_folds.keys(), key=lambda k: -np.mean(auroc_folds[k]))
    means = [np.mean(auroc_folds[k]) for k in names]
    stds = [np.std(auroc_folds[k]) for k in names]
    x = np.arange(len(names))
    plt.errorbar(
        x, means, yerr=stds, fmt="o", capsize=5, color="#3b82f6", ecolor="#94a3b8"
    )
    plt.axhline(0.5, color="k", linestyle="--", alpha=0.5, label="chance")
    plt.xticks(x, names, rotation=35, ha="right")
    plt.ylabel("AUROC (mean ± std across folds)")
    plt.ylim(0, 1.05)
    plt.title(
        "NARCBench-Core-Synthetic: Per-Fold AUROC Variability\n(5-fold Stratified CV, error bars = std)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, "NARCBench-Core-Synthetic_auroc_foldstd_by_variant.png"
        ),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: prediction score histograms by class for best group-aggregated probe
try:
    group_aggs = ["mean_pool", "max_pool", "concat"]
    avail = [g for g in group_aggs if g in auroc_variant]
    if avail and len(gt) > 0:
        best = max(avail, key=lambda k: auroc_variant[k])
        scores = np.array(predictions[best])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        axes[0].hist(scores[gt == 0], bins=20, color="#ef4444", alpha=0.8)
        axes[0].set_title("Left: Ground Truth = Non-Collusion (label 0)")
        axes[0].set_xlabel("Predicted collusion probability")
        axes[0].set_ylabel("Count")
        axes[1].hist(scores[gt == 1], bins=20, color="#10b981", alpha=0.8)
        axes[1].set_title("Right: Ground Truth = Collusion (label 1)")
        axes[1].set_xlabel("Predicted collusion probability")
        fig.suptitle(
            f"NARCBench-Core-Synthetic: Score Distribution by Class ({best} probe, AUROC={auroc_variant[best]:.3f})"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, f"NARCBench-Core-Synthetic_score_hist_{best}.png"
            ),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Print summary metric
try:
    print(
        f"collusion_detection_auroc = {data.get('collusion_detection_auroc', float('nan')):.4f}"
    )
    for k, v in sorted(auroc_variant.items(), key=lambda x: -x[1]):
        print(f"  {k:20s}: {v:.4f}")
except Exception as e:
    print(f"Error printing summary: {e}")
