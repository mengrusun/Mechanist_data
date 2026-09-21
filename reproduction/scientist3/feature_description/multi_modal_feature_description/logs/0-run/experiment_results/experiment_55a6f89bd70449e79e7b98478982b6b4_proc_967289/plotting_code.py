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

key = "imagenet_val_resnet50"
d = experiment_data.get(key, {})
concept_scores = np.asarray(d.get("predictions", []))
random_scores = np.asarray(d.get("ground_truth", []))
val_metrics = d.get("metrics", {}).get("val", [{}])[0] if d else {}

# Plot 1: Histogram
try:
    plt.figure(figsize=(6, 4))
    plt.hist(concept_scores, bins=20, alpha=0.7, label="top-k activating")
    plt.hist(random_scores, bins=20, alpha=0.7, label="random baseline")
    plt.xlabel("Mean pairwise CLIP cosine similarity")
    plt.ylabel("# channels")
    plt.title(
        "ImageNet-val / ResNet-50 layer4\nConcept Consistency Histogram (Top-k Activating vs Random Baseline)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "imagenetval_resnet50_concept_hist.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Sorted per-channel concept scores
try:
    plt.figure(figsize=(7, 4))
    sorted_c = np.sort(concept_scores)[::-1]
    plt.plot(sorted_c, marker="o", ms=3, lw=1, label="concept consistency (sorted)")
    if random_scores.size > 0:
        plt.axhline(
            float(random_scores.mean()),
            color="r",
            ls="--",
            label=f"random baseline mean = {random_scores.mean():.3f}",
        )
    plt.xlabel("Channel rank")
    plt.ylabel("Mean pairwise CLIP cos sim")
    plt.title(
        "ImageNet-val / ResNet-50 layer4\nSorted Per-Channel Concept Consistency Scores"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "imagenetval_resnet50_sorted_concept_scores.png"),
        dpi=140,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Summary metrics bar chart
try:
    labels, vals = [], []
    for k in [
        "concept_consistency_score_mean",
        "concept_consistency_score_median",
        "random_baseline_mean",
        "delta",
    ]:
        if k in val_metrics:
            labels.append(k.replace("_", "\n"))
            vals.append(float(val_metrics[k]))
    if vals:
        plt.figure(figsize=(7, 4))
        colors = ["tab:blue", "tab:cyan", "tab:orange", "tab:green"][: len(vals)]
        plt.bar(labels, vals, color=colors)
        for i, v in enumerate(vals):
            plt.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        plt.ylabel("Score")
        plt.title(
            "ImageNet-val / ResNet-50 layer4\nSummary Metrics (CLIP-based Concept Consistency)"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "imagenetval_resnet50_summary_metrics.png"),
            dpi=140,
        )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Print evaluation metrics
if val_metrics:
    print("Evaluation metrics:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v}")
