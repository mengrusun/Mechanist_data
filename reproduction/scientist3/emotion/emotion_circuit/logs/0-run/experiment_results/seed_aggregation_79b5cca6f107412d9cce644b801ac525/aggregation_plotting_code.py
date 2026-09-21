import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
LABELS = EMOTIONS + ["neutral"]

try:
    experiment_data_path_list = [
        "experiments/2026-07-14_16-45-26_emotion_circuit_attempt_8/logs/0-run/experiment_results/experiment_7e3bc251daf24de7a715184cf5e8428a_proc_2240285/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# Aggregate data across runs
per_emo_all = {e: [] for e in EMOTIONS}
overall_accs = []
pred_dist_all = {l: [] for l in LABELS}
cms = []

for exp in all_experiment_data:
    sev = exp.get("SEV", {})
    per_emo = sev.get("per_emotion_accuracy", {})
    preds = sev.get("predictions", [])
    gts = sev.get("ground_truth", [])
    val_metrics = sev.get("metrics", {}).get("val", [])

    for e in EMOTIONS:
        if e in per_emo:
            per_emo_all[e].append(per_emo[e])

    if val_metrics:
        overall_accs.append(val_metrics[0]["emotion_expression_accuracy"])
    elif preds:
        overall_accs.append(np.mean([p == g for p, g in zip(preds, gts)]))

    counts = {l: 0 for l in LABELS}
    for p in preds:
        if p in counts:
            counts[p] += 1
    for l in LABELS:
        pred_dist_all[l].append(counts[l])

    idx = {l: i for i, l in enumerate(LABELS)}
    cm = np.zeros((len(EMOTIONS), len(LABELS)), dtype=float)
    for p, g in zip(preds, gts):
        if g in idx and p in idx and g in EMOTIONS:
            cm[EMOTIONS.index(g), idx[p]] += 1
    cms.append(cm)

n_runs = len(all_experiment_data)


def mean_se(vals):
    if len(vals) == 0:
        return 0.0, 0.0
    arr = np.array(vals, dtype=float)
    m = arr.mean()
    se = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    return m, se


# Plot 1: per-emotion accuracy with SE bars
try:
    plt.figure(figsize=(8, 5))
    means = []
    ses = []
    for e in EMOTIONS:
        m, se = mean_se(per_emo_all[e])
        means.append(m)
        ses.append(se)
    plt.bar(
        EMOTIONS,
        means,
        yerr=ses,
        color="steelblue",
        capsize=5,
        label=f"Mean ± SE (n={n_runs})",
    )
    plt.axhline(1 / 7, color="r", linestyle="--", label="random (1/7)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    overall_m, overall_se = mean_se(overall_accs)
    plt.title(
        f"SEV Dataset: Per-Emotion Steering Accuracy (Aggregated)\n"
        f"Overall Mean Accuracy = {overall_m:.3f} ± {overall_se:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SEV_per_emotion_accuracy_aggregated.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: mean confusion matrix
try:
    plt.figure(figsize=(7, 6))
    if cms:
        cm_mean = np.mean(np.stack(cms, axis=0), axis=0)
    else:
        cm_mean = np.zeros((len(EMOTIONS), len(LABELS)))
    plt.imshow(cm_mean, cmap="Blues", aspect="auto")
    plt.colorbar(label="Mean Count")
    plt.xticks(range(len(LABELS)), LABELS, rotation=45)
    plt.yticks(range(len(EMOTIONS)), EMOTIONS)
    plt.xlabel("Predicted (Judge)")
    plt.ylabel("Target (Steered Emotion)")
    plt.title(
        f"SEV Dataset: Mean Confusion Matrix (n={n_runs} runs)\n"
        "Rows: Target Emotion, Cols: Judge Prediction"
    )
    for i in range(cm_mean.shape[0]):
        for j in range(cm_mean.shape[1]):
            plt.text(
                j,
                i,
                f"{cm_mean[i, j]:.1f}",
                ha="center",
                va="center",
                color="white" if cm_mean[i, j] > cm_mean.max() / 2 else "black",
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SEV_confusion_matrix_aggregated.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: predicted label distribution with SE
try:
    plt.figure(figsize=(8, 5))
    means = []
    ses = []
    for l in LABELS:
        m, se = mean_se(pred_dist_all[l])
        means.append(m)
        ses.append(se)
    plt.bar(
        LABELS,
        means,
        yerr=ses,
        color="darkorange",
        capsize=5,
        label=f"Mean ± SE (n={n_runs})",
    )
    plt.ylabel("Count")
    plt.title(
        "SEV Dataset: Distribution of Judge-Predicted Emotion Labels (Aggregated)\n"
        "Across All Steered Generations"
    )
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "SEV_predicted_label_distribution_aggregated.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: overall accuracy vs random baseline with SE
try:
    plt.figure(figsize=(5, 5))
    overall_m, overall_se = mean_se(overall_accs)
    plt.bar(
        ["Steering", "Random (1/7)"],
        [overall_m, 1 / 7],
        yerr=[overall_se, 0],
        capsize=8,
        color=["seagreen", "gray"],
        label=f"Mean ± SE (n={n_runs})",
    )
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(
        "SEV Dataset: Overall Emotion Expression Accuracy (Aggregated)\n"
        "Steered Generation vs Random Baseline"
    )
    plt.text(
        0,
        overall_m + overall_se + 0.02,
        f"{overall_m:.3f}±{overall_se:.3f}",
        ha="center",
    )
    plt.text(1, 1 / 7 + 0.02, f"{1/7:.3f}", ha="center")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "SEV_overall_accuracy_vs_random_aggregated.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Print aggregated metrics
overall_m, overall_se = mean_se(overall_accs)
print(f"Number of runs aggregated: {n_runs}")
print(f"Overall accuracy: mean={overall_m:.4f}, SE={overall_se:.4f}")
print("Per-emotion accuracy (mean ± SE):")
for e in EMOTIONS:
    m, se = mean_se(per_emo_all[e])
    print(f"  {e}: {m:.4f} ± {se:.4f}")
