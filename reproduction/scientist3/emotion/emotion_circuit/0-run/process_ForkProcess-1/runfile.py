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

EMOTIONS = ["joy", "sadness", "anger", "fear", "surprise", "disgust"]
LABELS = EMOTIONS + ["neutral"]

sev = experiment_data.get("SEV", {})
per_emo = sev.get("per_emotion_accuracy", {})
preds = sev.get("predictions", [])
gts = sev.get("ground_truth", [])
val_metrics = sev.get("metrics", {}).get("val", [])
overall_acc = (
    val_metrics[0]["emotion_expression_accuracy"]
    if val_metrics
    else (np.mean([p == g for p, g in zip(preds, gts)]) if preds else 0.0)
)

# Plot 1: per-emotion accuracy
try:
    plt.figure(figsize=(8, 5))
    emos = list(per_emo.keys())
    vals = [per_emo[e] for e in emos]
    plt.bar(emos, vals, color="steelblue")
    plt.axhline(1 / 7, color="r", linestyle="--", label="random (1/7)")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(
        f"SEV Dataset: Per-Emotion Steering Accuracy\nOverall Accuracy = {overall_acc:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SEV_per_emotion_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: confusion matrix
try:
    plt.figure(figsize=(7, 6))
    idx = {l: i for i, l in enumerate(LABELS)}
    cm = np.zeros((len(EMOTIONS), len(LABELS)), dtype=int)
    for p, g in zip(preds, gts):
        if g in idx and p in idx and g in EMOTIONS:
            cm[EMOTIONS.index(g), idx[p]] += 1
    plt.imshow(cm, cmap="Blues", aspect="auto")
    plt.colorbar(label="Count")
    plt.xticks(range(len(LABELS)), LABELS, rotation=45)
    plt.yticks(range(len(EMOTIONS)), EMOTIONS)
    plt.xlabel("Predicted (Judge)")
    plt.ylabel("Target (Steered Emotion)")
    plt.title(
        "SEV Dataset: Confusion Matrix\nRows: Target Emotion, Cols: Judge Prediction"
    )
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SEV_confusion_matrix.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: predicted label distribution
try:
    plt.figure(figsize=(8, 5))
    counts = {l: 0 for l in LABELS}
    for p in preds:
        if p in counts:
            counts[p] += 1
    plt.bar(list(counts.keys()), list(counts.values()), color="darkorange")
    plt.ylabel("Count")
    plt.title(
        "SEV Dataset: Distribution of Judge-Predicted Emotion Labels\n(Across All Steered Generations)"
    )
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SEV_predicted_label_distribution.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: overall summary (accuracy vs random baseline)
try:
    plt.figure(figsize=(5, 5))
    plt.bar(
        ["Steering", "Random (1/7)"], [overall_acc, 1 / 7], color=["seagreen", "gray"]
    )
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title(
        "SEV Dataset: Overall Emotion Expression Accuracy\nSteered Generation vs Random Baseline"
    )
    for i, v in enumerate([overall_acc, 1 / 7]):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SEV_overall_accuracy_vs_random.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

print(f"Overall accuracy: {overall_acc:.4f}")
print(f"Per-emotion accuracy: {per_emo}")
