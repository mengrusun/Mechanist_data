import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data = experiment_data.get("prop_logic_synth", {})
preds = np.array(data.get("predictions", []))
labels = np.array(data.get("ground_truth", []))
tl = np.array(data.get("true_logits", []))
fl = np.array(data.get("false_logits", []))
val_metrics = data.get("metrics", {}).get("val", [])
val_losses = data.get("losses", {}).get("val", [])
accuracy = (
    val_metrics[0]["task_accuracy"]
    if val_metrics
    else float((preds == labels).mean()) if len(preds) else 0.0
)
val_loss = val_losses[0]["val_loss"] if val_losses else None

# Plot 1: Logit gap histogram by label
try:
    plt.figure(figsize=(7, 4))
    diffs = tl - fl
    lab_arr = labels.astype(int)
    plt.hist(diffs[lab_arr == 1], bins=25, alpha=0.6, label="True label")
    plt.hist(diffs[lab_arr == 0], bins=25, alpha=0.6, label="False label")
    plt.axvline(0, color="k", linestyle="--")
    plt.xlabel("True_logit - False_logit")
    plt.ylabel("Count")
    plt.legend()
    plt.title(
        f"Prop Logic Synth: Logit Gap Distribution by Ground Truth Label\n(acc={accuracy:.3f})"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "prop_logic_synth_logit_gap_histogram.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Accuracy and validation loss bar
try:
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].bar(["accuracy"], [accuracy], color="steelblue")
    ax[0].set_ylim(0, 1)
    ax[0].set_title("Task Accuracy")
    for i, v in enumerate([accuracy]):
        ax[0].text(i, v + 0.02, f"{v:.3f}", ha="center")
    if val_loss is not None:
        ax[1].bar(["val_loss"], [val_loss], color="indianred")
        ax[1].set_title("Validation Loss (-log P(correct))")
        ax[1].text(0, val_loss * 1.02, f"{val_loss:.3f}", ha="center")
    fig.suptitle(
        "Prop Logic Synth: Evaluation Metrics (Left: Accuracy, Right: Val Loss)"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "prop_logic_synth_metrics_summary.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Confusion Matrix
try:
    plt.figure(figsize=(5, 4))
    cm = np.zeros((2, 2), dtype=int)
    for l, p in zip(labels.astype(int), preds.astype(int)):
        cm[l, p] += 1
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks([0, 1], ["Pred False", "Pred True"])
    plt.yticks([0, 1], ["True False", "True True"])
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
            )
    plt.title("Prop Logic Synth: Confusion Matrix\n(Ground Truth vs Predictions)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "prop_logic_synth_confusion_matrix.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Scatter of true vs false logits
try:
    plt.figure(figsize=(6, 5))
    lab_arr = labels.astype(int)
    plt.scatter(
        fl[lab_arr == 1], tl[lab_arr == 1], alpha=0.6, label="True label", c="green"
    )
    plt.scatter(
        fl[lab_arr == 0], tl[lab_arr == 0], alpha=0.6, label="False label", c="red"
    )
    lo = min(fl.min(), tl.min())
    hi = max(fl.max(), tl.max())
    plt.plot([lo, hi], [lo, hi], "k--", alpha=0.5, label="y=x (decision boundary)")
    plt.xlabel("False logit")
    plt.ylabel("True logit")
    plt.legend()
    plt.title(
        "Prop Logic Synth: True vs False Logits per Example\n(Colored by Ground Truth)"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "prop_logic_synth_logit_scatter.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

print(f"Accuracy: {accuracy:.4f}")
if val_loss is not None:
    print(f"Val loss: {val_loss:.4f}")
