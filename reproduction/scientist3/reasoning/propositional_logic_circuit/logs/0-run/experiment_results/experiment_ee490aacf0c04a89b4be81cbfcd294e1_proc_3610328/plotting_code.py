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

tuning = experiment_data.get("max_length_tuning", {})
mls = sorted([int(k) for k in tuning.keys()])

# Plot 1: accuracy vs max_length
try:
    accs = [tuning[str(m)]["metrics"]["val"][0]["task_accuracy"] for m in mls]
    plt.figure(figsize=(6, 4))
    plt.plot(mls, accs, "o-")
    plt.xscale("log", base=2)
    plt.xlabel("max_length")
    plt.ylabel("Accuracy")
    plt.title(
        "Synthetic Logic Dataset: Accuracy vs max_length\nMistral-7B Zero-shot Evaluation"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "synthetic_logic_accuracy_vs_max_length.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot 2: val loss vs max_length
try:
    losses = [tuning[str(m)]["losses"]["val"][0]["val_loss"] for m in mls]
    plt.figure(figsize=(6, 4))
    plt.plot(mls, losses, "o-", color="orange")
    plt.xscale("log", base=2)
    plt.xlabel("max_length")
    plt.ylabel("Validation Loss (NLL)")
    plt.title(
        "Synthetic Logic Dataset: Val Loss vs max_length\nMistral-7B Zero-shot Evaluation"
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "synthetic_logic_valloss_vs_max_length.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot 3: True vs False logit distributions per max_length
try:
    fig, axes = plt.subplots(1, len(mls), figsize=(4 * len(mls), 4), squeeze=False)
    for i, m in enumerate(mls):
        tl = np.array(tuning[str(m)]["true_logits"])
        fl = np.array(tuning[str(m)]["false_logits"])
        ax = axes[0, i]
        ax.hist(tl, bins=25, alpha=0.5, label="True logit", color="green")
        ax.hist(fl, bins=25, alpha=0.5, label="False logit", color="red")
        ax.set_title(f"max_length={m}")
        ax.set_xlabel("Logit value")
        ax.set_ylabel("Count")
        ax.legend()
    fig.suptitle(
        "Synthetic Logic Dataset: True vs False Logit Distributions\nLeft-to-Right: Increasing max_length"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "synthetic_logic_logit_distributions.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating logit distribution plot: {e}")
    plt.close()

# Plot 4: Confusion matrices per max_length
try:
    fig, axes = plt.subplots(1, len(mls), figsize=(3.5 * len(mls), 3.5), squeeze=False)
    for i, m in enumerate(mls):
        preds = np.array(tuning[str(m)]["predictions"])
        gts = np.array(tuning[str(m)]["ground_truth"])
        cm = np.zeros((2, 2), dtype=int)
        for p, g in zip(preds, gts):
            cm[int(g), int(p)] += 1
        ax = axes[0, i]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["False", "True"])
        ax.set_yticklabels(["False", "True"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Ground Truth")
        ax.set_title(f"max_length={m}")
        for r in range(2):
            for c in range(2):
                ax.text(
                    c,
                    r,
                    str(cm[r, c]),
                    ha="center",
                    va="center",
                    color="white" if cm[r, c] > cm.max() / 2 else "black",
                )
    fig.suptitle(
        "Synthetic Logic Dataset: Confusion Matrices\nRows: Ground Truth, Cols: Predicted"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "synthetic_logic_confusion_matrices.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# Plot 5: Margin (true - false logit) split by correctness for each max_length
try:
    fig, axes = plt.subplots(1, len(mls), figsize=(4 * len(mls), 4), squeeze=False)
    for i, m in enumerate(mls):
        tl = np.array(tuning[str(m)]["true_logits"])
        fl = np.array(tuning[str(m)]["false_logits"])
        preds = np.array(tuning[str(m)]["predictions"])
        gts = np.array(tuning[str(m)]["ground_truth"])
        margin = tl - fl
        correct_mask = preds == gts
        ax = axes[0, i]
        ax.hist(margin[correct_mask], bins=25, alpha=0.6, label="Correct", color="blue")
        ax.hist(
            margin[~correct_mask], bins=25, alpha=0.6, label="Incorrect", color="red"
        )
        ax.axvline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_title(f"max_length={m}")
        ax.set_xlabel("Margin (True - False logit)")
        ax.set_ylabel("Count")
        ax.legend()
    fig.suptitle("Synthetic Logic Dataset: Prediction Margin by Correctness")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "synthetic_logic_margin_by_correctness.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating margin plot: {e}")
    plt.close()

# Print summary metrics
print("===== Summary =====")
for m in mls:
    acc = tuning[str(m)]["metrics"]["val"][0]["task_accuracy"]
    loss = tuning[str(m)]["losses"]["val"][0]["val_loss"]
    print(f"max_length={m}: accuracy={acc:.4f}, val_loss={loss:.4f}")
