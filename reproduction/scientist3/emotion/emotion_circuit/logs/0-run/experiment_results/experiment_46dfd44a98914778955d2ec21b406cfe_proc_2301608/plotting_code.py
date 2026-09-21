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

sev = experiment_data.get("strength", {}).get("SEV", {})
strengths = sev.get("strength_values", [])
per_str_res = sev.get("per_strength_results", {})
best_strength = sev.get("best_strength", None)
best_acc = sev.get("best_accuracy", None)

# Plot 1: accuracy vs strength
try:
    plt.figure(figsize=(8, 5))
    accs = [per_str_res[str(s)]["accuracy"] for s in strengths]
    plt.plot(strengths, accs, marker="o", color="steelblue", label="Overall accuracy")
    plt.axhline(1 / 7, color="r", linestyle="--", label="Random (1/7)")
    plt.xlabel("Steering strength")
    plt.ylabel("Accuracy")
    plt.title(
        "SEV Dataset: Steering Accuracy vs Strength\n(Overall emotion expression accuracy)"
    )
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SEV_accuracy_vs_strength.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: per-emotion accuracy for best strength
try:
    plt.figure(figsize=(8, 5))
    best_per_emo = sev.get("per_emotion_accuracy", {})
    emos = list(best_per_emo.keys())
    vals = [best_per_emo[e] for e in emos]
    plt.bar(emos, vals, color="steelblue")
    plt.axhline(1 / 7, color="r", linestyle="--", label="Random (1/7)")
    plt.ylabel("Accuracy")
    plt.title(
        f"SEV Dataset: Per-Emotion Accuracy at Best Strength={best_strength}\n(Overall accuracy={best_acc:.3f})"
    )
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SEV_per_emotion_best_strength.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: heatmap of per-emotion accuracy across strengths
try:
    plt.figure(figsize=(9, 5))
    emos = list(next(iter(per_str_res.values()))["per_emotion_accuracy"].keys())
    mat = np.array(
        [
            [per_str_res[str(s)]["per_emotion_accuracy"][e] for e in emos]
            for s in strengths
        ]
    )
    im = plt.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    plt.colorbar(im, label="Accuracy")
    plt.xticks(range(len(emos)), emos)
    plt.yticks(range(len(strengths)), strengths)
    plt.xlabel("Emotion")
    plt.ylabel("Steering strength")
    plt.title(
        "SEV Dataset: Per-Emotion Accuracy Heatmap\n(Rows: strengths, Cols: emotions)"
    )
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            plt.text(
                j,
                i,
                f"{mat[i,j]:.2f}",
                ha="center",
                va="center",
                color="white" if mat[i, j] < 0.5 else "black",
                fontsize=8,
            )
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SEV_per_emotion_heatmap.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: confusion matrix at best strength
try:
    best_res = per_str_res.get(str(best_strength), {})
    preds = best_res.get("predictions", [])
    gts = best_res.get("ground_truth", [])
    labels = sorted(list(set(gts) | set(preds)))
    label2idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for g, p in zip(gts, preds):
        cm[label2idx[g], label2idx[p]] += 1
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, label="Count")
    plt.xticks(range(len(labels)), labels, rotation=45)
    plt.yticks(range(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth (target emotion)")
    plt.title(
        f"SEV Dataset: Confusion Matrix at Best Strength={best_strength}\n(Rows: target, Cols: judge prediction)"
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
    plt.savefig(os.path.join(working_dir, "SEV_confusion_matrix_best.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: loss (1-acc) vs strength
try:
    plt.figure(figsize=(8, 5))
    losses = [1 - per_str_res[str(s)]["accuracy"] for s in strengths]
    plt.plot(
        strengths, losses, marker="s", color="darkorange", label="Val loss (1 - acc)"
    )
    plt.xlabel("Steering strength")
    plt.ylabel("Loss (1 - accuracy)")
    plt.title("SEV Dataset: Validation Loss vs Steering Strength\n(Lower is better)")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "SEV_loss_vs_strength.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print summary metric
try:
    print(f"Best strength: {best_strength}, Best accuracy: {best_acc}")
    for s in strengths:
        print(f"  strength={s}: acc={per_str_res[str(s)]['accuracy']:.4f}")
except Exception as e:
    print(f"Error printing summary: {e}")
