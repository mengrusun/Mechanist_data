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

ds_key = "esmfold_hairpin_panel"
ds = experiment_data.get(ds_key, {})
per_prot = ds.get("per_protein", [])
names = [p["name"] for p in per_prot]
gts = np.array([int(p["gt_hairpin"]) for p in per_prot])
preds = np.array([int(p["pred_hairpin"]) for p in per_prot])
plddts = np.array([p.get("plddt", np.nan) for p in per_prot])
correct = np.array([int(p["correct"]) for p in per_prot])
val_metrics = ds.get("metrics", {}).get("val", [{}])
acc = (
    val_metrics[0].get("hairpin_dssp_accuracy", float("nan"))
    if val_metrics
    else float("nan")
)

# Plot 1: GT vs Predicted hairpin per protein
try:
    plt.figure(figsize=(9, 3.5))
    x = np.arange(len(names))
    plt.bar(x - 0.2, gts, 0.4, label="Ground Truth", color="steelblue")
    plt.bar(x + 0.2, preds, 0.4, label="Predicted", color="orange")
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("Hairpin (0/1)")
    plt.title(
        f"ESMFold Hairpin Panel: GT vs Predicted Hairpin\n(Left bar: Ground Truth, Right bar: Predicted) acc={acc:.3f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "esmfold_hairpin_panel_gt_vs_pred.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: pLDDT per protein colored by correctness
try:
    plt.figure(figsize=(9, 3.5))
    colors = ["green" if c else "red" for c in correct]
    x = np.arange(len(names))
    plt.bar(x, plddts, color=colors)
    plt.xticks(x, names, rotation=30, ha="right")
    plt.ylabel("mean pLDDT")
    plt.title(
        "ESMFold Hairpin Panel: Per-Protein pLDDT\n(Green: Correct hairpin call, Red: Incorrect)"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "esmfold_hairpin_panel_plddt.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Confusion matrix
try:
    cm = np.zeros((2, 2), dtype=int)
    for g, p in zip(gts, preds):
        cm[g, p] += 1
    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=14,
            )
    plt.xticks([0, 1], ["Pred: No", "Pred: Yes"])
    plt.yticks([0, 1], ["GT: No", "GT: Yes"])
    plt.title(
        "ESMFold Hairpin Panel: Confusion Matrix\n(Rows: Ground Truth, Cols: Predicted)"
    )
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "esmfold_hairpin_panel_confusion_matrix.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Accuracy summary
try:
    n_eval = (
        val_metrics[0].get("n_eval", len(per_prot)) if val_metrics else len(per_prot)
    )
    n_correct = (
        val_metrics[0].get("n_correct", int(correct.sum()))
        if val_metrics
        else int(correct.sum())
    )
    plt.figure(figsize=(4, 4))
    plt.bar(
        ["Accuracy", "Loss (1-acc)"], [acc, 1.0 - acc], color=["seagreen", "salmon"]
    )
    plt.ylim(0, 1)
    plt.title(
        f"ESMFold Hairpin Panel: Validation Summary\n({n_correct}/{n_eval} correct)"
    )
    plt.ylabel("value")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "esmfold_hairpin_panel_accuracy_summary.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

print(f"hairpin_dssp_accuracy = {acc:.4f}")
