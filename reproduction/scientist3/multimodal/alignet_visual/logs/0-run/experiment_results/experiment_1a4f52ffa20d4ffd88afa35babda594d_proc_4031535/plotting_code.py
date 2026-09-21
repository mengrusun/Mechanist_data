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

runs = (
    experiment_data.get("direct_feature_distillation", {})
    .get("THINGS", {})
    .get("N_EPOCHS", {})
)
baseline_ooo = None
teacher_ooo = None
if runs:
    first = next(iter(runs.values()))
    baseline_ooo = first.get("baseline_ooo")
    teacher_ooo = first.get("teacher_ooo")

# Plot 1: Training and Validation Loss Curves
try:
    plt.figure(figsize=(8, 5))
    for run_key, ed in runs.items():
        plt.plot(ed["losses"]["train"], label=f"train {run_key}")
        plt.plot(ed["losses"]["val"], linestyle="--", label=f"val {run_key}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (1 - cos_sim)")
    plt.title(
        "THINGS Dataset: Direct Feature Distillation\nTrain (solid) vs Val (dashed) Loss"
    )
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "THINGS_direct_feat_distill_loss_curves.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating loss plot: {e}")
    plt.close()

# Plot 2: Aligned OOO validation accuracy across epochs
try:
    plt.figure(figsize=(8, 5))
    for run_key, ed in runs.items():
        plt.plot(ed["metrics"]["val"], label=f"{run_key}")
    if baseline_ooo is not None:
        plt.axhline(
            baseline_ooo,
            color="k",
            linestyle="--",
            label=f"baseline DINOv2 ({baseline_ooo:.3f})",
        )
    if teacher_ooo is not None:
        plt.axhline(
            teacher_ooo,
            color="r",
            linestyle="--",
            label=f"teacher SigLIP ({teacher_ooo:.3f})",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Aligned OOO Accuracy")
    plt.title(
        "THINGS Dataset: Aligned OOO Accuracy over Epochs\nDirect Feature Distillation"
    )
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "THINGS_direct_feat_distill_ooo_curves.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating OOO plot: {e}")
    plt.close()

# Plot 3: Final Aligned OOO bar chart per run
try:
    plt.figure(figsize=(7, 5))
    run_names = list(runs.keys())
    finals = [runs[k].get("final_aligned_ooo", np.nan) for k in run_names]
    xs = np.arange(len(run_names))
    plt.bar(xs, finals, color="steelblue", label="Aligned (final)")
    plt.xticks(xs, run_names, rotation=30)
    if baseline_ooo is not None:
        plt.axhline(
            baseline_ooo,
            color="k",
            linestyle="--",
            label=f"baseline ({baseline_ooo:.3f})",
        )
    if teacher_ooo is not None:
        plt.axhline(
            teacher_ooo, color="r", linestyle="--", label=f"teacher ({teacher_ooo:.3f})"
        )
    plt.ylabel("Aligned OOO Accuracy")
    plt.title(
        "THINGS Dataset: Final Aligned OOO by N_EPOCHS\nDirect Feature Distillation"
    )
    plt.legend(fontsize=8)
    for i, v in enumerate(finals):
        if not np.isnan(v):
            plt.text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "THINGS_direct_feat_distill_final_ooo_bar.png"),
        dpi=100,
    )
    plt.close()
except Exception as e:
    print(f"Error creating bar plot: {e}")
    plt.close()

# Plot 4: Confusion matrix (Predictions vs Ground Truth) for best run
try:
    if runs:
        best_key = max(runs.keys(), key=lambda k: runs[k].get("final_aligned_ooo", -1))
        ed = runs[best_key]
        preds = np.array(ed["predictions"])
        gts = np.array(ed["ground_truth"])
        if len(preds) == len(gts) and len(preds) > 0:
            cm = np.zeros((3, 3), dtype=int)
            for g, p in zip(gts, preds):
                if 0 <= g < 3 and 0 <= p < 3:
                    cm[g, p] += 1
            plt.figure(figsize=(6, 5))
            plt.imshow(cm, cmap="Blues")
            plt.colorbar()
            for i in range(3):
                for j in range(3):
                    plt.text(
                        j,
                        i,
                        str(cm[i, j]),
                        ha="center",
                        va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black",
                    )
            plt.xticks([0, 1, 2], ["Pred 0", "Pred 1", "Pred 2"])
            plt.yticks([0, 1, 2], ["GT 0", "GT 1", "GT 2"])
            plt.xlabel("Predicted odd-one-out")
            plt.ylabel("Ground truth odd-one-out")
            plt.title(
                f"THINGS Dataset: Confusion Matrix (Best Run: {best_key})\nDirect Feature Distillation"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, "THINGS_direct_feat_distill_confusion_matrix.png"
                ),
                dpi=100,
            )
            plt.close()
except Exception as e:
    print(f"Error creating confusion matrix plot: {e}")
    plt.close()

# Print evaluation metrics
print("=== Evaluation Summary ===")
print(f"Baseline DINOv2 OOO: {baseline_ooo}")
print(f"Teacher SigLIP OOO:  {teacher_ooo}")
for k, ed in runs.items():
    print(f"  {k}: Final Aligned OOO = {ed.get('final_aligned_ooo')}")
