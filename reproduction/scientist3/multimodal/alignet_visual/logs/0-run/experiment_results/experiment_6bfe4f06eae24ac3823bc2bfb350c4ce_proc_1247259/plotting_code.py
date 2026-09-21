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
    experiment_data = None

if experiment_data is not None:
    things = experiment_data.get("THINGS", {})
    train_losses = things.get("losses", {}).get("train", [])
    val_losses = things.get("losses", {}).get("val", [])
    val_metrics = things.get("metrics", {}).get("val", [])
    baseline_ooo = things.get("baseline_ooo", None)
    teacher_ooo = things.get("teacher_ooo", None)
    final_ooo = things.get("final_aligned_ooo", None)
    preds = np.array(things.get("predictions", []))
    gt = np.array(things.get("ground_truth", []))

    # Plot 1: Loss curves
    try:
        plt.figure(figsize=(6, 4))
        epochs = np.arange(len(train_losses))
        plt.plot(epochs, train_losses, label="train", marker="o")
        plt.plot(epochs, val_losses, label="val", marker="s")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(
            "THINGS Dataset: Distillation Loss Curves\nLeft: Training, Right: Validation (same plot)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "THINGS_loss_curves.png"), dpi=100)
        plt.close()
    except Exception as e:
        print(f"Error creating loss plot: {e}")
        plt.close()

    # Plot 2: OOO accuracy over epochs w/ baseline & teacher
    try:
        plt.figure(figsize=(6, 4))
        epochs = np.arange(len(val_metrics))
        plt.plot(epochs, val_metrics, label="Aligned Student", color="C2", marker="o")
        if baseline_ooo is not None:
            plt.axhline(
                baseline_ooo,
                color="C0",
                linestyle="--",
                label=f"Baseline DINOv2 ({baseline_ooo:.3f})",
            )
        if teacher_ooo is not None:
            plt.axhline(
                teacher_ooo,
                color="C1",
                linestyle="--",
                label=f"Teacher SigLIP ({teacher_ooo:.3f})",
            )
        plt.xlabel("Epoch")
        plt.ylabel("OOO Accuracy")
        plt.title(
            "THINGS Dataset: Odd-One-Out Accuracy over Epochs\nAligned Student vs Baseline vs Teacher"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "THINGS_ooo_accuracy_curve.png"), dpi=100)
        plt.close()
    except Exception as e:
        print(f"Error creating OOO curve plot: {e}")
        plt.close()

    # Plot 3: Final comparison bar chart
    try:
        plt.figure(figsize=(6, 4))
        names, vals = [], []
        if baseline_ooo is not None:
            names.append("Baseline\nDINOv2")
            vals.append(baseline_ooo)
        if final_ooo is not None:
            names.append("Aligned\nStudent")
            vals.append(final_ooo)
        if teacher_ooo is not None:
            names.append("Teacher\nSigLIP")
            vals.append(teacher_ooo)
        colors = ["C0", "C2", "C1"][: len(vals)]
        bars = plt.bar(names, vals, color=colors)
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center")
        plt.ylabel("OOO Accuracy")
        plt.title(
            "THINGS Dataset: Final OOO Accuracy Comparison\nBaseline vs Aligned Student vs Teacher"
        )
        plt.ylim(0, max(vals) * 1.15 if vals else 1)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "THINGS_final_accuracy_bar.png"), dpi=100)
        plt.close()
    except Exception as e:
        print(f"Error creating bar plot: {e}")
        plt.close()

    # Plot 4: Prediction vs GT distribution
    try:
        if len(preds) > 0 and len(gt) > 0:
            fig, axes = plt.subplots(1, 2, figsize=(9, 4))
            classes = [0, 1, 2]
            gt_counts = [np.sum(gt == c) for c in classes]
            pred_counts = [np.sum(preds == c) for c in classes]
            axes[0].bar(classes, gt_counts, color="C0")
            axes[0].set_title("Ground Truth")
            axes[0].set_xlabel("Odd-One-Out Index")
            axes[0].set_ylabel("Count")
            axes[0].set_xticks(classes)
            axes[1].bar(classes, pred_counts, color="C2")
            axes[1].set_title("Aligned Student Predictions")
            axes[1].set_xlabel("Odd-One-Out Index")
            axes[1].set_xticks(classes)
            fig.suptitle(
                "THINGS Dataset: OOO Choice Distribution\nLeft: Ground Truth, Right: Aligned Student Predictions"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "THINGS_prediction_distribution.png"), dpi=100
            )
            plt.close()
    except Exception as e:
        print(f"Error creating distribution plot: {e}")
        plt.close()

    # Print metrics
    print(f"Baseline DINOv2 OOO: {baseline_ooo}")
    print(f"Teacher SigLIP OOO:  {teacher_ooo}")
    print(f"Final Aligned OOO:   {final_ooo}")
    if len(preds) > 0 and len(gt) > 0:
        acc = float(np.mean(preds == gt))
        print(f"Prediction accuracy vs GT: {acc:.4f}")
