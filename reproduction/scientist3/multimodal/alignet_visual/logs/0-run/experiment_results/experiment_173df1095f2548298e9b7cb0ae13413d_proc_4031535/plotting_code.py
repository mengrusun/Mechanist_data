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
    runs = experiment_data["N_EPOCHS"]
    run_keys = sorted(runs.keys(), key=lambda k: int(k.split("_")[1]))
    baseline_ooo = runs[run_keys[0]]["THINGS"]["baseline_ooo"]
    teacher_ooo = runs[run_keys[0]]["THINGS"]["teacher_ooo"]

    # Plot 1: Training loss curves
    try:
        plt.figure(figsize=(8, 5))
        for rk in run_keys:
            ed = runs[rk]["THINGS"]
            plt.plot(ed["losses"]["train"], label=f"E={ed['n_epochs']}")
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss (MSE)")
        plt.title(
            "THINGS Dataset: Training Loss Curves\nAcross Different N_EPOCHS Settings"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "THINGS_train_loss_curves.png"), dpi=100)
        plt.close()
    except Exception as e:
        print(f"Error creating plot1: {e}")
        plt.close()

    # Plot 2: Validation loss curves
    try:
        plt.figure(figsize=(8, 5))
        for rk in run_keys:
            ed = runs[rk]["THINGS"]
            plt.plot(ed["losses"]["val"], label=f"E={ed['n_epochs']}")
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss (MSE)")
        plt.title(
            "THINGS Dataset: Validation Loss Curves\nAcross Different N_EPOCHS Settings"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "THINGS_val_loss_curves.png"), dpi=100)
        plt.close()
    except Exception as e:
        print(f"Error creating plot2: {e}")
        plt.close()

    # Plot 3: Aligned OOO accuracy curves
    try:
        plt.figure(figsize=(8, 5))
        for rk in run_keys:
            ed = runs[rk]["THINGS"]
            plt.plot(ed["metrics"]["val"], label=f"E={ed['n_epochs']}")
        plt.axhline(
            baseline_ooo,
            color="k",
            linestyle="--",
            label=f"DINOv2 baseline ({baseline_ooo:.3f})",
        )
        plt.axhline(
            teacher_ooo,
            color="r",
            linestyle="--",
            label=f"SigLIP teacher ({teacher_ooo:.3f})",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Aligned OOO Accuracy")
        plt.title(
            "THINGS Dataset: Odd-One-Out Accuracy over Epochs\nAligned Features vs Baseline/Teacher"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "THINGS_ooo_accuracy_curves.png"), dpi=100
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot3: {e}")
        plt.close()

    # Plot 4: Final aligned OOO bar chart
    try:
        plt.figure(figsize=(8, 5))
        names = []
        vals = []
        for rk in run_keys:
            ed = runs[rk]["THINGS"]
            names.append(f"E={ed['n_epochs']}")
            vals.append(ed["final_aligned_ooo"])
        x = np.arange(len(names))
        plt.bar(x, vals, color="steelblue", label="Aligned")
        plt.axhline(
            baseline_ooo,
            color="k",
            linestyle="--",
            label=f"Baseline ({baseline_ooo:.3f})",
        )
        plt.axhline(
            teacher_ooo, color="r", linestyle="--", label=f"Teacher ({teacher_ooo:.3f})"
        )
        plt.xticks(x, names)
        plt.ylabel("Final Aligned OOO Accuracy")
        plt.title(
            "THINGS Dataset: Final OOO Accuracy by N_EPOCHS\nComparison with Baseline & Teacher"
        )
        plt.legend()
        for xi, v in zip(x, vals):
            plt.text(xi, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "THINGS_final_ooo_bar.png"), dpi=100)
        plt.close()
    except Exception as e:
        print(f"Error creating plot4: {e}")
        plt.close()

    # Plot 5: Prediction vs Ground Truth distribution for best run
    try:
        best_rk = max(run_keys, key=lambda k: runs[k]["THINGS"]["final_aligned_ooo"])
        ed = runs[best_rk]["THINGS"]
        preds = np.array(ed["predictions"])
        gts = np.array(ed["ground_truth"])
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].hist(gts, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8, color="green")
        axes[0].set_xticks([0, 1, 2])
        axes[0].set_title("Ground Truth")
        axes[0].set_xlabel("Odd-One-Out Position")
        axes[1].hist(preds, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8, color="orange")
        axes[1].set_xticks([0, 1, 2])
        axes[1].set_title("Predicted")
        axes[1].set_xlabel("Odd-One-Out Position")
        fig.suptitle(
            f"THINGS Dataset (Best: {best_rk}) - Left: Ground Truth, Right: Predicted OOO Choice Distribution"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "THINGS_pred_vs_gt_distribution.png"), dpi=100
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot5: {e}")
        plt.close()

    # Print eval metrics
    print(f"Baseline DINOv2 OOO: {baseline_ooo:.4f}")
    print(f"Teacher SigLIP OOO:  {teacher_ooo:.4f}")
    for rk in run_keys:
        ed = runs[rk]["THINGS"]
        print(f"  {rk}: Final Aligned OOO = {ed['final_aligned_ooo']:.4f}")
