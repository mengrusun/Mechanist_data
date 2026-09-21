import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_04-06-03_alignet_visual_attempt_1/logs/0-run/experiment_results/experiment_6bfe4f06eae24ac3823bc2bfb350c4ce_proc_1247259/experiment_data.npy"
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


def stack_pad(list_of_lists):
    """Stack lists of possibly different lengths, padding with NaN."""
    if not list_of_lists:
        return np.array([])
    max_len = max(len(x) for x in list_of_lists)
    arr = np.full((len(list_of_lists), max_len), np.nan)
    for i, x in enumerate(list_of_lists):
        arr[i, : len(x)] = x
    return arr


def mean_sem(arr):
    """Compute mean and standard error along axis 0, ignoring NaNs."""
    mean = np.nanmean(arr, axis=0)
    n = np.sum(~np.isnan(arr), axis=0)
    std = np.nanstd(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
    sem = np.where(n > 1, std / np.sqrt(np.maximum(n, 1)), 0)
    return mean, sem


if all_experiment_data:
    # Collect data across runs
    train_losses_runs = []
    val_losses_runs = []
    val_metrics_runs = []
    baseline_ooos = []
    teacher_ooos = []
    final_ooos = []

    for exp in all_experiment_data:
        things = exp.get("THINGS", {})
        tl = things.get("losses", {}).get("train", [])
        vl = things.get("losses", {}).get("val", [])
        vm = things.get("metrics", {}).get("val", [])
        if len(tl) > 0:
            train_losses_runs.append(tl)
        if len(vl) > 0:
            val_losses_runs.append(vl)
        if len(vm) > 0:
            val_metrics_runs.append(vm)
        if things.get("baseline_ooo") is not None:
            baseline_ooos.append(things.get("baseline_ooo"))
        if things.get("teacher_ooo") is not None:
            teacher_ooos.append(things.get("teacher_ooo"))
        if things.get("final_aligned_ooo") is not None:
            final_ooos.append(things.get("final_aligned_ooo"))

    n_runs = len(all_experiment_data)
    print(f"Number of runs aggregated: {n_runs}")

    # Plot 1: Aggregated loss curves
    try:
        plt.figure(figsize=(7, 4))
        if train_losses_runs:
            arr = stack_pad(train_losses_runs)
            mean, sem = mean_sem(arr)
            epochs = np.arange(len(mean))
            plt.plot(epochs, mean, label=f"Train Mean (n={arr.shape[0]})", color="C0")
            plt.fill_between(
                epochs,
                mean - sem,
                mean + sem,
                alpha=0.3,
                color="C0",
                label="Train SEM",
            )
        if val_losses_runs:
            arr = stack_pad(val_losses_runs)
            mean, sem = mean_sem(arr)
            epochs = np.arange(len(mean))
            plt.plot(epochs, mean, label=f"Val Mean (n={arr.shape[0]})", color="C1")
            plt.fill_between(
                epochs,
                mean - sem,
                mean + sem,
                alpha=0.3,
                color="C1",
                label="Val SEM",
            )
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.title(
            "THINGS Dataset: Aggregated Distillation Loss Curves\n"
            "Mean ± Standard Error across Runs"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "THINGS_aggregated_loss_curves.png"), dpi=100
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated loss plot: {e}")
        plt.close()

    # Plot 2: Aggregated OOO accuracy curves
    try:
        plt.figure(figsize=(7, 4))
        if val_metrics_runs:
            arr = stack_pad(val_metrics_runs)
            mean, sem = mean_sem(arr)
            epochs = np.arange(len(mean))
            plt.plot(
                epochs,
                mean,
                label=f"Aligned Student Mean (n={arr.shape[0]})",
                color="C2",
                marker="o",
            )
            plt.fill_between(
                epochs,
                mean - sem,
                mean + sem,
                alpha=0.3,
                color="C2",
                label="Aligned Student SEM",
            )
        if baseline_ooos:
            b_mean = np.mean(baseline_ooos)
            b_sem = (
                np.std(baseline_ooos, ddof=1) / np.sqrt(len(baseline_ooos))
                if len(baseline_ooos) > 1
                else 0
            )
            plt.axhline(
                b_mean,
                color="C0",
                linestyle="--",
                label=f"Baseline DINOv2 ({b_mean:.3f}±{b_sem:.3f})",
            )
        if teacher_ooos:
            t_mean = np.mean(teacher_ooos)
            t_sem = (
                np.std(teacher_ooos, ddof=1) / np.sqrt(len(teacher_ooos))
                if len(teacher_ooos) > 1
                else 0
            )
            plt.axhline(
                t_mean,
                color="C1",
                linestyle="--",
                label=f"Teacher SigLIP ({t_mean:.3f}±{t_sem:.3f})",
            )
        plt.xlabel("Epoch")
        plt.ylabel("OOO Accuracy")
        plt.title(
            "THINGS Dataset: Aggregated Odd-One-Out Accuracy over Epochs\n"
            "Mean ± Standard Error across Runs"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "THINGS_aggregated_ooo_accuracy_curve.png"),
            dpi=100,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated OOO curve plot: {e}")
        plt.close()

    # Plot 3: Aggregated final comparison bar chart with SEM
    try:
        plt.figure(figsize=(7, 4))
        names, means, sems = [], [], []
        if baseline_ooos:
            names.append("Baseline\nDINOv2")
            means.append(np.mean(baseline_ooos))
            sems.append(
                np.std(baseline_ooos, ddof=1) / np.sqrt(len(baseline_ooos))
                if len(baseline_ooos) > 1
                else 0
            )
        if final_ooos:
            names.append("Aligned\nStudent")
            means.append(np.mean(final_ooos))
            sems.append(
                np.std(final_ooos, ddof=1) / np.sqrt(len(final_ooos))
                if len(final_ooos) > 1
                else 0
            )
        if teacher_ooos:
            names.append("Teacher\nSigLIP")
            means.append(np.mean(teacher_ooos))
            sems.append(
                np.std(teacher_ooos, ddof=1) / np.sqrt(len(teacher_ooos))
                if len(teacher_ooos) > 1
                else 0
            )
        colors = ["C0", "C2", "C1"][: len(means)]
        bars = plt.bar(
            names,
            means,
            yerr=sems,
            color=colors,
            capsize=6,
            label="Mean ± SEM",
        )
        for b, v, s in zip(bars, means, sems):
            plt.text(
                b.get_x() + b.get_width() / 2,
                v + s + 0.005,
                f"{v:.3f}",
                ha="center",
            )
        plt.ylabel("OOO Accuracy")
        plt.title(
            f"THINGS Dataset: Aggregated Final OOO Accuracy (n={n_runs} runs)\n"
            "Mean ± Standard Error across Runs"
        )
        if means:
            plt.ylim(0, max(m + s for m, s in zip(means, sems)) * 1.15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "THINGS_aggregated_final_accuracy_bar.png"),
            dpi=100,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating aggregated bar plot: {e}")
        plt.close()

    # Print aggregate metrics
    print("=== Aggregated Evaluation Metrics ===")
    if baseline_ooos:
        print(
            f"Baseline DINOv2 OOO: mean={np.mean(baseline_ooos):.4f}, "
            f"sem={(np.std(baseline_ooos, ddof=1)/np.sqrt(len(baseline_ooos))) if len(baseline_ooos)>1 else 0:.4f}, "
            f"n={len(baseline_ooos)}"
        )
    if teacher_ooos:
        print(
            f"Teacher SigLIP OOO:  mean={np.mean(teacher_ooos):.4f}, "
            f"sem={(np.std(teacher_ooos, ddof=1)/np.sqrt(len(teacher_ooos))) if len(teacher_ooos)>1 else 0:.4f}, "
            f"n={len(teacher_ooos)}"
        )
    if final_ooos:
        print(
            f"Final Aligned OOO:   mean={np.mean(final_ooos):.4f}, "
            f"sem={(np.std(final_ooos, ddof=1)/np.sqrt(len(final_ooos))) if len(final_ooos)>1 else 0:.4f}, "
            f"n={len(final_ooos)}"
        )
