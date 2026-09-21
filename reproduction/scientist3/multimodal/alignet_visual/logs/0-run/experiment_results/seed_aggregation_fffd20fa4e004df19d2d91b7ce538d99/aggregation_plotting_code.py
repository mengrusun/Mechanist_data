import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_04-06-03_alignet_visual_attempt_1/logs/0-run/experiment_results/experiment_24c2675d2842491b87080bbb1a808342_proc_2155478/experiment_data.npy"
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


def aggregate_curves(list_of_arrays):
    """Aggregate arrays of possibly different lengths -> mean, sem, x."""
    arrs = [
        np.asarray(a, dtype=float)
        for a in list_of_arrays
        if a is not None and len(a) > 0
    ]
    if not arrs:
        return None, None, None
    min_len = min(len(a) for a in arrs)
    stacked = np.stack([a[:min_len] for a in arrs], axis=0)
    mean = stacked.mean(axis=0)
    sem = (
        stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
        if stacked.shape[0] > 1
        else np.zeros_like(mean)
    )
    x = np.arange(min_len)
    return x, mean, sem


if all_experiment_data:
    # Collect per-run data across all experiment files and all N_EPOCHS settings
    # Group by n_epochs value to aggregate across seeds/files
    grouped = {}  # key: n_epochs (int), val: list of run dicts
    baseline_vals = []
    teacher_vals = []

    for ed_file in all_experiment_data:
        runs = ed_file.get("N_EPOCHS", {})
        for rk, run in runs.items():
            things = run.get("THINGS", {})
            ne = things.get("n_epochs", None)
            if ne is None:
                continue
            grouped.setdefault(ne, []).append(things)
            baseline_vals.append(things.get("baseline_ooo", np.nan))
            teacher_vals.append(things.get("teacher_ooo", np.nan))

    baseline_mean = float(np.nanmean(baseline_vals)) if baseline_vals else None
    teacher_mean = float(np.nanmean(teacher_vals)) if teacher_vals else None

    sorted_ne = sorted(grouped.keys())

    # Plot 1: Aggregated train loss curves (mean ± SEM) per N_EPOCHS setting
    try:
        plt.figure(figsize=(8, 5))
        for ne in sorted_ne:
            train_losses = [r["losses"]["train"] for r in grouped[ne] if "losses" in r]
            x, mean, sem = aggregate_curves(train_losses)
            if mean is None:
                continue
            plt.plot(x, mean, label=f"E={ne} mean (n={len(train_losses)})")
            plt.fill_between(
                x, mean - sem, mean + sem, alpha=0.25, label=f"E={ne} ±SEM"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss (MSE)")
        plt.title(
            "THINGS Dataset: Aggregated Training Loss\nMean ± Standard Error across runs"
        )
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "THINGS_agg_train_loss_curves.png"), dpi=100
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot1: {e}")
        plt.close()

    # Plot 2: Aggregated validation loss curves
    try:
        plt.figure(figsize=(8, 5))
        for ne in sorted_ne:
            val_losses = [r["losses"]["val"] for r in grouped[ne] if "losses" in r]
            x, mean, sem = aggregate_curves(val_losses)
            if mean is None:
                continue
            plt.plot(x, mean, label=f"E={ne} mean (n={len(val_losses)})")
            plt.fill_between(
                x, mean - sem, mean + sem, alpha=0.25, label=f"E={ne} ±SEM"
            )
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss (MSE)")
        plt.title(
            "THINGS Dataset: Aggregated Validation Loss\nMean ± Standard Error across runs"
        )
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "THINGS_agg_val_loss_curves.png"), dpi=100
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot2: {e}")
        plt.close()

    # Plot 3: Aggregated OOO accuracy curves
    try:
        plt.figure(figsize=(8, 5))
        for ne in sorted_ne:
            metric_curves = [r["metrics"]["val"] for r in grouped[ne] if "metrics" in r]
            x, mean, sem = aggregate_curves(metric_curves)
            if mean is None:
                continue
            plt.plot(x, mean, label=f"E={ne} mean (n={len(metric_curves)})")
            plt.fill_between(
                x, mean - sem, mean + sem, alpha=0.25, label=f"E={ne} ±SEM"
            )
        if baseline_mean is not None:
            plt.axhline(
                baseline_mean,
                color="k",
                linestyle="--",
                label=f"DINOv2 baseline ({baseline_mean:.3f})",
            )
        if teacher_mean is not None:
            plt.axhline(
                teacher_mean,
                color="r",
                linestyle="--",
                label=f"SigLIP teacher ({teacher_mean:.3f})",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Aligned OOO Accuracy")
        plt.title(
            "THINGS Dataset: Aggregated OOO Accuracy\nMean ± Standard Error across runs"
        )
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "THINGS_agg_ooo_accuracy_curves.png"), dpi=100
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot3: {e}")
        plt.close()

    # Plot 4: Final aligned OOO bar chart with SEM error bars
    try:
        plt.figure(figsize=(8, 5))
        names, means, sems, ns = [], [], [], []
        for ne in sorted_ne:
            finals = [r.get("final_aligned_ooo", np.nan) for r in grouped[ne]]
            finals = [f for f in finals if not np.isnan(f)]
            if not finals:
                continue
            names.append(f"E={ne}")
            m = float(np.mean(finals))
            s = (
                float(np.std(finals, ddof=1) / np.sqrt(len(finals)))
                if len(finals) > 1
                else 0.0
            )
            means.append(m)
            sems.append(s)
            ns.append(len(finals))
        x = np.arange(len(names))
        plt.bar(
            x,
            means,
            yerr=sems,
            capsize=5,
            color="steelblue",
            label="Aligned mean ± SEM",
        )
        if baseline_mean is not None:
            plt.axhline(
                baseline_mean,
                color="k",
                linestyle="--",
                label=f"Baseline ({baseline_mean:.3f})",
            )
        if teacher_mean is not None:
            plt.axhline(
                teacher_mean,
                color="r",
                linestyle="--",
                label=f"Teacher ({teacher_mean:.3f})",
            )
        plt.xticks(x, names)
        plt.ylabel("Final Aligned OOO Accuracy")
        plt.title(
            "THINGS Dataset: Final OOO Accuracy by N_EPOCHS\nMean ± SEM across runs"
        )
        plt.legend(fontsize=8)
        for xi, m, s, n in zip(x, means, sems, ns):
            plt.text(xi, m + s + 0.003, f"{m:.3f}\n(n={n})", ha="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "THINGS_agg_final_ooo_bar.png"), dpi=100)
        plt.close()
    except Exception as e:
        print(f"Error creating plot4: {e}")
        plt.close()

    # Plot 5: Aggregated prediction vs ground truth distribution across all runs
    try:
        all_preds, all_gts = [], []
        for ne in sorted_ne:
            for r in grouped[ne]:
                p = r.get("predictions", None)
                g = r.get("ground_truth", None)
                if p is not None and g is not None:
                    all_preds.append(np.asarray(p))
                    all_gts.append(np.asarray(g))
        if all_preds:
            preds = np.concatenate(all_preds)
            gts = np.concatenate(all_gts)
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(gts, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8, color="green")
            axes[0].set_xticks([0, 1, 2])
            axes[0].set_title("Ground Truth (aggregated)")
            axes[0].set_xlabel("Odd-One-Out Position")
            axes[1].hist(preds, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8, color="orange")
            axes[1].set_xticks([0, 1, 2])
            axes[1].set_title("Predicted (aggregated)")
            axes[1].set_xlabel("Odd-One-Out Position")
            fig.suptitle(
                "THINGS Dataset - Left: Ground Truth, Right: Predicted OOO (aggregated across all runs)"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "THINGS_agg_pred_vs_gt_distribution.png"),
                dpi=100,
            )
            plt.close()
    except Exception as e:
        print(f"Error creating plot5: {e}")
        plt.close()

    # Print evaluation metrics
    print("=== Aggregated Evaluation Metrics (THINGS) ===")
    if baseline_mean is not None:
        print(f"Baseline DINOv2 OOO (mean): {baseline_mean:.4f}")
    if teacher_mean is not None:
        print(f"Teacher SigLIP OOO (mean):  {teacher_mean:.4f}")
    for ne in sorted_ne:
        finals = [r.get("final_aligned_ooo", np.nan) for r in grouped[ne]]
        finals = [f for f in finals if not np.isnan(f)]
        if not finals:
            continue
        m = float(np.mean(finals))
        s = (
            float(np.std(finals, ddof=1) / np.sqrt(len(finals)))
            if len(finals) > 1
            else 0.0
        )
        print(
            f"  N_EPOCHS={ne} (n={len(finals)}): Final Aligned OOO = {m:.4f} ± {s:.4f} (SEM)"
        )
