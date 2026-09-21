import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_04-06-03_alignet_visual_attempt_1/logs/0-run/experiment_results/experiment_173df1095f2548298e9b7cb0ae13413d_proc_4031535/experiment_data.npy"
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


def get_runs(ed):
    runs = ed["N_EPOCHS"]
    return runs, sorted(runs.keys(), key=lambda k: int(k.split("_")[1]))


# Gather baseline/teacher across experiments
baselines = []
teachers = []
for ed in all_experiment_data:
    runs, rk = get_runs(ed)
    baselines.append(runs[rk[0]]["THINGS"]["baseline_ooo"])
    teachers.append(runs[rk[0]]["THINGS"]["teacher_ooo"])
baseline_mean = np.mean(baselines) if baselines else 0
teacher_mean = np.mean(teachers) if teachers else 0

# Collect per-run-key data across all experiments
# For aggregation, we align by run key (e.g., "run_5", "run_10", ...)
if all_experiment_data:
    all_run_keys = set()
    for ed in all_experiment_data:
        _, rks = get_runs(ed)
        all_run_keys.update(rks)
    all_run_keys = sorted(all_run_keys, key=lambda k: int(k.split("_")[1]))

    # Aggregate train/val loss curves and ooo curves across experiments per run_key
    agg = {}
    for rk in all_run_keys:
        train_list, val_list, ooo_list, final_list, nep = [], [], [], [], None
        for ed in all_experiment_data:
            runs, _ = get_runs(ed)
            if rk in runs:
                d = runs[rk]["THINGS"]
                train_list.append(np.array(d["losses"]["train"]))
                val_list.append(np.array(d["losses"]["val"]))
                ooo_list.append(np.array(d["metrics"]["val"]))
                final_list.append(d["final_aligned_ooo"])
                nep = d["n_epochs"]
        agg[rk] = {
            "train": train_list,
            "val": val_list,
            "ooo": ooo_list,
            "final": final_list,
            "n_epochs": nep,
        }

    def stack_mean_se(list_of_arrs):
        if len(list_of_arrs) == 0:
            return None, None
        min_len = min(len(a) for a in list_of_arrs)
        arr = np.stack([a[:min_len] for a in list_of_arrs], axis=0)
        mean = arr.mean(axis=0)
        se = (
            arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
            if arr.shape[0] > 1
            else np.zeros_like(mean)
        )
        return mean, se

    # Plot 1: Aggregated Training Loss with SE
    try:
        plt.figure(figsize=(8, 5))
        for rk in all_run_keys:
            mean, se = stack_mean_se(agg[rk]["train"])
            if mean is None:
                continue
            x = np.arange(len(mean))
            plt.plot(
                x,
                mean,
                label=f"E={agg[rk]['n_epochs']} mean (n={len(agg[rk]['train'])})",
            )
            if np.any(se > 0):
                plt.fill_between(
                    x,
                    mean - se,
                    mean + se,
                    alpha=0.3,
                    label=f"E={agg[rk]['n_epochs']} ±SE",
                )
        plt.xlabel("Epoch")
        plt.ylabel("Train Loss (MSE)")
        plt.title(
            "THINGS Dataset: Aggregated Training Loss Curves\nMean ± Standard Error across runs"
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

    # Plot 2: Aggregated Validation Loss with SE
    try:
        plt.figure(figsize=(8, 5))
        for rk in all_run_keys:
            mean, se = stack_mean_se(agg[rk]["val"])
            if mean is None:
                continue
            x = np.arange(len(mean))
            plt.plot(x, mean, label=f"E={agg[rk]['n_epochs']} mean")
            if np.any(se > 0):
                plt.fill_between(
                    x,
                    mean - se,
                    mean + se,
                    alpha=0.3,
                    label=f"E={agg[rk]['n_epochs']} ±SE",
                )
        plt.xlabel("Epoch")
        plt.ylabel("Validation Loss (MSE)")
        plt.title(
            "THINGS Dataset: Aggregated Validation Loss Curves\nMean ± Standard Error across runs"
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

    # Plot 3: Aggregated OOO Accuracy curves with SE
    try:
        plt.figure(figsize=(8, 5))
        for rk in all_run_keys:
            mean, se = stack_mean_se(agg[rk]["ooo"])
            if mean is None:
                continue
            x = np.arange(len(mean))
            plt.plot(x, mean, label=f"E={agg[rk]['n_epochs']} mean")
            if np.any(se > 0):
                plt.fill_between(
                    x,
                    mean - se,
                    mean + se,
                    alpha=0.3,
                    label=f"E={agg[rk]['n_epochs']} ±SE",
                )
        plt.axhline(
            baseline_mean,
            color="k",
            linestyle="--",
            label=f"DINOv2 baseline ({baseline_mean:.3f})",
        )
        plt.axhline(
            teacher_mean,
            color="r",
            linestyle="--",
            label=f"SigLIP teacher ({teacher_mean:.3f})",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Aligned OOO Accuracy")
        plt.title(
            "THINGS Dataset: Aggregated Odd-One-Out Accuracy Curves\nMean ± Standard Error across runs"
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

    # Plot 4: Bar chart of final OOO with SE across experiments
    try:
        plt.figure(figsize=(8, 5))
        names, means, ses = [], [], []
        for rk in all_run_keys:
            finals = agg[rk]["final"]
            if len(finals) == 0:
                continue
            names.append(f"E={agg[rk]['n_epochs']}")
            means.append(np.mean(finals))
            ses.append(
                np.std(finals, ddof=1) / np.sqrt(len(finals)) if len(finals) > 1 else 0
            )
        x = np.arange(len(names))
        plt.bar(
            x,
            means,
            yerr=ses,
            capsize=5,
            color="steelblue",
            label="Aligned (mean ± SE)",
        )
        plt.axhline(
            baseline_mean,
            color="k",
            linestyle="--",
            label=f"Baseline ({baseline_mean:.3f})",
        )
        plt.axhline(
            teacher_mean,
            color="r",
            linestyle="--",
            label=f"Teacher ({teacher_mean:.3f})",
        )
        plt.xticks(x, names)
        plt.ylabel("Final Aligned OOO Accuracy")
        plt.title(
            "THINGS Dataset: Final OOO Accuracy by N_EPOCHS\nMean ± Standard Error across runs"
        )
        plt.legend()
        for xi, v, s in zip(x, means, ses):
            plt.text(xi, v + s + 0.002, f"{v:.3f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "THINGS_agg_final_ooo_bar.png"), dpi=100)
        plt.close()
    except Exception as e:
        print(f"Error creating plot4: {e}")
        plt.close()

    # Plot 5: Summary across all N_EPOCHS - mean final aligned OOO across configs
    try:
        plt.figure(figsize=(7, 5))
        all_finals = []
        for rk in all_run_keys:
            all_finals.extend(agg[rk]["final"])
        overall_mean = np.mean(all_finals) if all_finals else 0
        overall_se = (
            np.std(all_finals, ddof=1) / np.sqrt(len(all_finals))
            if len(all_finals) > 1
            else 0
        )
        categories = ["Baseline", "Teacher", "Aligned (all runs)"]
        vals = [baseline_mean, teacher_mean, overall_mean]
        errs = [0, 0, overall_se]
        plt.bar(
            categories, vals, yerr=errs, capsize=5, color=["gray", "red", "steelblue"]
        )
        plt.ylabel("OOO Accuracy")
        plt.title(
            "THINGS Dataset: Overall OOO Accuracy Comparison\nAligned aggregated across all N_EPOCHS runs (mean ± SE)"
        )
        for i, (v, e) in enumerate(zip(vals, errs)):
            plt.text(i, v + e + 0.002, f"{v:.3f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "THINGS_overall_ooo_summary.png"), dpi=100
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot5: {e}")
        plt.close()

    # Print evaluation metrics
    print(f"Baseline DINOv2 OOO (mean): {baseline_mean:.4f}")
    print(f"Teacher SigLIP OOO (mean):  {teacher_mean:.4f}")
    for rk in all_run_keys:
        finals = agg[rk]["final"]
        if not finals:
            continue
        m = np.mean(finals)
        s = np.std(finals, ddof=1) / np.sqrt(len(finals)) if len(finals) > 1 else 0
        print(
            f"  {rk} (E={agg[rk]['n_epochs']}): Final Aligned OOO = {m:.4f} ± {s:.4f} (n={len(finals)})"
        )
