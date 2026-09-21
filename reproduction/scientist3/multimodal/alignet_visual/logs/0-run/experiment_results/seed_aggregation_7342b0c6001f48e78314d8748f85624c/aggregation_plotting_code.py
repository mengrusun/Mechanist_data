import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_04-06-03_alignet_visual_attempt_1/logs/0-run/experiment_results/experiment_f8173464ecba48978473d1215ae451ca_proc_2848855/experiment_data.npy"
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


def sem(x, axis=0):
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    return (
        np.std(x, axis=axis, ddof=1) / np.sqrt(n)
        if n > 1
        else np.zeros_like(np.mean(x, axis=axis))
    )


# Collect per-run data across N_EPOCHS settings within each experiment
# Aggregate across the N_EPOCHS runs by truncating to common length
def collect_curves(all_experiment_data, key_type, split):
    # returns dict: run_key -> list of arrays (across experiments)
    curves_per_run = {}
    for exp in all_experiment_data:
        runs = exp.get("N_EPOCHS", {})
        for rk, run_data in runs.items():
            ed = run_data["THINGS"]
            if key_type == "loss":
                arr = np.array(ed["losses"][split])
            else:
                arr = np.array(ed["metrics"][split])
            curves_per_run.setdefault(rk, []).append(arr)
    return curves_per_run


def aggregate_all_runs(all_experiment_data, key_type, split):
    # Aggregate ALL runs across all N_EPOCHS settings and experiments by truncating to shortest
    curves = []
    for exp in all_experiment_data:
        runs = exp.get("N_EPOCHS", {})
        for rk, run_data in runs.items():
            ed = run_data["THINGS"]
            if key_type == "loss":
                curves.append(np.array(ed["losses"][split]))
            else:
                curves.append(np.array(ed["metrics"][split]))
    if not curves:
        return None, None, None
    min_len = min(len(c) for c in curves)
    stacked = np.stack([c[:min_len] for c in curves], axis=0)
    mean = np.mean(stacked, axis=0)
    se = sem(stacked, axis=0)
    return mean, se, stacked.shape[0]


baseline_ooo = None
teacher_ooo = None
if all_experiment_data:
    first_run = next(iter(all_experiment_data[0]["N_EPOCHS"].values()))
    baseline_ooo = first_run["THINGS"].get("baseline_ooo")
    teacher_ooo = first_run["THINGS"].get("teacher_ooo")

# Plot 1: Aggregated training loss with SEM
try:
    plt.figure(figsize=(8, 5))
    mean, se, n = aggregate_all_runs(all_experiment_data, "loss", "train")
    if mean is not None:
        epochs = np.arange(len(mean))
        plt.plot(epochs, mean, label=f"Mean Train Loss (n={n})", color="blue")
        plt.fill_between(
            epochs, mean - se, mean + se, alpha=0.3, color="blue", label="± SEM"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss (MSE)")
    plt.title(
        "THINGS Dataset: Aggregated Training Loss\nMean ± SEM across N_EPOCHS settings"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "THINGS_agg_train_loss.png"), dpi=100)
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Aggregated validation loss with SEM
try:
    plt.figure(figsize=(8, 5))
    mean, se, n = aggregate_all_runs(all_experiment_data, "loss", "val")
    if mean is not None:
        epochs = np.arange(len(mean))
        plt.plot(epochs, mean, label=f"Mean Val Loss (n={n})", color="orange")
        plt.fill_between(
            epochs, mean - se, mean + se, alpha=0.3, color="orange", label="± SEM"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Validation Loss (MSE)")
    plt.title(
        "THINGS Dataset: Aggregated Validation Loss\nMean ± SEM across N_EPOCHS settings"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "THINGS_agg_val_loss.png"), dpi=100)
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Aggregated OOO accuracy with SEM
try:
    plt.figure(figsize=(8, 5))
    mean, se, n = aggregate_all_runs(all_experiment_data, "metric", "val")
    if mean is not None:
        epochs = np.arange(len(mean))
        plt.plot(epochs, mean, label=f"Mean Aligned OOO (n={n})", color="green")
        plt.fill_between(
            epochs, mean - se, mean + se, alpha=0.3, color="green", label="± SEM"
        )
    if baseline_ooo is not None:
        plt.axhline(
            baseline_ooo,
            color="k",
            linestyle="--",
            label=f"DINOv2 baseline ({baseline_ooo:.3f})",
        )
    if teacher_ooo is not None:
        plt.axhline(
            teacher_ooo,
            color="r",
            linestyle="--",
            label=f"SigLIP teacher ({teacher_ooo:.3f})",
        )
    plt.xlabel("Epoch")
    plt.ylabel("Aligned OOO Accuracy")
    plt.title(
        "THINGS Dataset: Aggregated OOO Accuracy\nMean ± SEM across N_EPOCHS settings"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "THINGS_agg_ooo_accuracy.png"), dpi=100)
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Per-N_EPOCHS mean curves with SEM (across experiments)
try:
    plt.figure(figsize=(8, 5))
    curves_per_run = collect_curves(all_experiment_data, "metric", "val")
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(curves_per_run)))
    for (rk, curves), c in zip(
        sorted(curves_per_run.items(), key=lambda x: int(x[0].split("_")[1])), colors
    ):
        min_len = min(len(cv) for cv in curves)
        stacked = np.stack([cv[:min_len] for cv in curves], axis=0)
        m = np.mean(stacked, axis=0)
        s = sem(stacked, axis=0)
        ep = np.arange(len(m))
        plt.plot(ep, m, label=f"{rk} (n={stacked.shape[0]})", color=c)
        plt.fill_between(ep, m - s, m + s, alpha=0.25, color=c)
    if baseline_ooo is not None:
        plt.axhline(
            baseline_ooo,
            color="k",
            linestyle="--",
            label=f"Baseline ({baseline_ooo:.3f})",
        )
    if teacher_ooo is not None:
        plt.axhline(
            teacher_ooo, color="r", linestyle="--", label=f"Teacher ({teacher_ooo:.3f})"
        )
    plt.xlabel("Epoch")
    plt.ylabel("Aligned OOO Accuracy")
    plt.title("THINGS Dataset: OOO Accuracy by N_EPOCHS\nMean ± SEM across experiments")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "THINGS_per_setting_ooo_curves.png"), dpi=100)
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Final OOO bar chart with error bars across experiments per N_EPOCHS setting
try:
    plt.figure(figsize=(8, 5))
    finals_per_run = {}
    for exp in all_experiment_data:
        for rk, run_data in exp.get("N_EPOCHS", {}).items():
            finals_per_run.setdefault(rk, []).append(
                run_data["THINGS"]["final_aligned_ooo"]
            )
    sorted_keys = sorted(finals_per_run.keys(), key=lambda k: int(k.split("_")[1]))
    means = [np.mean(finals_per_run[k]) for k in sorted_keys]
    sems = [
        sem(np.array(finals_per_run[k])) if len(finals_per_run[k]) > 1 else 0
        for k in sorted_keys
    ]
    x = np.arange(len(sorted_keys))
    plt.bar(
        x, means, yerr=sems, color="steelblue", capsize=5, label="Mean Final OOO ± SEM"
    )
    if baseline_ooo is not None:
        plt.axhline(
            baseline_ooo,
            color="k",
            linestyle="--",
            label=f"Baseline ({baseline_ooo:.3f})",
        )
    if teacher_ooo is not None:
        plt.axhline(
            teacher_ooo, color="r", linestyle="--", label=f"Teacher ({teacher_ooo:.3f})"
        )
    plt.xticks(x, sorted_keys, rotation=30)
    plt.ylabel("Final Aligned OOO Accuracy")
    plt.title("THINGS Dataset: Final OOO by N_EPOCHS\nMean ± SEM across experiments")
    plt.legend()
    for xi, m in zip(x, means):
        plt.text(xi, m + 0.002, f"{m:.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "THINGS_agg_final_ooo_bar.png"), dpi=100)
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print aggregate eval metrics
print("=== Aggregate Evaluation Metrics ===")
if baseline_ooo is not None:
    print(f"Baseline DINOv2 OOO: {baseline_ooo:.4f}")
if teacher_ooo is not None:
    print(f"Teacher SigLIP OOO:  {teacher_ooo:.4f}")
finals_per_run = {}
for exp in all_experiment_data:
    for rk, run_data in exp.get("N_EPOCHS", {}).items():
        finals_per_run.setdefault(rk, []).append(
            run_data["THINGS"]["final_aligned_ooo"]
        )
for rk in sorted(finals_per_run.keys(), key=lambda k: int(k.split("_")[1])):
    vals = np.array(finals_per_run[rk])
    m = vals.mean()
    s = sem(vals) if len(vals) > 1 else 0.0
    print(f"  {rk}: Final Aligned OOO = {m:.4f} ± {s:.4f} (n={len(vals)})")
