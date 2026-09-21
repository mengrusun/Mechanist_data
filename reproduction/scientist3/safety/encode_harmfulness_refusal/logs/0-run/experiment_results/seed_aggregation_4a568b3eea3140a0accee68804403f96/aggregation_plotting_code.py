import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_02-20-28_encode_harmfulness_refusal_attempt_1/logs/0-run/experiment_results/experiment_754e57bc92c44b239b61ff24241c40bf_proc_647101/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        experiment_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


def sem(arr, axis=0):
    arr = np.asarray(arr, dtype=float)
    n = arr.shape[axis]
    if n <= 1:
        return np.zeros(arr.shape[1:]) if arr.ndim > 1 else 0.0
    return np.std(arr, axis=axis, ddof=1) / np.sqrt(n)


# Gather common structure
try:
    eds = [
        ed["N_PER_CLASS_tuning"]["llama3_advbench_alpaca"] for ed in all_experiment_data
    ]
    # Assume same N_candidates across runs
    N_candidates = eds[0]["N_candidates"]
    # Determine common layers per N
    common_layers = {}
    for N in N_candidates:
        layers_sets = [set(ed["per_config"][N]["per_layer"].keys()) for ed in eds]
        common = sorted(set.intersection(*layers_sets))
        common_layers[N] = common
    best_N = eds[0]["best_N"]
    best_layer = eds[0]["best_layer"]
except Exception as e:
    print(f"Error preparing aggregated data: {e}")
    eds = []

# Plot 1: Aggregated Val accuracy per layer for each N (mean +/- SEM)
try:
    plt.figure(figsize=(9, 5))
    for N in N_candidates:
        layers = common_layers[N]
        vals = np.array(
            [
                [ed["per_config"][N]["per_layer"][L]["val_acc"] for L in layers]
                for ed in eds
            ]
        )
        mean = vals.mean(axis=0)
        se = sem(vals, axis=0)
        (line,) = plt.plot(layers, mean, "o-", label=f"N={N} mean")
        plt.fill_between(
            layers,
            mean - se,
            mean + se,
            alpha=0.2,
            color=line.get_color(),
            label=f"N={N} ±SEM",
        )
    plt.xlabel("Layer")
    plt.ylabel("Validation Accuracy (mean ± SEM)")
    plt.title(
        "Llama3 AdvBench/Alpaca: Aggregated Val Accuracy vs Layer\nMean ± SEM across runs"
    )
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_val_acc_per_layer.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Aggregated Train vs Val at best N
try:
    plt.figure(figsize=(9, 5))
    layers = common_layers[best_N]
    tr = np.array(
        [
            [ed["per_config"][best_N]["per_layer"][L]["train_acc"] for L in layers]
            for ed in eds
        ]
    )
    va = np.array(
        [
            [ed["per_config"][best_N]["per_layer"][L]["val_acc"] for L in layers]
            for ed in eds
        ]
    )
    for arr, lbl, mk in [(tr, "Train Acc", "o-"), (va, "Val Acc", "s-")]:
        m = arr.mean(axis=0)
        s = sem(arr, axis=0)
        (line,) = plt.plot(layers, m, mk, label=f"{lbl} mean")
        plt.fill_between(
            layers, m - s, m + s, alpha=0.2, color=line.get_color(), label=f"{lbl} ±SEM"
        )
    plt.xlabel("Layer")
    plt.ylabel("Accuracy (mean ± SEM)")
    plt.title(
        f"Llama3 AdvBench/Alpaca: Train vs Val (N={best_N})\nAggregated across runs"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_train_vs_val_bestN.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Aggregated baselines comparison at best N
try:
    plt.figure(figsize=(9, 5))
    layers = common_layers[best_N]
    va = np.array(
        [
            [ed["per_config"][best_N]["per_layer"][L]["val_acc"] for L in layers]
            for ed in eds
        ]
    )
    ra = np.array(
        [
            [ed["per_config"][best_N]["per_layer"][L]["random_acc"] for L in layers]
            for ed in eds
        ]
    )
    sa = np.array(
        [
            [ed["per_config"][best_N]["per_layer"][L]["shuffled_acc"] for L in layers]
            for ed in eds
        ]
    )
    for arr, lbl, mk in [
        (va, "Mean-diff direction", "o-"),
        (ra, "Random direction", "s--"),
        (sa, "Shuffled labels", "^--"),
    ]:
        m = arr.mean(axis=0)
        s = sem(arr, axis=0)
        (line,) = plt.plot(layers, m, mk, label=f"{lbl} mean")
        plt.fill_between(
            layers, m - s, m + s, alpha=0.2, color=line.get_color(), label=f"{lbl} ±SEM"
        )
    plt.xlabel("Layer")
    plt.ylabel("Validation Accuracy (mean ± SEM)")
    plt.title(
        f"Llama3 AdvBench/Alpaca: Probe vs Baselines (N={best_N})\nAggregated across runs"
    )
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_baselines_bestN.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Aggregated Best val acc per N (mean ± SEM as error bars)
try:
    plt.figure(figsize=(7, 5))
    best_vals = np.array(
        [[ed["per_config"][N]["best_val_acc"] for N in N_candidates] for ed in eds]
    )
    m = best_vals.mean(axis=0)
    s = sem(best_vals, axis=0)
    plt.errorbar(
        N_candidates, m, yerr=s, fmt="o-", capsize=4, label="Mean best val acc ±SEM"
    )
    plt.xlabel("N per class")
    plt.ylabel("Best Val Accuracy (mean ± SEM)")
    plt.title("Llama3 AdvBench/Alpaca: Best Val Accuracy vs N\nAggregated across runs")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_bestacc_vs_N.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Aggregated confusion matrix (sum across runs)
try:
    cm = np.zeros((2, 2), dtype=float)
    per_run_cms = []
    for ed in eds:
        preds = np.array(ed["predictions"])
        gt = np.array(ed["ground_truth"])
        run_cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(gt, preds):
            run_cm[int(t), int(p)] += 1
        per_run_cms.append(run_cm)
        cm += run_cm
    per_run_cms = np.array(per_run_cms, dtype=float)
    mean_cm = per_run_cms.mean(axis=0)
    se_cm = sem(per_run_cms, axis=0)
    plt.figure(figsize=(5.5, 4.5))
    plt.imshow(mean_cm, cmap="Blues")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                f"{mean_cm[i,j]:.1f}\n±{se_cm[i,j]:.1f}",
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )
    plt.xticks([0, 1], ["Benign", "Harmful"])
    plt.yticks([0, 1], ["Benign", "Harmful"])
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(
        f"Llama3 AdvBench/Alpaca: Confusion Matrix\nMean ± SEM across runs (N={best_N}, layer={best_layer})"
    )
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_confusion_matrix.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print aggregated metric
try:
    accs = [ed["harmfulness_direction_probe_accuracy"] for ed in eds]
    accs = np.array(accs, dtype=float)
    print(f"Number of runs: {len(eds)}")
    print(
        f"Harmfulness direction probe accuracy: mean={accs.mean():.4f}, SEM={sem(accs):.4f}"
    )
    print(f"Best N (from first run): {best_N}, Best layer: {best_layer}")
except Exception as e:
    print(f"Error printing metric: {e}")
