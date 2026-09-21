import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_02-20-28_encode_harmfulness_refusal_attempt_1/logs/0-run/experiment_results/experiment_df77239ac0e24ba5a7cb35d000f5cf59_proc_395435/experiment_data.npy"
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        root = os.getenv("AI_SCIENTIST_ROOT", "")
        ed = np.load(os.path.join(root, p), allow_pickle=True).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

ds_key = "llama3_advbench_alpaca"

# Collect per-run arrays
runs = []
for ed in all_experiment_data:
    data = ed.get(ds_key, {})
    if not data:
        continue
    try:
        val_metrics = data["metrics"]["val"]
        train_metrics = data["metrics"]["train"]
        layers = np.array([m["layer"] for m in val_metrics])
        run = {
            "layers": layers,
            "val_acc": np.array([m["acc"] for m in val_metrics]),
            "train_acc": np.array([m["acc"] for m in train_metrics]),
            "rand": np.array([m["random"] for m in val_metrics]),
            "shuf": np.array([m["shuffled"] for m in val_metrics]),
            "train_loss": np.array([m["loss"] for m in data["losses"]["train"]]),
            "val_loss": np.array([m["loss"] for m in data["losses"]["val"]]),
            "preds": np.array(data.get("predictions", [])),
            "gts": np.array(data.get("ground_truth", [])),
            "best_layer": data.get("best_layer"),
            "hd_acc": data.get("harmfulness_direction_probe_accuracy"),
            "per_layer": data.get("per_layer", {}),
        }
        runs.append(run)
    except Exception as e:
        print(f"Error extracting run: {e}")


def stack_mean_sem(key, runs):
    arrs = [r[key] for r in runs]
    # ensure same shape
    min_len = min(a.shape[0] for a in arrs)
    arrs = np.stack([a[:min_len] for a in arrs], axis=0)
    mean = arrs.mean(axis=0)
    sem = (
        arrs.std(axis=0, ddof=1) / np.sqrt(arrs.shape[0])
        if arrs.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem, arrs.shape[0]


# Plot 1: Probe accuracy per layer with baselines (mean ± SEM)
try:
    if runs:
        layers = runs[0]["layers"]
        val_mean, val_sem, n = stack_mean_sem("val_acc", runs)
        rand_mean, rand_sem, _ = stack_mean_sem("rand", runs)
        shuf_mean, shuf_sem, _ = stack_mean_sem("shuf", runs)
        plt.figure(figsize=(7, 5))
        plt.errorbar(
            layers,
            val_mean,
            yerr=val_sem,
            fmt="o-",
            label=f"Diff-of-means val (mean±SEM, n={n})",
            capsize=3,
        )
        plt.errorbar(
            layers,
            rand_mean,
            yerr=rand_sem,
            fmt="s--",
            label="Random direction (mean±SEM)",
            capsize=3,
        )
        plt.errorbar(
            layers,
            shuf_mean,
            yerr=shuf_sem,
            fmt="^--",
            label="Shuffled labels (mean±SEM)",
            capsize=3,
        )
        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        plt.title(
            "Aggregated Probe Accuracy per Layer (mean ± SEM)\nDataset: AdvBench/Alpaca, Model: Llama-3-8B-Instruct"
        )
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(working_dir, f"{ds_key}_agg_probe_accuracy_baselines.png"),
            bbox_inches="tight",
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Train vs Val accuracy per layer (aggregated)
try:
    if runs:
        train_mean, train_sem, n = stack_mean_sem("train_acc", runs)
        val_mean, val_sem, _ = stack_mean_sem("val_acc", runs)
        plt.figure(figsize=(7, 5))
        plt.errorbar(
            layers,
            train_mean,
            yerr=train_sem,
            fmt="o-",
            label=f"Train (mean±SEM, n={n})",
            capsize=3,
        )
        plt.errorbar(
            layers,
            val_mean,
            yerr=val_sem,
            fmt="s-",
            label="Validation (mean±SEM)",
            capsize=3,
        )
        plt.xlabel("Layer")
        plt.ylabel("Accuracy")
        plt.title(
            "Aggregated Train vs Val Probe Accuracy (mean ± SEM)\nDataset: AdvBench/Alpaca, Model: Llama-3-8B-Instruct"
        )
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(working_dir, f"{ds_key}_agg_train_val_accuracy.png"),
            bbox_inches="tight",
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Loss per layer (aggregated)
try:
    if runs:
        tl_mean, tl_sem, n = stack_mean_sem("train_loss", runs)
        vl_mean, vl_sem, _ = stack_mean_sem("val_loss", runs)
        plt.figure(figsize=(7, 5))
        plt.errorbar(
            layers,
            tl_mean,
            yerr=tl_sem,
            fmt="o-",
            label=f"Train Loss (mean±SEM, n={n})",
            capsize=3,
        )
        plt.errorbar(
            layers,
            vl_mean,
            yerr=vl_sem,
            fmt="s-",
            label="Val Loss (mean±SEM)",
            capsize=3,
        )
        plt.xlabel("Layer")
        plt.ylabel("Loss (1 - accuracy)")
        plt.title(
            "Aggregated Probe Loss per Layer (mean ± SEM)\nDataset: AdvBench/Alpaca, Model: Llama-3-8B-Instruct"
        )
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(working_dir, f"{ds_key}_agg_loss_per_layer.png"),
            bbox_inches="tight",
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Aggregated confusion matrix (sum across runs)
try:
    if runs:
        cm_total = np.zeros((2, 2), dtype=int)
        for r in runs:
            for t, p in zip(r["gts"], r["preds"]):
                cm_total[int(t), int(p)] += 1
        plt.figure(figsize=(5, 4))
        plt.imshow(cm_total, cmap="Blues")
        plt.colorbar()
        for i in range(2):
            for j in range(2):
                plt.text(
                    j,
                    i,
                    str(cm_total[i, j]),
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=14,
                )
        plt.xticks([0, 1], ["Benign", "Harmful"])
        plt.yticks([0, 1], ["Benign", "Harmful"])
        plt.xlabel("Predicted")
        plt.ylabel("Ground Truth")
        plt.title(
            f"Aggregated Confusion Matrix (sum over {len(runs)} runs)\nDataset: AdvBench/Alpaca, Model: Llama-3-8B-Instruct"
        )
        plt.savefig(
            os.path.join(working_dir, f"{ds_key}_agg_confusion_matrix.png"),
            bbox_inches="tight",
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Direction norm per layer (aggregated)
try:
    if runs and all(r["per_layer"] for r in runs):
        common_layers = sorted(
            set.intersection(*[set(r["per_layer"].keys()) for r in runs])
        )
        norm_arr = np.array(
            [[r["per_layer"][L]["direction_norm"] for L in common_layers] for r in runs]
        )
        mean = norm_arr.mean(axis=0)
        sem = (
            norm_arr.std(axis=0, ddof=1) / np.sqrt(norm_arr.shape[0])
            if norm_arr.shape[0] > 1
            else np.zeros_like(mean)
        )
        plt.figure(figsize=(7, 5))
        plt.errorbar(
            common_layers,
            mean,
            yerr=sem,
            fmt="o-",
            color="purple",
            label=f"Direction norm (mean±SEM, n={norm_arr.shape[0]})",
            capsize=3,
        )
        plt.xlabel("Layer")
        plt.ylabel("||mu_harmful - mu_benign||")
        plt.title(
            "Aggregated Mean-Difference Direction Norm per Layer\nDataset: AdvBench/Alpaca, Model: Llama-3-8B-Instruct"
        )
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(working_dir, f"{ds_key}_agg_direction_norm.png"),
            bbox_inches="tight",
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print aggregated evaluation metrics
try:
    hd_accs = [r["hd_acc"] for r in runs if r["hd_acc"] is not None]
    best_layers = [r["best_layer"] for r in runs]
    print(f"Number of runs aggregated: {len(runs)}")
    print(f"Best layers per run: {best_layers}")
    if hd_accs:
        arr = np.array(hd_accs)
        sem = arr.std(ddof=1) / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
        print(
            f"Harmfulness direction probe accuracy: mean={arr.mean():.4f}, SEM={sem:.4f}"
        )
except Exception as e:
    print(f"Error printing metric: {e}")
