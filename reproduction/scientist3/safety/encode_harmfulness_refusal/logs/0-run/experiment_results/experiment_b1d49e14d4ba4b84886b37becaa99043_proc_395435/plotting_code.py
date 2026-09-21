import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_key = "llama3_advbench_alpaca"
data = experiment_data.get(ds_key, {})

# Plot 1: Probe accuracy per layer with baselines
try:
    val_metrics = data["metrics"]["val"]
    train_metrics = data["metrics"]["train"]
    layers = [m["layer"] for m in val_metrics]
    val_accs = [m["acc"] for m in val_metrics]
    rand_accs = [m["random"] for m in val_metrics]
    shuf_accs = [m["shuffled"] for m in val_metrics]
    train_accs = [m["acc"] for m in train_metrics]

    plt.figure(figsize=(7, 5))
    plt.plot(layers, val_accs, "o-", label="Diff-of-means (val)")
    plt.plot(layers, rand_accs, "s--", label="Random direction (val)")
    plt.plot(layers, shuf_accs, "^--", label="Shuffled labels (val)")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(
        "Harmfulness Probe Accuracy per Layer\nDataset: AdvBench (harmful) vs Alpaca (benign), Model: Llama-3-8B-Instruct"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, f"{ds_key}_probe_accuracy_baselines.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Train vs Val accuracy per layer
try:
    plt.figure(figsize=(7, 5))
    plt.plot(layers, train_accs, "o-", label="Train")
    plt.plot(layers, val_accs, "s-", label="Validation")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(
        "Train vs Validation Probe Accuracy per Layer\nDataset: AdvBench/Alpaca, Model: Llama-3-8B-Instruct"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, f"{ds_key}_train_val_accuracy.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Loss per layer
try:
    train_losses = [m["loss"] for m in data["losses"]["train"]]
    val_losses = [m["loss"] for m in data["losses"]["val"]]
    plt.figure(figsize=(7, 5))
    plt.plot(layers, train_losses, "o-", label="Train Loss (1 - acc)")
    plt.plot(layers, val_losses, "s-", label="Val Loss (1 - acc)")
    plt.xlabel("Layer")
    plt.ylabel("Loss (1 - accuracy)")
    plt.title(
        "Probe Loss per Layer\nDataset: AdvBench/Alpaca, Model: Llama-3-8B-Instruct"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, f"{ds_key}_loss_per_layer.png"), bbox_inches="tight"
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Confusion matrix at best layer
try:
    preds = np.array(data["predictions"])
    gts = np.array(data["ground_truth"])
    best_layer = data.get("best_layer", "N/A")
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(gts, preds):
        cm[int(t), int(p)] += 1

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm[i, j]),
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
        f"Confusion Matrix at Best Layer ({best_layer})\nDataset: AdvBench/Alpaca, Model: Llama-3-8B-Instruct"
    )
    plt.savefig(
        os.path.join(working_dir, f"{ds_key}_confusion_matrix_best_layer.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Direction norm per layer
try:
    per_layer = data.get("per_layer", {})
    ls = sorted(per_layer.keys())
    norms = [per_layer[L]["direction_norm"] for L in ls]
    plt.figure(figsize=(7, 5))
    plt.plot(ls, norms, "o-", color="purple")
    plt.xlabel("Layer")
    plt.ylabel("||mu_harmful - mu_benign||")
    plt.title(
        "Mean-Difference Direction Norm per Layer\nDataset: AdvBench/Alpaca, Model: Llama-3-8B-Instruct"
    )
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, f"{ds_key}_direction_norm_per_layer.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print evaluation metric
try:
    print(f"Best layer: {data.get('best_layer')}")
    print(
        f"Harmfulness direction probe accuracy: {data.get('harmfulness_direction_probe_accuracy'):.4f}"
    )
except Exception as e:
    print(f"Error printing metric: {e}")
