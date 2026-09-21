import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["N_PER_CLASS_tuning"]["llama3_advbench_alpaca"]
    N_candidates = ed["N_candidates"]
    per_config = ed["per_config"]
    best_N = ed["best_N"]
    best_layer = ed["best_layer"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None

# Plot 1: Val accuracy per layer for each N
try:
    plt.figure(figsize=(8, 5))
    for N in N_candidates:
        layers = sorted(per_config[N]["per_layer"].keys())
        vals = [per_config[N]["per_layer"][L]["val_acc"] for L in layers]
        plt.plot(layers, vals, "o-", label=f"N={N}")
    plt.xlabel("Layer")
    plt.ylabel("Validation Accuracy")
    plt.title(
        "Llama3 AdvBench/Alpaca: Probe Val Accuracy vs Layer\nSweep over N_PER_CLASS"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_val_acc_per_layer.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Train vs Val accuracy for best N
try:
    plt.figure(figsize=(8, 5))
    layers = sorted(per_config[best_N]["per_layer"].keys())
    tr = [per_config[best_N]["per_layer"][L]["train_acc"] for L in layers]
    va = [per_config[best_N]["per_layer"][L]["val_acc"] for L in layers]
    plt.plot(layers, tr, "o-", label="Train Acc")
    plt.plot(layers, va, "s-", label="Val Acc")
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(
        f"Llama3 AdvBench/Alpaca: Train vs Val Accuracy (N={best_N})\nBest layer={best_layer}"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_train_vs_val_bestN.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Baselines comparison at best N
try:
    plt.figure(figsize=(8, 5))
    layers = sorted(per_config[best_N]["per_layer"].keys())
    va = [per_config[best_N]["per_layer"][L]["val_acc"] for L in layers]
    ra = [per_config[best_N]["per_layer"][L]["random_acc"] for L in layers]
    sa = [per_config[best_N]["per_layer"][L]["shuffled_acc"] for L in layers]
    plt.plot(layers, va, "o-", label="Mean-diff direction")
    plt.plot(layers, ra, "s--", label="Random direction")
    plt.plot(layers, sa, "^--", label="Shuffled labels")
    plt.xlabel("Layer")
    plt.ylabel("Validation Accuracy")
    plt.title(
        f"Llama3 AdvBench/Alpaca: Probe vs Baselines (N={best_N})\nHarmfulness direction probe"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_baselines_bestN.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Best val acc per N
try:
    plt.figure(figsize=(7, 5))
    best_vals = [per_config[N]["best_val_acc"] for N in N_candidates]
    best_layers = [per_config[N]["best_layer"] for N in N_candidates]
    plt.plot(N_candidates, best_vals, "o-")
    for N, v, L in zip(N_candidates, best_vals, best_layers):
        plt.annotate(f"L={L}", (N, v), textcoords="offset points", xytext=(5, 5))
    plt.xlabel("N per class")
    plt.ylabel("Best Val Accuracy")
    plt.title(
        "Llama3 AdvBench/Alpaca: Best Val Accuracy vs N_PER_CLASS\n(Layer annotated per point)"
    )
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_bestacc_vs_N.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Confusion matrix for best config
try:
    preds = np.array(ed["predictions"])
    gt = np.array(ed["ground_truth"])
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(gt, preds):
        cm[int(t), int(p)] += 1
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
    plt.xticks([0, 1], ["Benign", "Harmful"])
    plt.yticks([0, 1], ["Benign", "Harmful"])
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(
        f"Llama3 AdvBench/Alpaca: Confusion Matrix\nBest config (N={best_N}, layer={best_layer})"
    )
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_confusion_matrix.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

try:
    print(
        f"Best N={ed['best_N']}, Best layer={ed['best_layer']}, Val Acc={ed['harmfulness_direction_probe_accuracy']:.4f}"
    )
except Exception as e:
    print(f"Error printing metric: {e}")
