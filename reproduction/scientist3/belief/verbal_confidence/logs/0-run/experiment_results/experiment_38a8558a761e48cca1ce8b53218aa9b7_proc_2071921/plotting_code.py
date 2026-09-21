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

D = experiment_data.get("answer_span_mean_pooling", {}).get(
    "triviaqa_gemma3_27b_pt", {}
)

# Plot 1: Per-layer probe accuracy
try:
    layer_indices = D.get("layer_indices", [])
    conf_acc = D.get("per_layer_confidence_acc", [])
    corr_acc = D.get("per_layer_correctness_acc", [])
    ctrl_acc = D.get("per_layer_control_conf_acc", [])
    maj_conf = D.get("majority_baseline_conf", 0)
    maj_corr = D.get("majority_baseline_corr", 0)

    plt.figure(figsize=(10, 6))
    plt.plot(layer_indices, conf_acc, "o-", label="Answer-span -> Confidence")
    plt.plot(layer_indices, ctrl_acc, "s--", label="Pre-answer (control) -> Confidence")
    plt.plot(layer_indices, corr_acc, "^-", label="Answer-span -> Correctness")
    plt.axhline(
        maj_conf, color="gray", linestyle=":", label=f"Majority conf ({maj_conf:.3f})"
    )
    plt.axhline(
        maj_corr, color="black", linestyle=":", label=f"Majority corr ({maj_corr:.3f})"
    )
    plt.xlabel("Layer")
    plt.ylabel("Probe Accuracy")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Per-Layer Probe Accuracy\nAnswer-Span Mean Pooling vs Control"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_per_layer_probe_accuracy.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Confidence distribution histogram (correct vs incorrect)
try:
    confs = np.array(D.get("raw_confidences", []))
    corrs = np.array(D.get("raw_correctness", []))
    plt.figure(figsize=(10, 6))
    if len(confs) > 0:
        plt.hist(confs[corrs == 1], bins=20, alpha=0.6, label="Correct", color="green")
        plt.hist(confs[corrs == 0], bins=20, alpha=0.6, label="Incorrect", color="red")
    plt.xlabel("Model Continuous Confidence (0-100)")
    plt.ylabel("Count")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Confidence Distribution\nLeft: Correct predictions, Right: Incorrect predictions (overlaid)"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_confidence_distribution.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Calibration plot (confidence vs accuracy)
try:
    confs = np.array(D.get("raw_confidences", []))
    corrs = np.array(D.get("raw_correctness", []))
    if len(confs) > 0:
        bins = np.linspace(0, 100, 11)
        bin_ids = np.digitize(confs, bins) - 1
        bin_ids = np.clip(bin_ids, 0, 9)
        bin_centers, bin_accs, bin_counts = [], [], []
        for b in range(10):
            mask = bin_ids == b
            if mask.sum() > 0:
                bin_centers.append((bins[b] + bins[b + 1]) / 2)
                bin_accs.append(corrs[mask].mean())
                bin_counts.append(mask.sum())
        plt.figure(figsize=(10, 6))
        plt.plot([0, 100], [0, 1], "k--", label="Perfect calibration")
        plt.scatter(
            bin_centers,
            bin_accs,
            s=np.array(bin_counts) * 10,
            alpha=0.7,
            label="Empirical accuracy",
        )
        plt.plot(bin_centers, bin_accs, "b-", alpha=0.5)
        plt.xlabel("Predicted Confidence (0-100)")
        plt.ylabel("Empirical Accuracy")
        plt.title(
            "TriviaQA (Gemma-3-27b-pt): Calibration Plot\nMarker size proportional to bin count"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "triviaqa_calibration_plot.png"), dpi=120)
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Best layer summary bar chart
try:
    best_conf_acc = D.get("cache_probe_accuracy", 0)
    maj_conf = D.get("majority_baseline_conf", 0)
    maj_corr = D.get("majority_baseline_corr", 0)
    corr_accs = D.get("per_layer_correctness_acc", [])
    best_corr_acc = max(corr_accs) if len(corr_accs) > 0 else 0

    labels = [
        "Confidence Probe\n(Best Layer)",
        "Confidence\nMajority Baseline",
        "Correctness Probe\n(Best Layer)",
        "Correctness\nMajority Baseline",
    ]
    values = [best_conf_acc, maj_conf, best_corr_acc, maj_corr]
    colors = ["steelblue", "lightgray", "darkorange", "lightgray"]
    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=colors)
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.0)
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Best Probe vs Majority Baseline\nLeft: Confidence Prediction, Right: Correctness Prediction"
    )
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_best_probe_vs_baseline.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Print summary metrics
try:
    print(f"Best layer (confidence): {D.get('best_layer_conf')}")
    print(f"Best layer (correctness): {D.get('best_layer_corr')}")
    print(f"Cache probe accuracy: {D.get('cache_probe_accuracy'):.4f}")
    print(f"Majority baseline (conf): {D.get('majority_baseline_conf'):.4f}")
    print(f"Majority baseline (corr): {D.get('majority_baseline_corr'):.4f}")
    corrs = np.array(D.get("raw_correctness", []))
    if len(corrs) > 0:
        print(f"Overall correctness rate: {corrs.mean():.4f}")
except Exception as e:
    print(f"Error printing metrics: {e}")
