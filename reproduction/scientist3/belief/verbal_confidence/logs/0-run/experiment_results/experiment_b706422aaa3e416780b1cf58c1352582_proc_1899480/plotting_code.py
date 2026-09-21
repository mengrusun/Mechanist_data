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

key = "triviaqa_gemma3_27b_pt"
data = experiment_data.get(key, {})

# Plot 1: Per-layer probe accuracy
try:
    layer_indices = data.get("layer_indices", [])
    conf_acc = data.get("per_layer_confidence_acc", [])
    corr_acc = data.get("per_layer_correctness_acc", [])
    ctrl_acc = data.get("per_layer_control_conf_acc", [])
    maj_conf = data.get("majority_baseline_conf", None)
    maj_corr = data.get("majority_baseline_corr", None)

    plt.figure(figsize=(10, 6))
    if len(layer_indices) and len(conf_acc):
        plt.plot(layer_indices, conf_acc, "o-", label="Post-answer -> Confidence")
    if len(layer_indices) and len(ctrl_acc):
        plt.plot(
            layer_indices, ctrl_acc, "s--", label="Pre-answer (control) -> Confidence"
        )
    if len(layer_indices) and len(corr_acc):
        plt.plot(layer_indices, corr_acc, "^-", label="Post-answer -> Correctness")
    if maj_conf is not None:
        plt.axhline(
            maj_conf,
            color="gray",
            linestyle=":",
            label=f"Majority conf ({maj_conf:.3f})",
        )
    if maj_corr is not None:
        plt.axhline(
            maj_corr,
            color="black",
            linestyle=":",
            label=f"Majority corr ({maj_corr:.3f})",
        )
    plt.xlabel("Layer")
    plt.ylabel("Probe accuracy")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Linear Probe Accuracy vs Layer\nConfidence & Correctness Probes with Pre-answer Control"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_gemma3_27b_pt_probe_accuracy_vs_layer.png"),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Distribution of raw verbal confidences
try:
    raw_conf = np.array(data.get("raw_confidences", []))
    plt.figure(figsize=(8, 5))
    if len(raw_conf):
        plt.hist(raw_conf, bins=20, color="steelblue", edgecolor="black")
        plt.axvline(
            raw_conf.mean(),
            color="red",
            linestyle="--",
            label=f"mean={raw_conf.mean():.2f}",
        )
        plt.legend()
    plt.xlabel("Continuous verbal confidence (expected first digit * 10)")
    plt.ylabel("Count")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Distribution of Verbal Confidences\nExpected First-Digit Confidence Across 200 Samples"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_gemma3_27b_pt_confidence_distribution.png"),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Confidence vs correctness (calibration-style boxplot)
try:
    raw_conf = np.array(data.get("raw_confidences", []))
    raw_corr = np.array(data.get("raw_correctness", []))
    plt.figure(figsize=(8, 5))
    if len(raw_conf) and len(raw_corr) and len(raw_conf) == len(raw_corr):
        c0 = raw_conf[raw_corr == 0]
        c1 = raw_conf[raw_corr == 1]
        plt.boxplot([c0, c1], labels=["Incorrect", "Correct"])
        plt.scatter(
            np.ones(len(c0)) + np.random.uniform(-0.05, 0.05, len(c0)),
            c0,
            alpha=0.4,
            color="red",
            s=15,
        )
        plt.scatter(
            2 * np.ones(len(c1)) + np.random.uniform(-0.05, 0.05, len(c1)),
            c1,
            alpha=0.4,
            color="green",
            s=15,
        )
    plt.ylabel("Verbal confidence")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Verbal Confidence by Correctness\nLeft: Incorrect Answers, Right: Correct Answers"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, "triviaqa_gemma3_27b_pt_confidence_vs_correctness.png"
        ),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Calibration curve - accuracy per confidence bin
try:
    raw_conf = np.array(data.get("raw_confidences", []))
    raw_corr = np.array(data.get("raw_correctness", []))
    plt.figure(figsize=(8, 5))
    if len(raw_conf) and len(raw_corr):
        bins = np.linspace(raw_conf.min(), raw_conf.max() + 1e-6, 6)
        bin_ids = np.digitize(raw_conf, bins) - 1
        bin_ids = np.clip(bin_ids, 0, len(bins) - 2)
        centers, accs, counts = [], [], []
        for b in range(len(bins) - 1):
            mask = bin_ids == b
            if mask.sum() > 0:
                centers.append((bins[b] + bins[b + 1]) / 2)
                accs.append(raw_corr[mask].mean())
                counts.append(mask.sum())
        plt.plot(centers, accs, "o-", color="purple", label="Accuracy per bin")
        for x, y, c in zip(centers, accs, counts):
            plt.annotate(
                f"n={c}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
            )
        plt.axhline(
            raw_corr.mean(),
            color="gray",
            linestyle=":",
            label=f"overall acc={raw_corr.mean():.3f}",
        )
    plt.xlabel("Verbal confidence (bin center)")
    plt.ylabel("Empirical accuracy")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Calibration Curve\nEmpirical Accuracy vs Reported Verbal Confidence"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_gemma3_27b_pt_calibration_curve.png"),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Summary bar chart of best probe accuracies vs baselines
try:
    best_conf = None
    best_corr = None
    val_metrics = data.get("metrics", {}).get("val", [])
    if val_metrics:
        best_conf = val_metrics[0].get("cache_probe_accuracy", None)
        best_corr = val_metrics[0].get("corr_probe_accuracy", None)
    maj_conf = data.get("majority_baseline_conf", None)
    maj_corr = data.get("majority_baseline_corr", None)

    labels = [
        "Conf Probe\n(best layer)",
        "Conf Majority",
        "Corr Probe\n(best layer)",
        "Corr Majority",
    ]
    values = [best_conf or 0, maj_conf or 0, best_corr or 0, maj_corr or 0]
    colors = ["steelblue", "lightsteelblue", "seagreen", "lightgreen"]
    plt.figure(figsize=(8, 5))
    plt.bar(labels, values, color=colors, edgecolor="black")
    for i, v in enumerate(values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Best Probe Accuracy vs Majority Baseline\nLeft Pair: Confidence, Right Pair: Correctness"
    )
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, "triviaqa_gemma3_27b_pt_probe_vs_baseline_summary.png"
        ),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print key metrics
try:
    print("=== Key Metrics ===")
    print(f"Majority baseline (confidence): {data.get('majority_baseline_conf')}")
    print(f"Majority baseline (correctness): {data.get('majority_baseline_corr')}")
    print(f"Best layer (confidence): {data.get('best_layer_conf')}")
    print(f"Best layer (correctness): {data.get('best_layer_corr')}")
    print(f"Cache probe accuracy: {data.get('cache_probe_accuracy')}")
except Exception as e:
    print(f"Error printing metrics: {e}")
