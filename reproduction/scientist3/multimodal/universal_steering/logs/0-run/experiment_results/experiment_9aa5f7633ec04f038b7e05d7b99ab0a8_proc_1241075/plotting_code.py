import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data = experiment_data.get("concept_benchmark", {})
per_concept = data.get("per_concept", [])
cfg = data.get("config", {})
alpha = cfg.get("alpha", "?")
layer = cfg.get("steer_layer", "?")

# Plot 1: success rate by concept class
try:
    plt.figure(figsize=(8, 5))
    classes = sorted(set(r["class"] for r in per_concept))
    cls_success = {c: [] for c in classes}
    cls_base = {c: [] for c in classes}
    for r in per_concept:
        cls_success[r["class"]].append(r["steered_success"] / r["total"])
        cls_base[r["class"]].append(r["baseline_success"] / r["total"])
    xs = np.arange(len(classes))
    steered_means = [np.mean(cls_success[c]) for c in classes]
    base_means = [np.mean(cls_base[c]) for c in classes]
    plt.bar(xs - 0.2, base_means, width=0.4, label="baseline")
    plt.bar(xs + 0.2, steered_means, width=0.4, label="steered")
    plt.xticks(xs, classes, rotation=30)
    plt.ylabel("Success rate")
    plt.title(
        f"Concept Benchmark: Success Rate by Class\nLeft bar: Baseline, Right bar: Steered (α={alpha}, layer={layer})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "concept_benchmark_success_by_class.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: per-concept success rate comparison
try:
    plt.figure(figsize=(12, 6))
    concepts = [r["concept"] for r in per_concept]
    base_rates = [r["baseline_success"] / r["total"] for r in per_concept]
    steer_rates = [r["steered_success"] / r["total"] for r in per_concept]
    xs = np.arange(len(concepts))
    plt.bar(xs - 0.2, base_rates, width=0.4, label="baseline")
    plt.bar(xs + 0.2, steer_rates, width=0.4, label="steered")
    plt.xticks(xs, concepts, rotation=75, fontsize=8)
    plt.ylabel("Success rate")
    plt.title(
        f"Concept Benchmark: Per-Concept Success Rates\nLeft: Baseline, Right: Steered (α={alpha}, layer={layer})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "concept_benchmark_per_concept_success.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: running success rate over concepts evaluated
try:
    val_metrics = data.get("metrics", {}).get("val", [])
    if val_metrics:
        idx = [m["concept_index"] for m in val_metrics]
        steered_run = [m["steered_success_rate_running"] for m in val_metrics]
        baseline_run = [m["baseline_success_rate_running"] for m in val_metrics]
        plt.figure(figsize=(8, 5))
        plt.plot(idx, baseline_run, label="baseline (running)", marker="o")
        plt.plot(idx, steered_run, label="steered (running)", marker="s")
        plt.xlabel("Concept index")
        plt.ylabel("Running success rate")
        plt.title(
            "Concept Benchmark: Running Success Rate\nBaseline vs Steered across concepts evaluated"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "concept_benchmark_running_success_rate.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: overall comparison
try:
    overall_steer = data.get("overall_steering_success_rate", None)
    overall_base = data.get("overall_baseline_success_rate", None)
    if overall_steer is not None and overall_base is not None:
        plt.figure(figsize=(5, 5))
        plt.bar(
            ["baseline", "steered"],
            [overall_base, overall_steer],
            color=["gray", "steelblue"],
        )
        plt.ylabel("Overall success rate")
        plt.ylim(0, 1)
        for i, v in enumerate([overall_base, overall_steer]):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")
        plt.title(
            f"Concept Benchmark: Overall Success Rate\nBaseline vs Steered (α={alpha}, layer={layer})"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "concept_benchmark_overall_success.png"))
        plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Print metrics
try:
    print("Overall baseline success rate:", data.get("overall_baseline_success_rate"))
    print("Overall steered success rate:", data.get("overall_steering_success_rate"))
except Exception as e:
    print(f"Error printing metrics: {e}")
