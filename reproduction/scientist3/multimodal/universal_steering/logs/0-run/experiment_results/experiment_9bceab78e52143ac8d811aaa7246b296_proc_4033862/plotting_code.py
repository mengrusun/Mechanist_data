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

cb = experiment_data.get("layer_ablation", {}).get("concept_benchmark", {})
per_layer = cb.get("per_layer", {})
baseline_rate = cb.get("overall_baseline_success_rate", None)
alpha = cb.get("alpha", None)
best_layer = cb.get("best_layer", None)

layer_keys = sorted(per_layer.keys(), key=lambda k: per_layer[k]["layer"])
layers = [per_layer[k]["layer"] for k in layer_keys]

# Plot 1: overall steering success across layers
try:
    plt.figure(figsize=(6, 4))
    ys = [per_layer[k]["overall_steering_success_rate"] for k in layer_keys]
    plt.plot(layers, ys, marker="o", label="steered")
    if baseline_rate is not None:
        plt.axhline(baseline_rate, color="gray", linestyle="--", label="baseline")
    plt.xlabel("Steering layer index")
    plt.ylabel("Overall success rate")
    plt.title(f"Concept Benchmark: Overall Steering Success vs Layer (α={alpha})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "concept_benchmark_overall_success_by_layer.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: per-class success rate across layers
try:
    if layer_keys:
        classes = sorted(
            set(r["class"] for r in per_layer[layer_keys[0]]["per_concept"])
        )
        xs = np.arange(len(classes))
        width = 0.8 / (len(layer_keys) + 1)
        plt.figure(figsize=(10, 5))
        # baseline
        base_by_class = {c: [] for c in classes}
        for r in per_layer[layer_keys[0]]["per_concept"]:
            base_by_class[r["class"]].append(r["baseline_success"] / r["total"])
        base_means = [np.mean(base_by_class[c]) for c in classes]
        plt.bar(xs - 0.4 + width / 2, base_means, width=width, label="baseline")
        for i, k in enumerate(layer_keys):
            L = per_layer[k]["layer"]
            cls_success = {c: [] for c in classes}
            for r in per_layer[k]["per_concept"]:
                cls_success[r["class"]].append(r["steered_success"] / r["total"])
            means = [np.mean(cls_success[c]) for c in classes]
            plt.bar(
                xs - 0.4 + width * (i + 1.5), means, width=width, label=f"layer={L}"
            )
        plt.xticks(xs, classes, rotation=30)
        plt.ylabel("Success rate")
        plt.title(
            f"Concept Benchmark: Per-Class Success Rate Across Layers (α={alpha})"
        )
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "concept_benchmark_per_class_success_by_layer.png"
            )
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: per-concept success at best layer (baseline vs steered)
try:
    if best_layer is not None:
        bkey = f"layer_{best_layer}"
        recs = per_layer[bkey]["per_concept"]
        concepts = [r["concept"] for r in recs]
        base_vals = [r["baseline_success"] / r["total"] for r in recs]
        steer_vals = [r["steered_success"] / r["total"] for r in recs]
        xs = np.arange(len(concepts))
        plt.figure(figsize=(12, 5))
        plt.bar(xs - 0.2, base_vals, width=0.4, label="baseline")
        plt.bar(xs + 0.2, steer_vals, width=0.4, label="steered")
        plt.xticks(xs, concepts, rotation=60, ha="right", fontsize=8)
        plt.ylabel("Success rate")
        plt.title(
            f"Concept Benchmark: Per-Concept Success at Best Layer={best_layer}\nLeft bars: Baseline, Right bars: Steered (α={alpha})"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "concept_benchmark_per_concept_best_layer.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: running success rate curves per layer
try:
    plt.figure(figsize=(8, 5))
    for k in layer_keys:
        rm = per_layer[k].get("running_metrics", [])
        if not rm:
            continue
        idxs = [m["concept_index"] for m in rm]
        vals = [m["steered_success_rate_running"] for m in rm]
        plt.plot(idxs, vals, marker=".", label=f"layer={per_layer[k]['layer']}")
    if baseline_rate is not None:
        plt.axhline(baseline_rate, color="gray", linestyle="--", label="baseline")
    plt.xlabel("Concept index")
    plt.ylabel("Running steered success rate")
    plt.title(f"Concept Benchmark: Running Success Rate Across Concepts (α={alpha})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "concept_benchmark_running_success_by_layer.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Print evaluation metrics
try:
    print("=== Evaluation Metrics ===")
    if baseline_rate is not None:
        print(f"Baseline success rate: {baseline_rate:.4f}")
    for k in layer_keys:
        v = per_layer[k]
        print(
            f"{k}: steered success = {v['overall_steering_success_rate']:.4f}, val_loss = {v.get('val_loss', float('nan')):.4f}"
        )
    print(f"Best layer: {best_layer}")
except Exception as e:
    print(f"Error printing metrics: {e}")
