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

cb = experiment_data.get("alpha_tuning", {}).get("concept_benchmark", {})
per_alpha = cb.get("per_alpha", {})
baseline_rate = cb.get("overall_baseline_success_rate", None)
best_alpha = cb.get("best_alpha", None)
alpha_keys = list(per_alpha.keys())
alphas = [per_alpha[k]["alpha"] for k in alpha_keys]

# Plot 1: Overall success rate vs alpha
try:
    plt.figure(figsize=(6, 4))
    ys = [per_alpha[k]["overall_steering_success_rate"] for k in alpha_keys]
    plt.plot(alphas, ys, marker="o", label="steered")
    if baseline_rate is not None:
        plt.axhline(baseline_rate, color="gray", linestyle="--", label="baseline")
    plt.xlabel("α (steering strength)")
    plt.ylabel("Overall success rate")
    plt.title(
        "Concept Benchmark: Overall Steering Success Rate vs α\n(Llama-3.1-8B-Instruct)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "concept_benchmark_overall_success_vs_alpha.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Per-class success rate across alphas
try:
    if alpha_keys:
        classes = sorted(
            set(r["class"] for r in per_alpha[alpha_keys[0]]["per_concept"])
        )
        xs = np.arange(len(classes))
        width = 0.8 / (len(alphas) + 1)
        plt.figure(figsize=(10, 5))
        base_by_class = {c: [] for c in classes}
        for r in per_alpha[alpha_keys[0]]["per_concept"]:
            base_by_class[r["class"]].append(r["baseline_success"] / r["total"])
        base_means = [np.mean(base_by_class[c]) for c in classes]
        plt.bar(xs - 0.4 + width / 2, base_means, width=width, label="baseline")
        for i, a in enumerate(alphas):
            akey = f"alpha_{a}"
            cls_s = {c: [] for c in classes}
            for r in per_alpha[akey]["per_concept"]:
                cls_s[r["class"]].append(r["steered_success"] / r["total"])
            means = [np.mean(cls_s[c]) for c in classes]
            plt.bar(xs - 0.4 + width * (i + 1.5), means, width=width, label=f"α={a}")
        plt.xticks(xs, classes, rotation=30)
        plt.ylabel("Success rate")
        plt.title(
            "Concept Benchmark: Success Rate by Concept Class\n(Baseline vs Steered α values)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "concept_benchmark_success_by_class.png"))
        plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Per-concept success at best alpha vs baseline
try:
    if best_alpha is not None:
        akey = f"alpha_{best_alpha}"
        recs = per_alpha[akey]["per_concept"]
        concepts = [r["concept"] for r in recs]
        baseline_rates = [r["baseline_success"] / r["total"] for r in recs]
        steered_rates = [r["steered_success"] / r["total"] for r in recs]
        xs = np.arange(len(concepts))
        plt.figure(figsize=(12, 6))
        plt.bar(xs - 0.2, baseline_rates, width=0.4, label="baseline")
        plt.bar(xs + 0.2, steered_rates, width=0.4, label=f"steered (α={best_alpha})")
        plt.xticks(xs, concepts, rotation=60, ha="right", fontsize=8)
        plt.ylabel("Success rate")
        plt.title(
            f"Concept Benchmark: Per-Concept Success at Best α={best_alpha}\nLeft bars: Baseline, Right bars: Steered"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "concept_benchmark_per_concept_best_alpha.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Val loss vs alpha
try:
    plt.figure(figsize=(6, 4))
    val_losses = [per_alpha[k]["val_loss"] for k in alpha_keys]
    plt.plot(
        alphas, val_losses, marker="s", color="red", label="val loss (1 - success)"
    )
    plt.xlabel("α (steering strength)")
    plt.ylabel("Validation loss")
    plt.title(
        "Concept Benchmark: Validation Loss vs α\n(Loss = 1 - steering success rate)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "concept_benchmark_val_loss_vs_alpha.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Running steered success rate curve for each alpha
try:
    plt.figure(figsize=(8, 5))
    for k in alpha_keys:
        rm = per_alpha[k].get("running_metrics", [])
        if rm:
            xs = [r["concept_index"] for r in rm]
            ys = [r["steered_success_rate_running"] for r in rm]
            plt.plot(xs, ys, marker=".", label=f"α={per_alpha[k]['alpha']}")
    if baseline_rate is not None:
        plt.axhline(baseline_rate, color="gray", linestyle="--", label="baseline")
    plt.xlabel("Concept index (evaluation order)")
    plt.ylabel("Running steered success rate")
    plt.title("Concept Benchmark: Running Success Rate During Evaluation\n(per α)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "concept_benchmark_running_success.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print evaluation metrics
print(f"Baseline success rate: {baseline_rate}")
for k in alpha_keys:
    v = per_alpha[k]
    print(
        f"{k}: steered success = {v['overall_steering_success_rate']:.4f}, val_loss = {v['val_loss']:.4f}"
    )
print(f"Best alpha: {best_alpha}")
