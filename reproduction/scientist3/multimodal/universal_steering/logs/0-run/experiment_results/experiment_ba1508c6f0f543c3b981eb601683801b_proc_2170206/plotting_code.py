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

cb = experiment_data.get("layer_alpha_sweep", {}).get("concept_benchmark", {})
per_config = cb.get("per_config", {})
baseline = cb.get("baseline", {})
prompt_baseline = cb.get("prompt_baseline", {})
best_key = cb.get("best_config", None)
best_by_succ = cb.get("best_config_by_success", None)

baseline_rate = baseline.get("success_rate", None)
baseline_off_rate = baseline.get("off_target_rate", None)
baseline_ppl_med = baseline.get("ppl_median", None)
prompt_rate = prompt_baseline.get("success_rate", None)
prompt_off_rate = prompt_baseline.get("off_target_rate", None)

# Extract layers/alphas from fixed configs
fixed_keys = [k for k in per_config if per_config[k]["mode"] == "fixed"]
adaptive_keys = [k for k in per_config if per_config[k]["mode"] == "adaptive"]
layers = sorted(set(per_config[k]["layer"] for k in fixed_keys))
fixed_alphas = sorted(set(per_config[k]["alpha"] for k in fixed_keys))

# Plot 1: Heatmap layer x alpha (success & off-target) - Concept Benchmark
try:
    grid_succ = np.full((len(layers), len(fixed_alphas)), np.nan)
    grid_off = np.full((len(layers), len(fixed_alphas)), np.nan)
    for i, L in enumerate(layers):
        for j, a in enumerate(fixed_alphas):
            k = f"L{L}_a{a}_fixed"
            if k in per_config:
                grid_succ[i, j] = per_config[k]["success_rate"]
                grid_off[i, j] = per_config[k]["off_target_rate"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    im0 = axes[0].imshow(grid_succ, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(fixed_alphas)))
    axes[0].set_xticklabels(fixed_alphas)
    axes[0].set_yticks(range(len(layers)))
    axes[0].set_yticklabels(layers)
    axes[0].set_title("Left: Steering Success Rate")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("layer")
    for i in range(len(layers)):
        for j in range(len(fixed_alphas)):
            axes[0].text(
                j,
                i,
                f"{grid_succ[i,j]:.2f}",
                ha="center",
                va="center",
                color="w",
                fontsize=8,
            )
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(grid_off, cmap="magma", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(fixed_alphas)))
    axes[1].set_xticklabels(fixed_alphas)
    axes[1].set_yticks(range(len(layers)))
    axes[1].set_yticklabels(layers)
    axes[1].set_title("Right: Off-target (Distractor) Rate")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("layer")
    for i in range(len(layers)):
        for j in range(len(fixed_alphas)):
            axes[1].text(
                j,
                i,
                f"{grid_off[i,j]:.2f}",
                ha="center",
                va="center",
                color="w",
                fontsize=8,
            )
    plt.colorbar(im1, ax=axes[1])
    plt.suptitle("Concept Benchmark: Layer × Alpha Sweep (Fixed Mode)")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "concept_benchmark_layer_alpha_heatmap.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Methods comparison bar chart
try:
    best_fixed_key = (
        max(fixed_keys, key=lambda k: per_config[k]["success_rate"])
        if fixed_keys
        else None
    )
    best_adapt_key = (
        max(adaptive_keys, key=lambda k: per_config[k]["success_rate"])
        if adaptive_keys
        else None
    )
    names, succ_vals, off_vals = [], [], []
    if baseline_rate is not None:
        names.append("Unsteered")
        succ_vals.append(baseline_rate)
        off_vals.append(baseline_off_rate)
    if prompt_rate is not None:
        names.append("Prompt")
        succ_vals.append(prompt_rate)
        off_vals.append(prompt_off_rate)
    if best_fixed_key:
        names.append(f"Fixed\n({best_fixed_key})")
        succ_vals.append(per_config[best_fixed_key]["success_rate"])
        off_vals.append(per_config[best_fixed_key]["off_target_rate"])
    if best_adapt_key:
        names.append(f"Adaptive\n({best_adapt_key})")
        succ_vals.append(per_config[best_adapt_key]["success_rate"])
        off_vals.append(per_config[best_adapt_key]["off_target_rate"])
    x = np.arange(len(names))
    w = 0.35
    plt.figure(figsize=(9, 4.5))
    plt.bar(x - w / 2, succ_vals, w, label="on-target success", color="steelblue")
    plt.bar(x + w / 2, off_vals, w, label="off-target (distractor)", color="salmon")
    plt.xticks(x, names, fontsize=9)
    plt.ylabel("Rate")
    plt.title(
        "Concept Benchmark: Steering Methods Comparison\nOn-target vs Off-target rates"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "concept_benchmark_methods_comparison.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Perplexity vs alpha per layer
try:
    plt.figure(figsize=(7, 4))
    for L in layers:
        ys = []
        for a in fixed_alphas:
            k = f"L{L}_a{a}_fixed"
            ys.append(per_config[k]["ppl_median"] if k in per_config else np.nan)
        plt.plot(fixed_alphas, ys, marker="o", label=f"layer {L}")
    if baseline_ppl_med is not None:
        plt.axhline(
            baseline_ppl_med, color="gray", linestyle="--", label="baseline ppl"
        )
    plt.xlabel("alpha")
    plt.ylabel("median PPL")
    plt.title(
        "Concept Benchmark: Coherence (Median PPL) vs Steering Strength\n(Lower is better)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "concept_benchmark_ppl_vs_alpha.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Validation loss across all configs
try:
    keys_sorted = sorted(per_config.keys())
    val_losses = [1.0 - per_config[k]["success_rate"] for k in keys_sorted]
    composites = [per_config[k]["composite_score"] for k in keys_sorted]
    x = np.arange(len(keys_sorted))
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.bar(
        x - 0.2,
        val_losses,
        width=0.4,
        color="red",
        alpha=0.7,
        label="val loss (1-success)",
    )
    ax1.set_ylabel("Validation loss", color="red")
    ax1.set_xticks(x)
    ax1.set_xticklabels(keys_sorted, rotation=60, ha="right", fontsize=7)
    ax2 = ax1.twinx()
    ax2.bar(
        x + 0.2,
        composites,
        width=0.4,
        color="green",
        alpha=0.7,
        label="composite score",
    )
    ax2.set_ylabel("Composite score", color="green")
    plt.title(
        "Concept Benchmark: Validation Loss & Composite Score per Configuration\n(Layer × Alpha × Mode Sweep)"
    )
    fig.tight_layout()
    plt.savefig(os.path.join(working_dir, "concept_benchmark_val_loss_composite.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Per-concept success at best config vs baseline
try:
    if best_key and best_key in per_config:
        recs = per_config[best_key]["per_concept"]
        concepts = [r["concept"] for r in recs]
        base_rates = [r["baseline_success"] / r["total"] for r in recs]
        steered_rates = [r["steered_success"] / r["total"] for r in recs]
        xs = np.arange(len(concepts))
        plt.figure(figsize=(12, 6))
        plt.bar(xs - 0.2, base_rates, width=0.4, label="baseline")
        plt.bar(xs + 0.2, steered_rates, width=0.4, label=f"steered ({best_key})")
        plt.xticks(xs, concepts, rotation=60, ha="right", fontsize=8)
        plt.ylabel("Success rate")
        plt.title(
            f"Concept Benchmark: Per-Concept Success at Best Config ({best_key})\nLeft bars: Baseline, Right bars: Steered"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "concept_benchmark_per_concept_best_config.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print metrics
print(f"Baseline success rate: {baseline_rate}")
print(f"Prompt baseline success rate: {prompt_rate}")
for k in sorted(per_config.keys()):
    v = per_config[k]
    print(
        f"{k}: success={v['success_rate']:.4f}, off={v['off_target_rate']:.4f}, "
        f"ppl_med={v['ppl_median']:.3f}, composite={v['composite_score']:.4f}"
    )
print(f"Best config (composite): {best_key}")
print(f"Best config (raw success): {best_by_succ}")
