import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_04-03-48_universal_steering_attempt_1/logs/0-run/experiment_results/experiment_ae1f1ee8c19c4caa8d642478e49828f9_proc_4033862/experiment_data.npy"
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


# Aggregate across runs
def sem(x):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return 0.0
    return np.std(x, ddof=1) / np.sqrt(len(x))


def mean_nan(x):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return np.mean(x)


# Collect keys and structure
all_cb = [
    ed.get("layer_alpha_sweep", {}).get("concept_benchmark", {})
    for ed in all_experiment_data
]
n_runs = len(all_cb)

# union of per_config keys
all_pc_keys = set()
for cb in all_cb:
    all_pc_keys.update(cb.get("per_config", {}).keys())

# Aggregate per_config
agg_pc = {}
for k in all_pc_keys:
    entries = [
        cb.get("per_config", {}).get(k)
        for cb in all_cb
        if cb.get("per_config", {}).get(k) is not None
    ]
    if not entries:
        continue
    agg_pc[k] = {
        "mode": entries[0].get("mode"),
        "layer": entries[0].get("layer"),
        "alpha": entries[0].get("alpha"),
        "success_rate_mean": mean_nan([e["success_rate"] for e in entries]),
        "success_rate_sem": sem([e["success_rate"] for e in entries]),
        "off_target_rate_mean": mean_nan([e["off_target_rate"] for e in entries]),
        "off_target_rate_sem": sem([e["off_target_rate"] for e in entries]),
        "ppl_median_mean": mean_nan([e["ppl_median"] for e in entries]),
        "ppl_median_sem": sem([e["ppl_median"] for e in entries]),
        "composite_score_mean": mean_nan(
            [e.get("composite_score", np.nan) for e in entries]
        ),
        "composite_score_sem": sem([e.get("composite_score", np.nan) for e in entries]),
        "n": len(entries),
        "entries": entries,
    }

baseline_succ = [
    cb.get("baseline", {}).get("success_rate")
    for cb in all_cb
    if cb.get("baseline", {}).get("success_rate") is not None
]
baseline_off = [
    cb.get("baseline", {}).get("off_target_rate")
    for cb in all_cb
    if cb.get("baseline", {}).get("off_target_rate") is not None
]
baseline_ppl = [
    cb.get("baseline", {}).get("ppl_median")
    for cb in all_cb
    if cb.get("baseline", {}).get("ppl_median") is not None
]
prompt_succ = [
    cb.get("prompt_baseline", {}).get("success_rate")
    for cb in all_cb
    if cb.get("prompt_baseline", {}).get("success_rate") is not None
]
prompt_off = [
    cb.get("prompt_baseline", {}).get("off_target_rate")
    for cb in all_cb
    if cb.get("prompt_baseline", {}).get("off_target_rate") is not None
]

baseline_rate_mean = mean_nan(baseline_succ) if baseline_succ else None
baseline_rate_sem = sem(baseline_succ) if baseline_succ else 0
baseline_off_mean = mean_nan(baseline_off) if baseline_off else None
baseline_off_sem = sem(baseline_off) if baseline_off else 0
baseline_ppl_mean = mean_nan(baseline_ppl) if baseline_ppl else None
prompt_rate_mean = mean_nan(prompt_succ) if prompt_succ else None
prompt_rate_sem = sem(prompt_succ) if prompt_succ else 0
prompt_off_mean = mean_nan(prompt_off) if prompt_off else None
prompt_off_sem = sem(prompt_off) if prompt_off else 0

fixed_keys = [k for k in agg_pc if agg_pc[k]["mode"] == "fixed"]
adaptive_keys = [k for k in agg_pc if agg_pc[k]["mode"] == "adaptive"]
layers = sorted(set(agg_pc[k]["layer"] for k in fixed_keys))
fixed_alphas = sorted(set(agg_pc[k]["alpha"] for k in fixed_keys))

# Plot 1: Heatmap of mean success rate and SEM overlay
try:
    grid_succ = np.full((len(layers), len(fixed_alphas)), np.nan)
    grid_succ_sem = np.full((len(layers), len(fixed_alphas)), np.nan)
    grid_off = np.full((len(layers), len(fixed_alphas)), np.nan)
    grid_off_sem = np.full((len(layers), len(fixed_alphas)), np.nan)
    for i, L in enumerate(layers):
        for j, a in enumerate(fixed_alphas):
            k = f"L{L}_a{a}_fixed"
            if k in agg_pc:
                grid_succ[i, j] = agg_pc[k]["success_rate_mean"]
                grid_succ_sem[i, j] = agg_pc[k]["success_rate_sem"]
                grid_off[i, j] = agg_pc[k]["off_target_rate_mean"]
                grid_off_sem[i, j] = agg_pc[k]["off_target_rate_sem"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    im0 = axes[0].imshow(grid_succ, cmap="viridis", vmin=0, vmax=1)
    axes[0].set_xticks(range(len(fixed_alphas)))
    axes[0].set_xticklabels(fixed_alphas)
    axes[0].set_yticks(range(len(layers)))
    axes[0].set_yticklabels(layers)
    axes[0].set_title(f"Left: Mean Success Rate (±SEM, n={n_runs})")
    axes[0].set_xlabel("alpha")
    axes[0].set_ylabel("layer")
    for i in range(len(layers)):
        for j in range(len(fixed_alphas)):
            if not np.isnan(grid_succ[i, j]):
                axes[0].text(
                    j,
                    i,
                    f"{grid_succ[i,j]:.2f}\n±{grid_succ_sem[i,j]:.2f}",
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=7,
                )
    plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].imshow(grid_off, cmap="magma", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(fixed_alphas)))
    axes[1].set_xticklabels(fixed_alphas)
    axes[1].set_yticks(range(len(layers)))
    axes[1].set_yticklabels(layers)
    axes[1].set_title(f"Right: Mean Off-target Rate (±SEM, n={n_runs})")
    axes[1].set_xlabel("alpha")
    axes[1].set_ylabel("layer")
    for i in range(len(layers)):
        for j in range(len(fixed_alphas)):
            if not np.isnan(grid_off[i, j]):
                axes[1].text(
                    j,
                    i,
                    f"{grid_off[i,j]:.2f}\n±{grid_off_sem[i,j]:.2f}",
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=7,
                )
    plt.colorbar(im1, ax=axes[1])
    plt.suptitle("Concept Benchmark: Aggregated Layer × Alpha Sweep (Fixed Mode)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "concept_benchmark_agg_layer_alpha_heatmap.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Methods comparison bar chart with error bars
try:
    best_fixed_key = (
        max(fixed_keys, key=lambda k: agg_pc[k]["success_rate_mean"])
        if fixed_keys
        else None
    )
    best_adapt_key = (
        max(adaptive_keys, key=lambda k: agg_pc[k]["success_rate_mean"])
        if adaptive_keys
        else None
    )
    names, succ_vals, succ_errs, off_vals, off_errs = [], [], [], [], []
    if baseline_rate_mean is not None:
        names.append("Unsteered")
        succ_vals.append(baseline_rate_mean)
        succ_errs.append(baseline_rate_sem)
        off_vals.append(baseline_off_mean)
        off_errs.append(baseline_off_sem)
    if prompt_rate_mean is not None:
        names.append("Prompt")
        succ_vals.append(prompt_rate_mean)
        succ_errs.append(prompt_rate_sem)
        off_vals.append(prompt_off_mean)
        off_errs.append(prompt_off_sem)
    if best_fixed_key:
        names.append(f"Fixed\n({best_fixed_key})")
        succ_vals.append(agg_pc[best_fixed_key]["success_rate_mean"])
        succ_errs.append(agg_pc[best_fixed_key]["success_rate_sem"])
        off_vals.append(agg_pc[best_fixed_key]["off_target_rate_mean"])
        off_errs.append(agg_pc[best_fixed_key]["off_target_rate_sem"])
    if best_adapt_key:
        names.append(f"Adaptive\n({best_adapt_key})")
        succ_vals.append(agg_pc[best_adapt_key]["success_rate_mean"])
        succ_errs.append(agg_pc[best_adapt_key]["success_rate_sem"])
        off_vals.append(agg_pc[best_adapt_key]["off_target_rate_mean"])
        off_errs.append(agg_pc[best_adapt_key]["off_target_rate_sem"])
    x = np.arange(len(names))
    w = 0.35
    plt.figure(figsize=(9, 4.8))
    plt.bar(
        x - w / 2,
        succ_vals,
        w,
        yerr=succ_errs,
        capsize=4,
        label="on-target success (mean ± SEM)",
        color="steelblue",
    )
    plt.bar(
        x + w / 2,
        off_vals,
        w,
        yerr=off_errs,
        capsize=4,
        label="off-target (mean ± SEM)",
        color="salmon",
    )
    plt.xticks(x, names, fontsize=9)
    plt.ylabel("Rate")
    plt.title(
        f"Concept Benchmark: Aggregated Methods Comparison (n={n_runs} runs)\nOn-target vs Off-target rates with SEM"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "concept_benchmark_agg_methods_comparison.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Perplexity vs alpha per layer with SEM
try:
    plt.figure(figsize=(8, 5))
    for L in layers:
        ys, errs = [], []
        for a in fixed_alphas:
            k = f"L{L}_a{a}_fixed"
            if k in agg_pc:
                ys.append(agg_pc[k]["ppl_median_mean"])
                errs.append(agg_pc[k]["ppl_median_sem"])
            else:
                ys.append(np.nan)
                errs.append(0)
        ys = np.array(ys)
        errs = np.array(errs)
        plt.errorbar(
            fixed_alphas, ys, yerr=errs, marker="o", capsize=3, label=f"layer {L}"
        )
    if baseline_ppl_mean is not None:
        plt.axhline(
            baseline_ppl_mean, color="gray", linestyle="--", label="baseline ppl (mean)"
        )
    plt.xlabel("alpha")
    plt.ylabel("median PPL (mean ± SEM)")
    plt.title(
        f"Concept Benchmark: Coherence vs Steering Strength (n={n_runs} runs)\n(Lower is better)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "concept_benchmark_agg_ppl_vs_alpha.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Validation loss and composite per config with error bars
try:
    keys_sorted = sorted(agg_pc.keys())
    val_losses = [1.0 - agg_pc[k]["success_rate_mean"] for k in keys_sorted]
    val_errs = [agg_pc[k]["success_rate_sem"] for k in keys_sorted]
    composites = [agg_pc[k]["composite_score_mean"] for k in keys_sorted]
    comp_errs = [agg_pc[k]["composite_score_sem"] for k in keys_sorted]
    x = np.arange(len(keys_sorted))
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(
        x - 0.2,
        val_losses,
        width=0.4,
        yerr=val_errs,
        capsize=2,
        color="red",
        alpha=0.7,
        label="val loss (1-success) ±SEM",
    )
    ax1.set_ylabel("Validation loss", color="red")
    ax1.set_xticks(x)
    ax1.set_xticklabels(keys_sorted, rotation=60, ha="right", fontsize=7)
    ax2 = ax1.twinx()
    ax2.bar(
        x + 0.2,
        composites,
        width=0.4,
        yerr=comp_errs,
        capsize=2,
        color="green",
        alpha=0.7,
        label="composite ±SEM",
    )
    ax2.set_ylabel("Composite score", color="green")
    plt.title(
        f"Concept Benchmark: Aggregated Val Loss & Composite Score (n={n_runs} runs)\n(Layer × Alpha × Mode Sweep, mean ± SEM)"
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
    fig.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "concept_benchmark_agg_val_loss_composite.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Per-concept success at best config vs baseline (aggregated)
try:
    # pick best key by aggregated composite/success
    best_key = None
    if agg_pc:
        best_key = max(agg_pc.keys(), key=lambda k: agg_pc[k]["success_rate_mean"])
    if best_key:
        # collect per-concept records across runs
        concept_base = {}
        concept_steer = {}
        for cb in all_cb:
            pc = cb.get("per_config", {}).get(best_key)
            if not pc:
                continue
            for r in pc.get("per_concept", []):
                c = r["concept"]
                total = r["total"] if r["total"] > 0 else 1
                concept_base.setdefault(c, []).append(r["baseline_success"] / total)
                concept_steer.setdefault(c, []).append(r["steered_success"] / total)
        concepts = sorted(concept_base.keys())
        base_means = [mean_nan(concept_base[c]) for c in concepts]
        base_errs = [sem(concept_base[c]) for c in concepts]
        steer_means = [mean_nan(concept_steer[c]) for c in concepts]
        steer_errs = [sem(concept_steer[c]) for c in concepts]
        xs = np.arange(len(concepts))
        plt.figure(figsize=(12, 6))
        plt.bar(
            xs - 0.2,
            base_means,
            width=0.4,
            yerr=base_errs,
            capsize=2,
            label="baseline (mean ±SEM)",
        )
        plt.bar(
            xs + 0.2,
            steer_means,
            width=0.4,
            yerr=steer_errs,
            capsize=2,
            label=f"steered ({best_key}) mean ±SEM",
        )
        plt.xticks(xs, concepts, rotation=60, ha="right", fontsize=8)
        plt.ylabel("Success rate")
        plt.title(
            f"Concept Benchmark: Aggregated Per-Concept Success at Best Config ({best_key})\nLeft: Baseline, Right: Steered (n={n_runs} runs)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "concept_benchmark_agg_per_concept_best_config.png"
            )
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print metrics
print(f"Number of runs aggregated: {n_runs}")
print(f"Baseline success rate: mean={baseline_rate_mean}, sem={baseline_rate_sem}")
print(f"Prompt baseline success rate: mean={prompt_rate_mean}, sem={prompt_rate_sem}")
for k in sorted(agg_pc.keys()):
    v = agg_pc[k]
    print(
        f"{k}: success={v['success_rate_mean']:.4f}±{v['success_rate_sem']:.4f}, "
        f"off={v['off_target_rate_mean']:.4f}±{v['off_target_rate_sem']:.4f}, "
        f"ppl_med={v['ppl_median_mean']:.3f}±{v['ppl_median_sem']:.3f}, "
        f"composite={v['composite_score_mean']:.4f}±{v['composite_score_sem']:.4f}, n={v['n']}"
    )
if agg_pc:
    best = max(agg_pc.keys(), key=lambda k: agg_pc[k]["success_rate_mean"])
    print(f"Best config by aggregated success: {best}")
