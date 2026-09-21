import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_04-03-48_universal_steering_attempt_1/logs/0-run/experiment_results/experiment_ba1508c6f0f543c3b981eb601683801b_proc_2170206/experiment_data.npy"
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


def sem(arr, axis=0):
    arr = np.asarray(arr, dtype=float)
    n = np.sum(~np.isnan(arr), axis=axis)
    n = np.where(n > 1, n, np.nan)
    return np.nanstd(arr, axis=axis, ddof=1) / np.sqrt(n)


# Aggregate across runs
cbs = [
    ed.get("layer_alpha_sweep", {}).get("concept_benchmark", {})
    for ed in all_experiment_data
]
cbs = [cb for cb in cbs if cb]
n_runs = len(cbs)
print(f"Number of runs aggregated: {n_runs}")

if n_runs > 0:
    # Collect layers and alphas from all runs
    all_layers, all_alphas = set(), set()
    for cb in cbs:
        for k, v in cb.get("per_config", {}).items():
            if v.get("mode") == "fixed":
                all_layers.add(v["layer"])
                all_alphas.add(v["alpha"])
    layers = sorted(all_layers)
    fixed_alphas = sorted(all_alphas)

    # Baseline metrics across runs
    baseline_rates = [cb.get("baseline", {}).get("success_rate", np.nan) for cb in cbs]
    baseline_off = [cb.get("baseline", {}).get("off_target_rate", np.nan) for cb in cbs]
    baseline_ppl = [cb.get("baseline", {}).get("ppl_median", np.nan) for cb in cbs]
    prompt_rates = [
        cb.get("prompt_baseline", {}).get("success_rate", np.nan) for cb in cbs
    ]
    prompt_off = [
        cb.get("prompt_baseline", {}).get("off_target_rate", np.nan) for cb in cbs
    ]

    # Plot 1: Aggregated heatmaps (mean success and off-target)
    try:
        grid_succ = np.full((n_runs, len(layers), len(fixed_alphas)), np.nan)
        grid_off = np.full((n_runs, len(layers), len(fixed_alphas)), np.nan)
        for r, cb in enumerate(cbs):
            pc = cb.get("per_config", {})
            for i, L in enumerate(layers):
                for j, a in enumerate(fixed_alphas):
                    k = f"L{L}_a{a}_fixed"
                    if k in pc:
                        grid_succ[r, i, j] = pc[k]["success_rate"]
                        grid_off[r, i, j] = pc[k]["off_target_rate"]
        mean_succ = np.nanmean(grid_succ, axis=0)
        sem_succ = sem(grid_succ, axis=0)
        mean_off = np.nanmean(grid_off, axis=0)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        im0 = axes[0].imshow(mean_succ, cmap="viridis", vmin=0, vmax=1)
        axes[0].set_xticks(range(len(fixed_alphas)))
        axes[0].set_xticklabels(fixed_alphas)
        axes[0].set_yticks(range(len(layers)))
        axes[0].set_yticklabels(layers)
        axes[0].set_title(f"Left: Mean Success Rate (n={n_runs})")
        axes[0].set_xlabel("alpha")
        axes[0].set_ylabel("layer")
        for i in range(len(layers)):
            for j in range(len(fixed_alphas)):
                s = sem_succ[i, j] if not np.isnan(sem_succ[i, j]) else 0.0
                axes[0].text(
                    j,
                    i,
                    f"{mean_succ[i,j]:.2f}\n±{s:.2f}",
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=7,
                )
        plt.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(mean_off, cmap="magma", vmin=0, vmax=1)
        axes[1].set_xticks(range(len(fixed_alphas)))
        axes[1].set_xticklabels(fixed_alphas)
        axes[1].set_yticks(range(len(layers)))
        axes[1].set_yticklabels(layers)
        axes[1].set_title(f"Right: Mean Off-target Rate (n={n_runs})")
        axes[1].set_xlabel("alpha")
        axes[1].set_ylabel("layer")
        for i in range(len(layers)):
            for j in range(len(fixed_alphas)):
                axes[1].text(
                    j,
                    i,
                    f"{mean_off[i,j]:.2f}",
                    ha="center",
                    va="center",
                    color="w",
                    fontsize=8,
                )
        plt.colorbar(im1, ax=axes[1])
        plt.suptitle("Concept Benchmark: Aggregated Layer × Alpha Sweep (Fixed Mode)")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "concept_benchmark_agg_heatmap.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot1: {e}")
        plt.close()

    # Plot 2: Methods comparison bar chart with error bars
    try:
        # Collect per-run best fixed and adaptive
        fixed_succ, fixed_off = [], []
        adapt_succ, adapt_off = [], []
        for cb in cbs:
            pc = cb.get("per_config", {})
            fk = [k for k in pc if pc[k].get("mode") == "fixed"]
            ak = [k for k in pc if pc[k].get("mode") == "adaptive"]
            if fk:
                bk = max(fk, key=lambda k: pc[k]["success_rate"])
                fixed_succ.append(pc[bk]["success_rate"])
                fixed_off.append(pc[bk]["off_target_rate"])
            if ak:
                bk = max(ak, key=lambda k: pc[k]["success_rate"])
                adapt_succ.append(pc[bk]["success_rate"])
                adapt_off.append(pc[bk]["off_target_rate"])

        names, succ_means, succ_sems, off_means, off_sems = [], [], [], [], []

        def add_method(name, s_list, o_list):
            if len(s_list) > 0:
                names.append(name)
                succ_means.append(np.nanmean(s_list))
                succ_sems.append(
                    sem(np.array(s_list).reshape(-1, 1))[0] if len(s_list) > 1 else 0.0
                )
                off_means.append(np.nanmean(o_list))
                off_sems.append(
                    sem(np.array(o_list).reshape(-1, 1))[0] if len(o_list) > 1 else 0.0
                )

        add_method("Unsteered", baseline_rates, baseline_off)
        add_method("Prompt", prompt_rates, prompt_off)
        add_method("Best Fixed", fixed_succ, fixed_off)
        add_method("Best Adaptive", adapt_succ, adapt_off)

        x = np.arange(len(names))
        w = 0.35
        plt.figure(figsize=(9, 4.5))
        plt.bar(
            x - w / 2,
            succ_means,
            w,
            yerr=succ_sems,
            capsize=4,
            label="on-target success (mean ± SEM)",
            color="steelblue",
        )
        plt.bar(
            x + w / 2,
            off_means,
            w,
            yerr=off_sems,
            capsize=4,
            label="off-target (mean ± SEM)",
            color="salmon",
        )
        plt.xticks(x, names, fontsize=9)
        plt.ylabel("Rate")
        plt.title(
            f"Concept Benchmark: Aggregated Methods Comparison (n={n_runs})\nOn-target vs Off-target rates"
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

    # Plot 3: Perplexity vs alpha per layer with SE shading
    try:
        plt.figure(figsize=(8, 4.5))
        for L in layers:
            all_ys = np.full((n_runs, len(fixed_alphas)), np.nan)
            for r, cb in enumerate(cbs):
                pc = cb.get("per_config", {})
                for j, a in enumerate(fixed_alphas):
                    k = f"L{L}_a{a}_fixed"
                    if k in pc:
                        all_ys[r, j] = pc[k]["ppl_median"]
            mean_ys = np.nanmean(all_ys, axis=0)
            sem_ys = sem(all_ys, axis=0)
            (line,) = plt.plot(
                fixed_alphas, mean_ys, marker="o", label=f"layer {L} (mean)"
            )
            plt.fill_between(
                fixed_alphas,
                mean_ys - sem_ys,
                mean_ys + sem_ys,
                alpha=0.2,
                color=line.get_color(),
                label=f"layer {L} ±SEM",
            )
        base_ppl_mean = np.nanmean(baseline_ppl)
        if not np.isnan(base_ppl_mean):
            plt.axhline(
                base_ppl_mean,
                color="gray",
                linestyle="--",
                label=f"baseline ppl (mean)",
            )
        plt.xlabel("alpha")
        plt.ylabel("median PPL")
        plt.title(
            f"Concept Benchmark: Coherence (Median PPL) vs Steering Strength (n={n_runs})\n(Lower is better; shaded = ±SEM)"
        )
        plt.legend(fontsize=7, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "concept_benchmark_agg_ppl_vs_alpha.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot3: {e}")
        plt.close()

    # Plot 4: Validation loss & composite score per config with error bars
    try:
        # Union of all config keys
        all_keys = set()
        for cb in cbs:
            all_keys.update(cb.get("per_config", {}).keys())
        keys_sorted = sorted(all_keys)

        val_arr = np.full((n_runs, len(keys_sorted)), np.nan)
        comp_arr = np.full((n_runs, len(keys_sorted)), np.nan)
        for r, cb in enumerate(cbs):
            pc = cb.get("per_config", {})
            for j, k in enumerate(keys_sorted):
                if k in pc:
                    val_arr[r, j] = 1.0 - pc[k]["success_rate"]
                    comp_arr[r, j] = pc[k]["composite_score"]
        val_mean, val_sem = np.nanmean(val_arr, axis=0), sem(val_arr, axis=0)
        comp_mean, comp_sem = np.nanmean(comp_arr, axis=0), sem(comp_arr, axis=0)

        x = np.arange(len(keys_sorted))
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.bar(
            x - 0.2,
            val_mean,
            width=0.4,
            yerr=val_sem,
            capsize=3,
            color="red",
            alpha=0.7,
            label="val loss (mean ± SEM)",
        )
        ax1.set_ylabel("Validation loss", color="red")
        ax1.set_xticks(x)
        ax1.set_xticklabels(keys_sorted, rotation=60, ha="right", fontsize=7)
        ax2 = ax1.twinx()
        ax2.bar(
            x + 0.2,
            comp_mean,
            width=0.4,
            yerr=comp_sem,
            capsize=3,
            color="green",
            alpha=0.7,
            label="composite (mean ± SEM)",
        )
        ax2.set_ylabel("Composite score", color="green")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)
        plt.title(
            f"Concept Benchmark: Aggregated Val Loss & Composite per Configuration (n={n_runs})"
        )
        fig.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "concept_benchmark_agg_val_loss_composite.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot4: {e}")
        plt.close()

    # Plot 5: Per-concept success at best config aggregated
    try:
        # Determine best config per run then aggregate per-concept
        concept_base = {}  # concept -> list of baseline rates
        concept_steered = {}
        for cb in cbs:
            pc = cb.get("per_config", {})
            best_key = cb.get("best_config", None)
            if best_key and best_key in pc:
                for r in pc[best_key]["per_concept"]:
                    c = r["concept"]
                    concept_base.setdefault(c, []).append(
                        r["baseline_success"] / r["total"]
                    )
                    concept_steered.setdefault(c, []).append(
                        r["steered_success"] / r["total"]
                    )
        concepts = sorted(concept_base.keys())
        if concepts:
            base_mean = [np.nanmean(concept_base[c]) for c in concepts]
            base_sem = [
                (
                    sem(np.array(concept_base[c]).reshape(-1, 1))[0]
                    if len(concept_base[c]) > 1
                    else 0.0
                )
                for c in concepts
            ]
            steer_mean = [np.nanmean(concept_steered[c]) for c in concepts]
            steer_sem = [
                (
                    sem(np.array(concept_steered[c]).reshape(-1, 1))[0]
                    if len(concept_steered[c]) > 1
                    else 0.0
                )
                for c in concepts
            ]
            xs = np.arange(len(concepts))
            plt.figure(figsize=(12, 6))
            plt.bar(
                xs - 0.2,
                base_mean,
                width=0.4,
                yerr=base_sem,
                capsize=3,
                label="baseline (mean ± SEM)",
            )
            plt.bar(
                xs + 0.2,
                steer_mean,
                width=0.4,
                yerr=steer_sem,
                capsize=3,
                label="steered best (mean ± SEM)",
            )
            plt.xticks(xs, concepts, rotation=60, ha="right", fontsize=8)
            plt.ylabel("Success rate")
            plt.title(
                f"Concept Benchmark: Per-Concept Success at Best Config, Aggregated (n={n_runs})\nLeft bars: Baseline, Right bars: Steered"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "concept_benchmark_agg_per_concept.png")
            )
            plt.close()
    except Exception as e:
        print(f"Error creating plot5: {e}")
        plt.close()

    # Print aggregated metrics
    print(f"\n=== Aggregated Metrics (n={n_runs}) ===")
    print(
        f"Baseline success rate: mean={np.nanmean(baseline_rates):.4f}, sem={sem(np.array(baseline_rates).reshape(-1,1))[0] if n_runs>1 else 0.0:.4f}"
    )
    print(f"Prompt baseline success rate: mean={np.nanmean(prompt_rates):.4f}")
    print(f"Baseline off-target rate: mean={np.nanmean(baseline_off):.4f}")
    print(f"Baseline median PPL: mean={np.nanmean(baseline_ppl):.4f}")
    # Per-config aggregate stats
    all_keys = set()
    for cb in cbs:
        all_keys.update(cb.get("per_config", {}).keys())
    for k in sorted(all_keys):
        succs, offs, ppls, comps = [], [], [], []
        for cb in cbs:
            pc = cb.get("per_config", {})
            if k in pc:
                succs.append(pc[k]["success_rate"])
                offs.append(pc[k]["off_target_rate"])
                ppls.append(pc[k]["ppl_median"])
                comps.append(pc[k]["composite_score"])
        if succs:
            print(
                f"{k}: succ={np.nanmean(succs):.4f}±{(sem(np.array(succs).reshape(-1,1))[0] if len(succs)>1 else 0.0):.4f}, "
                f"off={np.nanmean(offs):.4f}, ppl={np.nanmean(ppls):.3f}, comp={np.nanmean(comps):.4f}"
            )
else:
    print("No experiment data available for aggregation.")
