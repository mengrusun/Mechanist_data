import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_06-08-42_propositional_logic_circuit_attempt_1/logs/0-run/experiment_results/experiment_a90f3e42fd834dbfa0df9dbe00e8c6be_proc_3928150/experiment_data.npy"
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

ds_key = "propositional_logic_circuit"
runs = [ed.get(ds_key, {}) for ed in all_experiment_data]
n_runs = len(runs)
print(f"Loaded {n_runs} run(s)")


def sem(x, axis=0):
    x = np.asarray(x, dtype=float)
    if x.shape[axis] <= 1:
        return np.zeros_like(np.mean(x, axis=axis))
    return np.std(x, axis=axis, ddof=1) / np.sqrt(x.shape[axis])


# Plot 1: Mean head effects heatmap across runs
try:
    heads_list = [
        np.array(r.get("head_effects", []))
        for r in runs
        if len(r.get("head_effects", [])) > 0
    ]
    if heads_list:
        stack = np.stack(heads_list, axis=0)
        mean_heads = stack.mean(axis=0)
        plt.figure(figsize=(10, 6))
        vmax = np.abs(mean_heads).max() if mean_heads.size else 1
        im = plt.imshow(mean_heads, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.colorbar(im, label="Mean Recovery")
        plt.xlabel("Head")
        plt.ylabel("Layer")
        plt.title(
            f"Propositional Logic Dataset: Mean Attention Head Patching Recovery\n(Aggregated over {len(heads_list)} run(s))"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "propositional_logic_head_patching_mean_heatmap.png"
            ),
            dpi=100,
        )
        plt.close()
except Exception as e:
    print(f"Error creating mean head heatmap: {e}")
    plt.close()

# Plot 2: MLP recovery per layer with SEM
try:
    mlp_list = [
        np.array(r.get("mlp_effects", []))
        for r in runs
        if len(r.get("mlp_effects", [])) > 0
    ]
    if mlp_list:
        stack = np.stack(mlp_list, axis=0)
        mean_mlp = stack.mean(axis=0)
        sem_mlp = sem(stack, axis=0)
        plt.figure(figsize=(8, 4))
        x = np.arange(len(mean_mlp))
        plt.bar(
            x,
            mean_mlp,
            yerr=sem_mlp,
            capsize=4,
            color="steelblue",
            label=f"Mean ± SEM (n={len(mlp_list)})",
        )
        plt.xlabel("Layer")
        plt.ylabel("Recovery")
        plt.title(
            "Propositional Logic Dataset: MLP Patching Recovery per Layer\n(Mean ± SEM across runs)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "propositional_logic_mlp_recovery_mean_sem.png"),
            dpi=100,
        )
        plt.close()
except Exception as e:
    print(f"Error creating MLP recovery mean plot: {e}")
    plt.close()

# Plot 3: Faithfulness vs K with SEM
try:
    # Collect common Ks across runs
    faith_dicts = [r.get("faithfulness", {}) for r in runs]
    faith_dicts = [f for f in faith_dicts if f]
    if faith_dicts:
        common_ks = set(faith_dicts[0].keys())
        for f in faith_dicts[1:]:
            common_ks &= set(f.keys())
        Ks = sorted(common_ks, key=lambda k: int(k))
        Ks_int = [int(k) for k in Ks]

        def collect(metric):
            arr = np.array([[fd[k][metric] for k in Ks] for fd in faith_dicts])
            return arr.mean(axis=0), sem(arr, axis=0)

        m_match, s_match = collect("match")
        m_spar, s_spar = collect("sparsity")
        m_faith, s_faith = collect("faithfulness")

        plt.figure(figsize=(7, 4.5))
        plt.errorbar(
            Ks_int,
            m_match,
            yerr=s_match,
            fmt="o-",
            capsize=3,
            label="Match (mean ± SEM)",
        )
        plt.errorbar(
            Ks_int,
            m_spar,
            yerr=s_spar,
            fmt="^-",
            capsize=3,
            label="Sparsity (mean ± SEM)",
        )
        plt.errorbar(
            Ks_int,
            m_faith,
            yerr=s_faith,
            fmt="s-",
            capsize=3,
            label="Faithfulness (mean ± SEM)",
        )
        plt.xlabel("Circuit Size K")
        plt.ylabel("Score")
        plt.title(
            f"Propositional Logic Dataset: Circuit Faithfulness vs Size\nAggregated over {len(faith_dicts)} run(s)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "propositional_logic_faithfulness_mean_sem.png"),
            dpi=100,
        )
        plt.close()
except Exception as e:
    print(f"Error creating faithfulness mean plot: {e}")
    plt.close()

# Plot 4: Validation metrics mean ± SEM
try:
    keys = ["clean_acc", "corr_acc", "mean_clean_logit_diff", "mean_corr_logit_diff"]
    vals_matrix = []
    for r in runs:
        val = r.get("metrics", {}).get("val", [])
        if val:
            vals_matrix.append([val[0].get(k, np.nan) for k in keys])
    if vals_matrix:
        vals_matrix = np.array(vals_matrix, dtype=float)
        means = np.nanmean(vals_matrix, axis=0)
        sems = sem(vals_matrix, axis=0)
        plt.figure(figsize=(7, 4))
        colors = ["green", "green", "steelblue", "steelblue"]
        x = np.arange(len(keys))
        plt.bar(
            x,
            means,
            yerr=sems,
            capsize=4,
            color=colors,
            label=f"Mean ± SEM (n={len(vals_matrix)})",
        )
        plt.xticks(x, keys, rotation=20, ha="right")
        plt.ylabel("Value")
        plt.title(
            "Propositional Logic Dataset: Baseline Validation Metrics\n(Mean ± SEM across runs)"
        )
        plt.axhline(0, color="black", linewidth=0.7)
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "propositional_logic_val_metrics_mean_sem.png"),
            dpi=100,
        )
        plt.close()
except Exception as e:
    print(f"Error creating val metrics mean plot: {e}")
    plt.close()

# Plot 5: Role counts (top-20) mean ± SEM
try:
    role_dicts = [
        r.get("role_counts_top20", {}) for r in runs if r.get("role_counts_top20", {})
    ]
    if role_dicts:
        all_labels = sorted({k for d in role_dicts for k in d.keys()})
        mat = np.array(
            [[d.get(k, 0) for k in all_labels] for d in role_dicts], dtype=float
        )
        means = mat.mean(axis=0)
        sems = sem(mat, axis=0)
        plt.figure(figsize=(7, 4.5))
        x = np.arange(len(all_labels))
        plt.bar(
            x,
            means,
            yerr=sems,
            capsize=4,
            color="#8ecae6",
            label=f"Mean ± SEM (n={len(role_dicts)})",
        )
        plt.xticks(x, all_labels, rotation=20, ha="right")
        plt.ylabel("Count in Top-20")
        plt.title(
            "Propositional Logic Dataset: Functional Role Distribution (Top-20)\n(Mean ± SEM across runs)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "propositional_logic_role_distribution_mean_sem.png"
            ),
            dpi=100,
        )
        plt.close()
except Exception as e:
    print(f"Error creating role distribution mean plot: {e}")
    plt.close()

# Print aggregated summary
print("===== Aggregated Summary =====")
print(f"Number of runs: {n_runs}")
try:
    if vals_matrix.size:
        for i, k in enumerate(keys):
            print(f"{k}: mean={means[i]:.3f} sem={sems[i]:.3f}")
except Exception:
    pass
try:
    if faith_dicts:
        for i, k in enumerate(Ks):
            print(
                f"K={k}: match={m_match[i]:.3f}±{s_match[i]:.3f}, "
                f"sparsity={m_spar[i]:.3f}±{s_spar[i]:.3f}, "
                f"faithfulness={m_faith[i]:.3f}±{s_faith[i]:.3f}"
            )
except Exception:
    pass
