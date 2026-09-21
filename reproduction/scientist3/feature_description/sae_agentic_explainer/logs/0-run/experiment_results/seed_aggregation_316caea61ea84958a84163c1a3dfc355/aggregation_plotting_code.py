import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-14_09-16-36_sae_agentic_explainer_attempt_3/logs/0-run/experiment_results/experiment_593e10342e12481789d990ad18bf20c5_proc_3987024/experiment_data.npy",
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

ds_key = "gemma2_2b_gemmascope_res_16k"

# Collect summary data across runs
np_means_across_runs = []
sage_means_across_runs = []  # list of lists (per iter)
sage_finals_across_runs = []
deltas_across_runs = []
layer = "NA"
features_ref = []

for exp in all_experiment_data:
    data = exp.get(ds_key, {})
    summary = data.get("summary", {})
    layer = data.get("layer", layer)
    if not features_ref:
        features_ref = data.get("features", [])
    np_means_across_runs.append(summary.get("neuronpedia_mean_gen_acc", np.nan))
    sage_means_across_runs.append(list(summary.get("sage_mean_gen_acc_per_iter", [])))
    sage_finals_across_runs.append(summary.get("sage_final_gen_acc", np.nan))
    deltas_across_runs.append(summary.get("delta", np.nan))

n_runs = len(all_experiment_data)


def sem(x):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) <= 1:
        return 0.0
    return np.std(x, ddof=1) / np.sqrt(len(x))


def mean_safe(x):
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return np.mean(x)


# Plot 1: Aggregated mean generative accuracy bar with SEM
try:
    max_iters = max((len(s) for s in sage_means_across_runs), default=0)
    sage_iter_matrix = np.full((n_runs, max_iters), np.nan)
    for i, s in enumerate(sage_means_across_runs):
        sage_iter_matrix[i, : len(s)] = s

    np_mean_agg = mean_safe(np_means_across_runs)
    np_sem_agg = sem(np_means_across_runs)
    sage_iter_means = [mean_safe(sage_iter_matrix[:, j]) for j in range(max_iters)]
    sage_iter_sems = [sem(sage_iter_matrix[:, j]) for j in range(max_iters)]

    labels = ["Neuronpedia"] + [f"SAGE it{i}" for i in range(max_iters)]
    means = [np_mean_agg] + sage_iter_means
    errs = [np_sem_agg] + sage_iter_sems
    colors = ["gray"] + ["C0"] * max_iters

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        labels,
        means,
        yerr=errs,
        color=colors,
        capsize=5,
        label=f"Mean ± SEM (n={n_runs})",
    )
    ax.set_ylabel("Generative Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Aggregated Mean Generative Accuracy (Mean ± SEM across {n_runs} runs)\n"
        f"Dataset: Gemma-2-2B GemmaScope-Res-16k (Layer {layer})"
    )
    for i, (m, e) in enumerate(zip(means, errs)):
        if not np.isnan(m):
            ax.text(i, m + e + 0.02, f"{m:.2f}", ha="center", fontsize=8)
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "gemma2_2b_agg_mean_gen_accuracy_bar.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Per-feature accuracy across iterations aggregated across runs
try:
    fig, ax = plt.subplots(figsize=(10, 6))
    # For each feature, gather across runs
    feature_ids = features_ref
    for fid in feature_ids:
        runs_iter_accs = []
        np_accs = []
        for exp in all_experiment_data:
            data = exp.get(ds_key, {})
            pf = data.get("per_feature", {}).get(fid, {})
            iters = pf.get("iterations", [])
            accs = [it["gen_acc"] for it in iters]
            runs_iter_accs.append(accs)
            np_acc = pf.get("neuronpedia", {}).get("gen_acc", None)
            if np_acc is not None:
                np_accs.append(np_acc)
        if not runs_iter_accs:
            continue
        max_it = max(len(a) for a in runs_iter_accs)
        mat = np.full((len(runs_iter_accs), max_it), np.nan)
        for i, a in enumerate(runs_iter_accs):
            mat[i, : len(a)] = a
        means = [mean_safe(mat[:, j]) for j in range(max_it)]
        errs = [sem(mat[:, j]) for j in range(max_it)]
        xs = np.arange(max_it)
        ax.errorbar(
            xs,
            means,
            yerr=errs,
            marker="o",
            capsize=3,
            label=f"feat {fid} SAGE (Mean±SEM)",
        )
        if np_accs:
            np_m = mean_safe(np_accs)
            ax.axhline(
                np_m,
                linestyle="--",
                alpha=0.5,
                label=f"feat {fid} Neuronpedia mean={np_m:.2f}",
            )
    ax.set_xlabel("SAGE Iteration")
    ax.set_ylabel("Generative Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"Per-Feature Generative Accuracy across SAGE Iterations (Mean ± SEM, n={n_runs})\n"
        f"Dataset: Gemma-2-2B GemmaScope-Res-16k (Layer {layer})"
    )
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "gemma2_2b_agg_per_feature_accuracy_curves.png"),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Aggregated validation loss per feature (mean ± SEM)
try:
    # Gather per-feature validation losses across runs (using features_ref order)
    feature_ids = features_ref
    per_feat_losses = {fid: [] for fid in feature_ids}
    for exp in all_experiment_data:
        data = exp.get(ds_key, {})
        val_losses = data.get("losses", {}).get("val", [])
        feats = data.get("features", [])
        for fid, vl in zip(feats, val_losses):
            if fid in per_feat_losses:
                per_feat_losses[fid].append(vl)

    if any(len(v) > 0 for v in per_feat_losses.values()):
        fig, ax = plt.subplots(figsize=(8, 5))
        xs = [f"feat {f}" for f in feature_ids]
        means = [mean_safe(per_feat_losses[f]) for f in feature_ids]
        errs = [sem(per_feat_losses[f]) for f in feature_ids]
        ax.errorbar(
            xs,
            means,
            yerr=errs,
            marker="s",
            color="C3",
            capsize=5,
            label=f"Mean ± SEM (n={n_runs})",
        )
        ax.set_ylabel("Validation Loss (1 - final gen_acc)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(
            f"Aggregated Validation Loss per Feature (final SAGE iter)\n"
            f"Dataset: Gemma-2-2B GemmaScope-Res-16k (Layer {layer})"
        )
        for i, (m, e) in enumerate(zip(means, errs)):
            if not np.isnan(m):
                ax.text(i, m + e + 0.02, f"{m:.2f}", ha="center", fontsize=8)
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemma2_2b_agg_val_loss_per_feature.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Aggregated final SAGE gen acc vs Neuronpedia scatter/bar
try:
    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ["Neuronpedia", "SAGE final"]
    means = [mean_safe(np_means_across_runs), mean_safe(sage_finals_across_runs)]
    errs = [sem(np_means_across_runs), sem(sage_finals_across_runs)]
    ax.bar(
        labels,
        means,
        yerr=errs,
        color=["gray", "C2"],
        capsize=5,
        label=f"Mean ± SEM (n={n_runs})",
    )
    ax.set_ylabel("Generative Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Final SAGE vs Neuronpedia (Aggregated, n={n_runs})\n"
        f"Dataset: Gemma-2-2B GemmaScope-Res-16k (Layer {layer})"
    )
    for i, (m, e) in enumerate(zip(means, errs)):
        if not np.isnan(m):
            ax.text(i, m + e + 0.02, f"{m:.2f}", ha="center", fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "gemma2_2b_agg_final_vs_neuronpedia.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Print aggregated summary metrics
try:
    print(f"Number of runs aggregated: {n_runs}")
    print(
        f"Neuronpedia mean gen acc (mean ± SEM): {mean_safe(np_means_across_runs):.4f} ± {sem(np_means_across_runs):.4f}"
    )
    print(
        f"SAGE final gen acc (mean ± SEM): {mean_safe(sage_finals_across_runs):.4f} ± {sem(sage_finals_across_runs):.4f}"
    )
    print(
        f"Delta (mean ± SEM): {mean_safe(deltas_across_runs):.4f} ± {sem(deltas_across_runs):.4f}"
    )
except Exception as e:
    print(f"Error printing summary: {e}")
