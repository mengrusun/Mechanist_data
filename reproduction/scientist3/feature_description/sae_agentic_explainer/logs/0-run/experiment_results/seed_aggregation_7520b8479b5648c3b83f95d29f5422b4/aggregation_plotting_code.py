import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-14_09-16-36_sae_agentic_explainer_attempt_3/logs/0-run/experiment_results/experiment_355ab6e8b3d94210896400b04c1e59b0_proc_4068890/experiment_data.npy"
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

# Collect aggregated data
runs_data = [ed.get(ds_key, {}) for ed in all_experiment_data if ds_key in ed]
n_runs = len(runs_data)
print(f"Number of runs: {n_runs}")

layer = runs_data[0].get("layer", "NA") if runs_data else "NA"


def sem(arr, axis=0):
    arr = np.array(arr, dtype=float)
    if arr.shape[axis] <= 1:
        return np.zeros(arr.shape[1:]) if arr.ndim > 1 else 0.0
    return np.std(arr, axis=axis, ddof=1) / np.sqrt(arr.shape[axis])


# Plot 1: Aggregated Mean Generative Accuracy - Neuronpedia vs SAGE iterations
try:
    np_means = []
    sage_means_list = []
    for r in runs_data:
        s = r.get("summary", {})
        np_means.append(s.get("neuronpedia_mean_gen_acc", np.nan))
        sage_means_list.append(list(s.get("sage_mean_gen_acc_per_iter", [])))
    max_iters = max((len(x) for x in sage_means_list), default=0)
    # Pad with nan
    sage_arr = np.full((n_runs, max_iters), np.nan)
    for i, sm in enumerate(sage_means_list):
        sage_arr[i, : len(sm)] = sm

    np_mean_val = np.nanmean(np_means)
    np_sem_val = np.nanstd(np_means, ddof=1) / np.sqrt(n_runs) if n_runs > 1 else 0.0
    sage_mean_vals = np.nanmean(sage_arr, axis=0)
    sage_sem_vals = (
        np.nanstd(sage_arr, axis=0, ddof=1) / np.sqrt(n_runs)
        if n_runs > 1
        else np.zeros(max_iters)
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Neuronpedia"] + [f"SAGE it{i}" for i in range(max_iters)]
    vals = [np_mean_val] + list(sage_mean_vals)
    errs = [np_sem_val] + list(sage_sem_vals)
    colors = ["gray"] + ["C0"] * max_iters
    ax.bar(
        labels,
        vals,
        yerr=errs,
        color=colors,
        capsize=5,
        label=f"Mean ± SEM (n={n_runs})",
    )
    ax.set_ylabel("Generative Accuracy")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        f"Aggregated Mean Generative Accuracy: SAGE vs Neuronpedia\nDataset: Gemma-2-2B GemmaScope-Res-16k (Layer {layer}), n={n_runs} runs"
    )
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(i, v + e + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "gemma2_2b_agg_mean_gen_accuracy_bar.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Aggregated per-feature accuracy across SAGE iterations (mean ± SEM across runs)
try:
    # Get union of features
    all_features = []
    for r in runs_data:
        for f in r.get("features", []):
            if f not in all_features:
                all_features.append(f)

    fig, ax = plt.subplots(figsize=(9, 5))
    for fid in all_features:
        # collect per-run iteration accuracies
        iter_accs_runs = []
        np_accs_runs = []
        for r in runs_data:
            pf = r.get("per_feature", {}).get(fid, {})
            iters = pf.get("iterations", [])
            accs = [it["gen_acc"] for it in iters]
            iter_accs_runs.append(accs)
            np_acc = pf.get("neuronpedia", {}).get("gen_acc", None)
            if np_acc is not None:
                np_accs_runs.append(np_acc)
        max_it = max((len(a) for a in iter_accs_runs), default=0)
        if max_it == 0:
            continue
        arr = np.full((len(iter_accs_runs), max_it), np.nan)
        for i, a in enumerate(iter_accs_runs):
            arr[i, : len(a)] = a
        m = np.nanmean(arr, axis=0)
        s = (
            np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
            if n_runs > 1
            else np.zeros(max_it)
        )
        xs = np.arange(max_it)
        (line,) = ax.plot(xs, m, marker="o", label=f"feat {fid} SAGE (mean±SEM)")
        ax.fill_between(xs, m - s, m + s, alpha=0.2, color=line.get_color())
        if np_accs_runs:
            np_m = np.mean(np_accs_runs)
            ax.axhline(
                np_m,
                linestyle="--",
                alpha=0.5,
                color=line.get_color(),
                label=f"feat {fid} Neuronpedia mean={np_m:.2f}",
            )
    ax.set_xlabel("SAGE Iteration")
    ax.set_ylabel("Generative Accuracy")
    ax.set_ylim(-0.05, 1.1)
    ax.set_title(
        f"Aggregated Per-Feature Gen Accuracy across SAGE Iterations\nDataset: Gemma-2-2B GemmaScope-Res-16k (Layer {layer}), n={n_runs} runs"
    )
    ax.legend(fontsize=7, loc="best")
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
    all_features = []
    for r in runs_data:
        for f in r.get("features", []):
            if f not in all_features:
                all_features.append(f)

    means = []
    sems = []
    feat_labels = []
    for i_f, fid in enumerate(all_features):
        vals = []
        for r in runs_data:
            vl = r.get("losses", {}).get("val", [])
            feats = r.get("features", [])
            if fid in feats:
                idx = feats.index(fid)
                if idx < len(vl):
                    vals.append(vl[idx])
        if vals:
            means.append(np.mean(vals))
            sems.append(
                np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            )
            feat_labels.append(f"feat {fid}")

    if means:
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = np.arange(len(means))
        ax.errorbar(
            xs,
            means,
            yerr=sems,
            marker="s",
            color="C3",
            capsize=5,
            label=f"Mean ± SEM (n={n_runs})",
        )
        ax.set_xticks(xs)
        ax.set_xticklabels(feat_labels, rotation=30)
        ax.set_ylabel("Validation Loss (1 - final gen_acc)")
        ax.set_title(
            f"Aggregated Validation Loss per Feature (final SAGE iter)\nDataset: Gemma-2-2B GemmaScope-Res-16k (Layer {layer}), n={n_runs} runs"
        )
        ax.set_ylim(-0.05, 1.1)
        ax.legend()
        for i, (v, e) in enumerate(zip(means, sems)):
            ax.text(i, v + e + 0.02, f"{v:.2f}", ha="center", fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemma2_2b_agg_val_loss_per_feature.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Print summary metrics
try:
    sage_finals = [
        r.get("summary", {}).get("sage_final_gen_acc", np.nan) for r in runs_data
    ]
    np_means_all = [
        r.get("summary", {}).get("neuronpedia_mean_gen_acc", np.nan) for r in runs_data
    ]
    deltas = [r.get("summary", {}).get("delta", np.nan) for r in runs_data]

    def mean_sem(vals):
        vals = [
            v
            for v in vals
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        ]
        if not vals:
            return float("nan"), float("nan")
        m = np.mean(vals)
        s = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        return m, s

    m, s = mean_sem(sage_finals)
    print(f"SAGE final gen_acc: mean={m:.4f}, SEM={s:.4f}")
    m, s = mean_sem(np_means_all)
    print(f"Neuronpedia mean gen_acc: mean={m:.4f}, SEM={s:.4f}")
    m, s = mean_sem(deltas)
    print(f"Delta (SAGE - Neuronpedia): mean={m:.4f}, SEM={s:.4f}")
except Exception as e:
    print(f"Error printing summary: {e}")
