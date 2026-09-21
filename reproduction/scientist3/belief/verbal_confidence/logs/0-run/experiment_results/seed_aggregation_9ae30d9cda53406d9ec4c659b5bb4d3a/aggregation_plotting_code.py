import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_22-41-30_verbal_confidence_attempt_2/logs/0-run/experiment_results/experiment_2dd568da673341ec8fac9119f69bd7c8_proc_1757903/experiment_data.npy"
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

key = "triviaqa_gemma3_27b_pt"
runs = [ed.get(key, {}) for ed in all_experiment_data if key in ed]
n_runs = len(runs)
print(f"Number of runs aggregated: {n_runs}")


def stack_and_stats(list_of_arrays):
    """Return mean and SEM along axis 0 for a list of 1D arrays of same length."""
    arrs = [np.array(a) for a in list_of_arrays if a is not None and len(a) > 0]
    if not arrs:
        return None, None, None
    min_len = min(len(a) for a in arrs)
    arrs = np.stack([a[:min_len] for a in arrs], axis=0)
    mean = arrs.mean(axis=0)
    sem = (
        arrs.std(axis=0, ddof=1) / np.sqrt(arrs.shape[0])
        if arrs.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem, min_len


def scalar_stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    vals = np.array(vals, dtype=float)
    mean = vals.mean()
    sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return mean, sem


# Plot 1: Aggregated per-layer probe accuracy with SEM error bars
try:
    layer_indices_list = [r.get("layer_indices", []) for r in runs]
    layer_indices = None
    for li in layer_indices_list:
        if len(li) > 0:
            layer_indices = np.array(li)
            break

    conf_mean, conf_sem, L = stack_and_stats(
        [r.get("per_layer_confidence_acc", []) for r in runs]
    )
    corr_mean, corr_sem, _ = stack_and_stats(
        [r.get("per_layer_correctness_acc", []) for r in runs]
    )
    ctrl_mean, ctrl_sem, _ = stack_and_stats(
        [r.get("per_layer_control_conf_acc", []) for r in runs]
    )

    maj_conf_mean, maj_conf_sem = scalar_stats(
        [r.get("majority_baseline_conf") for r in runs]
    )
    maj_corr_mean, maj_corr_sem = scalar_stats(
        [r.get("majority_baseline_corr") for r in runs]
    )

    plt.figure(figsize=(10, 6))
    if layer_indices is not None and conf_mean is not None:
        x = layer_indices[:L]
        plt.errorbar(
            x,
            conf_mean,
            yerr=conf_sem,
            fmt="o-",
            capsize=3,
            label=f"Post-answer -> Confidence (mean ± SEM, n={n_runs})",
        )
    if layer_indices is not None and ctrl_mean is not None:
        x = layer_indices[: len(ctrl_mean)]
        plt.errorbar(
            x,
            ctrl_mean,
            yerr=ctrl_sem,
            fmt="s--",
            capsize=3,
            label=f"Pre-answer (control) -> Confidence (mean ± SEM)",
        )
    if layer_indices is not None and corr_mean is not None:
        x = layer_indices[: len(corr_mean)]
        plt.errorbar(
            x,
            corr_mean,
            yerr=corr_sem,
            fmt="^-",
            capsize=3,
            label=f"Post-answer -> Correctness (mean ± SEM)",
        )
    if maj_conf_mean is not None:
        plt.axhline(
            maj_conf_mean,
            color="gray",
            linestyle=":",
            label=f"Majority conf ({maj_conf_mean:.3f}±{maj_conf_sem:.3f})",
        )
    if maj_corr_mean is not None:
        plt.axhline(
            maj_corr_mean,
            color="black",
            linestyle=":",
            label=f"Majority corr ({maj_corr_mean:.3f}±{maj_corr_sem:.3f})",
        )
    plt.xlabel("Layer")
    plt.ylabel("Probe accuracy")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Aggregated Linear Probe Accuracy vs Layer\n"
        f"Mean ± SEM across {n_runs} run(s)"
    )
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, "triviaqa_gemma3_27b_pt_agg_probe_accuracy_vs_layer.png"
        ),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Aggregated distribution of verbal confidences (pooled across runs)
try:
    all_raw_conf = []
    for r in runs:
        rc = r.get("raw_confidences", [])
        if len(rc) > 0:
            all_raw_conf.append(np.array(rc))
    plt.figure(figsize=(8, 5))
    if all_raw_conf:
        pooled = np.concatenate(all_raw_conf)
        plt.hist(
            pooled,
            bins=20,
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
            label=f"Pooled (n_total={len(pooled)})",
        )
        # Per-run means with SEM
        per_run_means = np.array([a.mean() for a in all_raw_conf])
        m_mean = per_run_means.mean()
        m_sem = (
            per_run_means.std(ddof=1) / np.sqrt(len(per_run_means))
            if len(per_run_means) > 1
            else 0.0
        )
        plt.axvline(
            m_mean,
            color="red",
            linestyle="--",
            label=f"mean of run means={m_mean:.2f} ± {m_sem:.2f} SEM",
        )
        plt.legend()
    plt.xlabel("Continuous verbal confidence")
    plt.ylabel("Count")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Aggregated Distribution of Verbal Confidences\n"
        f"Pooled across {n_runs} run(s)"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, "triviaqa_gemma3_27b_pt_agg_confidence_distribution.png"
        ),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Aggregated confidence by correctness (pooled)
try:
    c0_list, c1_list = [], []
    for r in runs:
        rc = np.array(r.get("raw_confidences", []))
        cr = np.array(r.get("raw_correctness", []))
        if len(rc) and len(cr) and len(rc) == len(cr):
            c0_list.append(rc[cr == 0])
            c1_list.append(rc[cr == 1])
    plt.figure(figsize=(8, 5))
    if c0_list and c1_list:
        c0 = np.concatenate(c0_list) if c0_list else np.array([])
        c1 = np.concatenate(c1_list) if c1_list else np.array([])
        plt.boxplot([c0, c1], labels=["Incorrect", "Correct"])
        # per-run means with SEM
        m0 = [a.mean() for a in c0_list if len(a) > 0]
        m1 = [a.mean() for a in c1_list if len(a) > 0]
        if m0:
            mm0, ss0 = np.mean(m0), (
                np.std(m0, ddof=1) / np.sqrt(len(m0)) if len(m0) > 1 else 0.0
            )
            plt.errorbar(
                [1],
                [mm0],
                yerr=[ss0],
                fmt="D",
                color="red",
                capsize=5,
                label=f"Incorrect mean±SEM: {mm0:.2f}±{ss0:.2f}",
            )
        if m1:
            mm1, ss1 = np.mean(m1), (
                np.std(m1, ddof=1) / np.sqrt(len(m1)) if len(m1) > 1 else 0.0
            )
            plt.errorbar(
                [2],
                [mm1],
                yerr=[ss1],
                fmt="D",
                color="green",
                capsize=5,
                label=f"Correct mean±SEM: {mm1:.2f}±{ss1:.2f}",
            )
        plt.legend()
    plt.ylabel("Verbal confidence")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Aggregated Verbal Confidence by Correctness\n"
        f"Pooled across {n_runs} run(s); Left: Incorrect, Right: Correct"
    )
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, "triviaqa_gemma3_27b_pt_agg_confidence_vs_correctness.png"
        ),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Aggregated calibration curve with SEM error bars
try:
    all_conf = []
    all_corr = []
    for r in runs:
        rc = np.array(r.get("raw_confidences", []))
        cr = np.array(r.get("raw_correctness", []))
        if len(rc) and len(cr) and len(rc) == len(cr):
            all_conf.append(rc)
            all_corr.append(cr)

    plt.figure(figsize=(8, 5))
    if all_conf:
        pooled_conf = np.concatenate(all_conf)
        bins = np.linspace(pooled_conf.min(), pooled_conf.max() + 1e-6, 6)
        centers = (bins[:-1] + bins[1:]) / 2

        # For each run, compute accuracy per bin
        per_run_bin_acc = []
        for rc, cr in zip(all_conf, all_corr):
            bin_ids = np.clip(np.digitize(rc, bins) - 1, 0, len(bins) - 2)
            accs = []
            for b in range(len(bins) - 1):
                mask = bin_ids == b
                accs.append(cr[mask].mean() if mask.sum() > 0 else np.nan)
            per_run_bin_acc.append(accs)
        per_run_bin_acc = np.array(per_run_bin_acc, dtype=float)
        mean_acc = np.nanmean(per_run_bin_acc, axis=0)
        if per_run_bin_acc.shape[0] > 1:
            sem_acc = np.nanstd(per_run_bin_acc, axis=0, ddof=1) / np.sqrt(
                np.sum(~np.isnan(per_run_bin_acc), axis=0)
            )
        else:
            sem_acc = np.zeros_like(mean_acc)

        # Total counts per bin (pooled)
        pooled_corr = np.concatenate(all_corr)
        bin_ids_pool = np.clip(np.digitize(pooled_conf, bins) - 1, 0, len(bins) - 2)
        counts = [(bin_ids_pool == b).sum() for b in range(len(bins) - 1)]

        plt.errorbar(
            centers,
            mean_acc,
            yerr=sem_acc,
            fmt="o-",
            color="purple",
            capsize=3,
            label=f"Mean accuracy ± SEM (n_runs={n_runs})",
        )
        for x, y, c in zip(centers, mean_acc, counts):
            if not np.isnan(y):
                plt.annotate(
                    f"n={c}",
                    (x, y),
                    textcoords="offset points",
                    xytext=(0, 8),
                    ha="center",
                    fontsize=8,
                )

        overall_acc_per_run = [cr.mean() for cr in all_corr]
        oa_mean = np.mean(overall_acc_per_run)
        oa_sem = (
            np.std(overall_acc_per_run, ddof=1) / np.sqrt(len(overall_acc_per_run))
            if len(overall_acc_per_run) > 1
            else 0.0
        )
        plt.axhline(
            oa_mean,
            color="gray",
            linestyle=":",
            label=f"Overall acc={oa_mean:.3f}±{oa_sem:.3f} SEM",
        )
    plt.xlabel("Verbal confidence (bin center)")
    plt.ylabel("Empirical accuracy")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Aggregated Calibration Curve\n"
        f"Mean ± SEM across {n_runs} run(s)"
    )
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_gemma3_27b_pt_agg_calibration_curve.png"),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Aggregated summary bar chart with SEM error bars
try:
    best_conf_vals, best_corr_vals = [], []
    for r in runs:
        vm = r.get("metrics", {}).get("val", [])
        if vm:
            best_conf_vals.append(vm[0].get("cache_probe_accuracy"))
            best_corr_vals.append(vm[0].get("corr_probe_accuracy"))
    maj_conf_vals = [r.get("majority_baseline_conf") for r in runs]
    maj_corr_vals = [r.get("majority_baseline_corr") for r in runs]

    bc_m, bc_s = scalar_stats(best_conf_vals)
    mc_m, mc_s = scalar_stats(maj_conf_vals)
    br_m, br_s = scalar_stats(best_corr_vals)
    mr_m, mr_s = scalar_stats(maj_corr_vals)

    labels = [
        "Conf Probe\n(best layer)",
        "Conf Majority",
        "Corr Probe\n(best layer)",
        "Corr Majority",
    ]
    means = [bc_m or 0, mc_m or 0, br_m or 0, mr_m or 0]
    sems = [bc_s or 0, mc_s or 0, br_s or 0, mr_s or 0]
    colors = ["steelblue", "lightsteelblue", "seagreen", "lightgreen"]

    plt.figure(figsize=(8, 5))
    plt.bar(
        labels,
        means,
        yerr=sems,
        color=colors,
        edgecolor="black",
        capsize=5,
        label=f"Mean ± SEM (n_runs={n_runs})",
    )
    for i, (m, s) in enumerate(zip(means, sems)):
        plt.text(i, m + s + 0.02, f"{m:.3f}±{s:.3f}", ha="center", fontsize=8)
    plt.ylim(0, 1.15)
    plt.ylabel("Accuracy")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Aggregated Best Probe Accuracy vs Majority Baseline\n"
        f"Mean ± SEM across {n_runs} run(s)"
    )
    plt.legend()
    plt.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, "triviaqa_gemma3_27b_pt_agg_probe_vs_baseline_summary.png"
        ),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print aggregated metrics
try:
    print("=== Aggregated Key Metrics (mean ± SEM) ===")
    print(f"Number of runs: {n_runs}")
    mc_m, mc_s = scalar_stats([r.get("majority_baseline_conf") for r in runs])
    mr_m, mr_s = scalar_stats([r.get("majority_baseline_corr") for r in runs])
    print(f"Majority baseline (confidence): {mc_m} ± {mc_s}")
    print(f"Majority baseline (correctness): {mr_m} ± {mr_s}")
    best_layers_conf = [r.get("best_layer_conf") for r in runs]
    best_layers_corr = [r.get("best_layer_corr") for r in runs]
    print(f"Best layer (confidence) per run: {best_layers_conf}")
    print(f"Best layer (correctness) per run: {best_layers_corr}")
    cp_m, cp_s = scalar_stats([r.get("cache_probe_accuracy") for r in runs])
    print(f"Cache probe accuracy: {cp_m} ± {cp_s}")
except Exception as e:
    print(f"Error printing metrics: {e}")
