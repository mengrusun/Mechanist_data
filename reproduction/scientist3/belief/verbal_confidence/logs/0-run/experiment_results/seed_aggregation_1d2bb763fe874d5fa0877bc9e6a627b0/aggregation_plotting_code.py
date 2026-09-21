import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_22-41-30_verbal_confidence_attempt_2/logs/0-run/experiment_results/experiment_b706422aaa3e416780b1cf58c1352582_proc_1899480/experiment_data.npy"
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

key = "triviaqa_gemma3_27b_pt"
runs = [ed.get(key, {}) for ed in all_experiment_data if key in ed]
n_runs = len(runs)
print(f"Number of runs aggregated: {n_runs}")


def stack_and_stats(list_of_arrays):
    """Return mean, sem across runs, given a list of 1D arrays of the same length."""
    arrs = [
        np.asarray(a, dtype=float)
        for a in list_of_arrays
        if a is not None and len(a) > 0
    ]
    if not arrs:
        return None, None
    min_len = min(len(a) for a in arrs)
    arrs = np.stack([a[:min_len] for a in arrs], axis=0)
    mean = arrs.mean(axis=0)
    sem = (
        arrs.std(axis=0, ddof=1) / np.sqrt(arrs.shape[0])
        if arrs.shape[0] > 1
        else np.zeros_like(mean)
    )
    return mean, sem


def scalar_stats(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    vals = np.asarray(vals, dtype=float)
    mean = vals.mean()
    sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return mean, sem


# Plot 1: Per-layer probe accuracy (aggregated with SEM)
try:
    layer_indices = runs[0].get("layer_indices", []) if runs else []

    conf_mean, conf_sem = stack_and_stats(
        [r.get("per_layer_confidence_acc", []) for r in runs]
    )
    corr_mean, corr_sem = stack_and_stats(
        [r.get("per_layer_correctness_acc", []) for r in runs]
    )
    ctrl_mean, ctrl_sem = stack_and_stats(
        [r.get("per_layer_control_conf_acc", []) for r in runs]
    )

    maj_conf_mean, maj_conf_sem = scalar_stats(
        [r.get("majority_baseline_conf") for r in runs]
    )
    maj_corr_mean, maj_corr_sem = scalar_stats(
        [r.get("majority_baseline_corr") for r in runs]
    )

    plt.figure(figsize=(10, 6))
    if conf_mean is not None:
        x = np.array(layer_indices[: len(conf_mean)])
        plt.errorbar(
            x,
            conf_mean,
            yerr=conf_sem,
            fmt="o-",
            label="Post-answer → Confidence (mean ± SEM)",
            capsize=3,
        )
    if ctrl_mean is not None:
        x = np.array(layer_indices[: len(ctrl_mean)])
        plt.errorbar(
            x,
            ctrl_mean,
            yerr=ctrl_sem,
            fmt="s--",
            label="Pre-answer (control) → Confidence (mean ± SEM)",
            capsize=3,
        )
    if corr_mean is not None:
        x = np.array(layer_indices[: len(corr_mean)])
        plt.errorbar(
            x,
            corr_mean,
            yerr=corr_sem,
            fmt="^-",
            label="Post-answer → Correctness (mean ± SEM)",
            capsize=3,
        )
    if maj_conf_mean is not None:
        plt.axhline(
            maj_conf_mean,
            color="gray",
            linestyle=":",
            label=f"Majority conf mean ({maj_conf_mean:.3f})",
        )
    if maj_corr_mean is not None:
        plt.axhline(
            maj_corr_mean,
            color="black",
            linestyle=":",
            label=f"Majority corr mean ({maj_corr_mean:.3f})",
        )
    plt.xlabel("Layer")
    plt.ylabel("Probe accuracy")
    plt.title(
        f"TriviaQA (Gemma-3-27b-pt): Probe Accuracy vs Layer (Aggregated over {n_runs} runs)\nMean ± SEM for Confidence, Correctness, and Pre-answer Control Probes"
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

# Plot 2: Aggregated distribution of raw verbal confidences
try:
    all_raw_conf = []
    for r in runs:
        rc = np.asarray(r.get("raw_confidences", []), dtype=float)
        if len(rc):
            all_raw_conf.append(rc)
    plt.figure(figsize=(8, 5))
    if all_raw_conf:
        pooled = np.concatenate(all_raw_conf)
        plt.hist(pooled, bins=20, color="steelblue", edgecolor="black", alpha=0.8)
        run_means = np.array([a.mean() for a in all_raw_conf])
        overall_mean = pooled.mean()
        sem = (
            run_means.std(ddof=1) / np.sqrt(len(run_means))
            if len(run_means) > 1
            else 0.0
        )
        plt.axvline(
            overall_mean,
            color="red",
            linestyle="--",
            label=f"mean={overall_mean:.2f} ± {sem:.2f} SEM",
        )
        plt.legend()
    plt.xlabel("Continuous verbal confidence")
    plt.ylabel("Count (pooled across runs)")
    plt.title(
        f"TriviaQA (Gemma-3-27b-pt): Aggregated Verbal Confidence Distribution\nPooled across {n_runs} runs"
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

# Plot 3: Aggregated confidence vs correctness (mean ± SEM bar chart)
try:
    incorrect_means = []
    correct_means = []
    for r in runs:
        rc = np.asarray(r.get("raw_confidences", []), dtype=float)
        rk = np.asarray(r.get("raw_correctness", []), dtype=float)
        if len(rc) and len(rc) == len(rk):
            if (rk == 0).any():
                incorrect_means.append(rc[rk == 0].mean())
            if (rk == 1).any():
                correct_means.append(rc[rk == 1].mean())

    inc_m, inc_s = scalar_stats(incorrect_means)
    cor_m, cor_s = scalar_stats(correct_means)

    plt.figure(figsize=(7, 5))
    if inc_m is not None and cor_m is not None:
        plt.bar(
            ["Incorrect", "Correct"],
            [inc_m, cor_m],
            yerr=[inc_s or 0, cor_s or 0],
            color=["salmon", "lightgreen"],
            edgecolor="black",
            capsize=6,
            label="Mean ± SEM across runs",
        )
        plt.legend()
    plt.ylabel("Mean verbal confidence")
    plt.title(
        f"TriviaQA (Gemma-3-27b-pt): Confidence by Correctness (Aggregated over {n_runs} runs)\nLeft: Incorrect Answers, Right: Correct Answers"
    )
    plt.grid(alpha=0.3, axis="y")
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

# Plot 4: Aggregated calibration curve
try:
    # Use common bin edges based on pooled data
    all_conf, all_corr = [], []
    for r in runs:
        rc = np.asarray(r.get("raw_confidences", []), dtype=float)
        rk = np.asarray(r.get("raw_correctness", []), dtype=float)
        if len(rc) and len(rc) == len(rk):
            all_conf.append(rc)
            all_corr.append(rk)

    plt.figure(figsize=(8, 5))
    if all_conf:
        pooled_conf = np.concatenate(all_conf)
        bins = np.linspace(pooled_conf.min(), pooled_conf.max() + 1e-6, 6)
        centers = (bins[:-1] + bins[1:]) / 2

        # per-run accuracy per bin
        per_run_bin_acc = []  # shape (n_runs, n_bins) with nan for empty
        for rc, rk in zip(all_conf, all_corr):
            bin_ids = np.clip(np.digitize(rc, bins) - 1, 0, len(bins) - 2)
            accs = []
            for b in range(len(bins) - 1):
                mask = bin_ids == b
                accs.append(rk[mask].mean() if mask.sum() > 0 else np.nan)
            per_run_bin_acc.append(accs)
        per_run_bin_acc = np.array(per_run_bin_acc, dtype=float)

        mean_acc = np.nanmean(per_run_bin_acc, axis=0)
        if per_run_bin_acc.shape[0] > 1:
            sem_acc = np.nanstd(per_run_bin_acc, axis=0, ddof=1) / np.sqrt(
                np.sum(~np.isnan(per_run_bin_acc), axis=0).clip(min=1)
            )
        else:
            sem_acc = np.zeros_like(mean_acc)

        plt.errorbar(
            centers,
            mean_acc,
            yerr=sem_acc,
            fmt="o-",
            color="purple",
            capsize=4,
            label="Accuracy per bin (mean ± SEM)",
        )

        overall_accs = [rk.mean() for rk in all_corr]
        oa_m, oa_s = scalar_stats(overall_accs)
        if oa_m is not None:
            plt.axhline(
                oa_m,
                color="gray",
                linestyle=":",
                label=f"overall acc={oa_m:.3f} ± {oa_s:.3f} SEM",
            )
    plt.xlabel("Verbal confidence (bin center)")
    plt.ylabel("Empirical accuracy")
    plt.title(
        f"TriviaQA (Gemma-3-27b-pt): Aggregated Calibration Curve ({n_runs} runs)\nEmpirical Accuracy vs Reported Verbal Confidence (Mean ± SEM)"
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

# Plot 5: Summary bar chart with SEM error bars
try:
    best_conf_vals, best_corr_vals = [], []
    maj_conf_vals, maj_corr_vals = [], []
    for r in runs:
        val_metrics = r.get("metrics", {}).get("val", [])
        if val_metrics:
            best_conf_vals.append(val_metrics[0].get("cache_probe_accuracy"))
            best_corr_vals.append(val_metrics[0].get("corr_probe_accuracy"))
        maj_conf_vals.append(r.get("majority_baseline_conf"))
        maj_corr_vals.append(r.get("majority_baseline_corr"))

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
    values = [bc_m or 0, mc_m or 0, br_m or 0, mr_m or 0]
    errors = [bc_s or 0, mc_s or 0, br_s or 0, mr_s or 0]
    colors = ["steelblue", "lightsteelblue", "seagreen", "lightgreen"]
    plt.figure(figsize=(8, 5))
    plt.bar(
        labels,
        values,
        yerr=errors,
        color=colors,
        edgecolor="black",
        capsize=6,
        label="Mean ± SEM",
    )
    for i, (v, e) in enumerate(zip(values, errors)):
        plt.text(i, v + (e or 0) + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylim(0, 1.1)
    plt.ylabel("Accuracy")
    plt.title(
        f"TriviaQA (Gemma-3-27b-pt): Best Probe vs Majority Baseline (Aggregated over {n_runs} runs)\nLeft Pair: Confidence, Right Pair: Correctness (Mean ± SEM)"
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

# Print aggregated key metrics
try:
    print("=== Aggregated Key Metrics (mean ± SEM across runs) ===")
    print(f"Number of runs: {n_runs}")
    mc_m, mc_s = scalar_stats([r.get("majority_baseline_conf") for r in runs])
    mr_m, mr_s = scalar_stats([r.get("majority_baseline_corr") for r in runs])
    print(f"Majority baseline (confidence): {mc_m} ± {mc_s}")
    print(f"Majority baseline (correctness): {mr_m} ± {mr_s}")

    best_conf_vals, best_corr_vals = [], []
    for r in runs:
        val_metrics = r.get("metrics", {}).get("val", [])
        if val_metrics:
            best_conf_vals.append(val_metrics[0].get("cache_probe_accuracy"))
            best_corr_vals.append(val_metrics[0].get("corr_probe_accuracy"))
    bc_m, bc_s = scalar_stats(best_conf_vals)
    br_m, br_s = scalar_stats(best_corr_vals)
    print(f"Best confidence probe accuracy: {bc_m} ± {bc_s}")
    print(f"Best correctness probe accuracy: {br_m} ± {br_s}")
except Exception as e:
    print(f"Error printing metrics: {e}")
