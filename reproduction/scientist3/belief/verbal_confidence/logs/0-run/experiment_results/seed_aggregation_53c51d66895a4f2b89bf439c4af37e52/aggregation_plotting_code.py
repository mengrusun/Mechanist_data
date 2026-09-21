import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_22-41-30_verbal_confidence_attempt_2/logs/0-run/experiment_results/experiment_38a8558a761e48cca1ce8b53218aa9b7_proc_2071921/experiment_data.npy",
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

# Extract per-run data
runs = []
for ed in all_experiment_data:
    D = ed.get("answer_span_mean_pooling", {}).get("triviaqa_gemma3_27b_pt", {})
    if D:
        runs.append(D)

n_runs = len(runs)
print(f"Number of runs aggregated: {n_runs}")


def sem(a, axis=0):
    a = np.asarray(a, dtype=float)
    if a.shape[axis] <= 1:
        return np.zeros(a.shape[1:] if axis == 0 else a.shape[:-1])
    return a.std(axis=axis, ddof=1) / np.sqrt(a.shape[axis])


# Plot 1: Per-layer probe accuracy with mean and SEM
try:
    if n_runs > 0:
        # Align by layer_indices (assume same across runs; use min length)
        layer_indices_list = [np.array(r.get("layer_indices", [])) for r in runs]
        min_len = min(len(li) for li in layer_indices_list)
        layer_indices = layer_indices_list[0][:min_len]

        conf_acc = np.array(
            [r.get("per_layer_confidence_acc", [])[:min_len] for r in runs]
        )
        corr_acc = np.array(
            [r.get("per_layer_correctness_acc", [])[:min_len] for r in runs]
        )
        ctrl_acc = np.array(
            [r.get("per_layer_control_conf_acc", [])[:min_len] for r in runs]
        )

        plt.figure(figsize=(10, 6))
        for arr, label, marker, color in [
            (conf_acc, "Answer-span -> Confidence", "o-", "C0"),
            (ctrl_acc, "Pre-answer (control) -> Confidence", "s--", "C1"),
            (corr_acc, "Answer-span -> Correctness", "^-", "C2"),
        ]:
            m = arr.mean(axis=0)
            se = sem(arr, axis=0)
            plt.plot(
                layer_indices,
                m,
                marker,
                label=f"{label} (mean, n={n_runs})",
                color=color,
            )
            plt.fill_between(
                layer_indices,
                m - se,
                m + se,
                alpha=0.2,
                color=color,
                label=f"{label} ± SEM",
            )

        maj_confs = [r.get("majority_baseline_conf", 0) for r in runs]
        maj_corrs = [r.get("majority_baseline_corr", 0) for r in runs]
        plt.axhline(
            np.mean(maj_confs),
            color="gray",
            linestyle=":",
            label=f"Majority conf mean ({np.mean(maj_confs):.3f})",
        )
        plt.axhline(
            np.mean(maj_corrs),
            color="black",
            linestyle=":",
            label=f"Majority corr mean ({np.mean(maj_corrs):.3f})",
        )

        plt.xlabel("Layer")
        plt.ylabel("Probe Accuracy")
        plt.title(
            "TriviaQA (Gemma-3-27b-pt): Per-Layer Probe Accuracy (Aggregated)\n"
            "Mean ± Standard Error across runs; Answer-Span Mean Pooling vs Control"
        )
        plt.legend(fontsize=8, loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "triviaqa_per_layer_probe_accuracy_aggregated.png"
            ),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Pooled confidence distribution
try:
    if n_runs > 0:
        all_confs = np.concatenate(
            [np.array(r.get("raw_confidences", [])) for r in runs]
        )
        all_corrs = np.concatenate(
            [np.array(r.get("raw_correctness", [])) for r in runs]
        )

        plt.figure(figsize=(10, 6))
        if len(all_confs) > 0:
            plt.hist(
                all_confs[all_corrs == 1],
                bins=20,
                alpha=0.6,
                label=f"Correct (n={int((all_corrs==1).sum())})",
                color="green",
            )
            plt.hist(
                all_confs[all_corrs == 0],
                bins=20,
                alpha=0.6,
                label=f"Incorrect (n={int((all_corrs==0).sum())})",
                color="red",
            )
        plt.xlabel("Model Continuous Confidence (0-100)")
        plt.ylabel("Count")
        plt.title(
            f"TriviaQA (Gemma-3-27b-pt): Pooled Confidence Distribution ({n_runs} runs)\n"
            "Overlaid: Correct (green) vs Incorrect (red) predictions"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "triviaqa_confidence_distribution_aggregated.png"
            ),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Calibration plot with mean and SEM across runs
try:
    if n_runs > 0:
        bins = np.linspace(0, 100, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        per_run_accs = np.full((n_runs, 10), np.nan)
        per_run_counts = np.zeros((n_runs, 10))
        for i, r in enumerate(runs):
            confs = np.array(r.get("raw_confidences", []))
            corrs = np.array(r.get("raw_correctness", []))
            if len(confs) == 0:
                continue
            bin_ids = np.clip(np.digitize(confs, bins) - 1, 0, 9)
            for b in range(10):
                mask = bin_ids == b
                if mask.sum() > 0:
                    per_run_accs[i, b] = corrs[mask].mean()
                    per_run_counts[i, b] = mask.sum()

        mean_accs = np.nanmean(per_run_accs, axis=0)
        # SEM ignoring NaNs
        se_accs = np.array(
            [
                (
                    (
                        np.nanstd(per_run_accs[:, b], ddof=1)
                        / np.sqrt(np.sum(~np.isnan(per_run_accs[:, b])))
                    )
                    if np.sum(~np.isnan(per_run_accs[:, b])) > 1
                    else 0
                )
                for b in range(10)
            ]
        )
        total_counts = per_run_counts.sum(axis=0)

        plt.figure(figsize=(10, 6))
        plt.plot([0, 100], [0, 1], "k--", label="Perfect calibration")
        valid = ~np.isnan(mean_accs)
        plt.errorbar(
            bin_centers[valid],
            mean_accs[valid],
            yerr=se_accs[valid],
            fmt="o-",
            color="blue",
            capsize=4,
            label=f"Mean empirical accuracy ± SEM (n={n_runs})",
        )
        for x, y, c in zip(bin_centers[valid], mean_accs[valid], total_counts[valid]):
            plt.annotate(
                f"{int(c)}",
                (x, y),
                textcoords="offset points",
                xytext=(0, 8),
                fontsize=8,
                ha="center",
            )

        plt.xlabel("Predicted Confidence (0-100)")
        plt.ylabel("Empirical Accuracy")
        plt.title(
            "TriviaQA (Gemma-3-27b-pt): Calibration Plot (Aggregated)\n"
            "Mean ± SEM across runs; annotations = total sample counts"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "triviaqa_calibration_plot_aggregated.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Best-layer summary bar chart with SEM error bars
try:
    if n_runs > 0:
        best_conf = np.array([r.get("cache_probe_accuracy", 0) for r in runs])
        maj_conf = np.array([r.get("majority_baseline_conf", 0) for r in runs])
        maj_corr = np.array([r.get("majority_baseline_corr", 0) for r in runs])
        best_corr = np.array(
            [
                (
                    max(r.get("per_layer_correctness_acc", [0]))
                    if len(r.get("per_layer_correctness_acc", [])) > 0
                    else 0
                )
                for r in runs
            ]
        )

        labels = [
            "Confidence Probe\n(Best Layer)",
            "Confidence\nMajority Baseline",
            "Correctness Probe\n(Best Layer)",
            "Correctness\nMajority Baseline",
        ]
        means = [best_conf.mean(), maj_conf.mean(), best_corr.mean(), maj_corr.mean()]
        sems = [
            sem(best_conf.reshape(-1, 1)).item() if n_runs > 1 else 0,
            sem(maj_conf.reshape(-1, 1)).item() if n_runs > 1 else 0,
            sem(best_corr.reshape(-1, 1)).item() if n_runs > 1 else 0,
            sem(maj_corr.reshape(-1, 1)).item() if n_runs > 1 else 0,
        ]
        colors = ["steelblue", "lightgray", "darkorange", "lightgray"]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(
            labels,
            means,
            color=colors,
            yerr=sems,
            capsize=6,
            label=f"Mean ± SEM (n={n_runs})",
        )
        for i, (m, s) in enumerate(zip(means, sems)):
            plt.text(i, m + s + 0.01, f"{m:.3f}±{s:.3f}", ha="center", fontsize=9)
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.0)
        plt.title(
            "TriviaQA (Gemma-3-27b-pt): Best Probe vs Majority Baseline (Aggregated)\n"
            "Left pair: Confidence Prediction, Right pair: Correctness Prediction"
        )
        plt.legend()
        plt.grid(alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "triviaqa_best_probe_vs_baseline_aggregated.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Print aggregated summary metrics
try:
    if n_runs > 0:
        best_conf = np.array([r.get("cache_probe_accuracy", 0) for r in runs])
        maj_conf = np.array([r.get("majority_baseline_conf", 0) for r in runs])
        maj_corr = np.array([r.get("majority_baseline_corr", 0) for r in runs])
        best_corr = np.array(
            [
                (
                    max(r.get("per_layer_correctness_acc", [0]))
                    if len(r.get("per_layer_correctness_acc", [])) > 0
                    else 0
                )
                for r in runs
            ]
        )
        all_corrs = np.concatenate(
            [np.array(r.get("raw_correctness", [])) for r in runs]
        )

        def ms(a):
            if len(a) <= 1:
                return f"{a.mean():.4f} (n={len(a)})"
            return f"{a.mean():.4f} ± {a.std(ddof=1)/np.sqrt(len(a)):.4f} (n={len(a)})"

        print(f"Cache probe accuracy (conf, best layer): {ms(best_conf)}")
        print(f"Best correctness probe accuracy: {ms(best_corr)}")
        print(f"Majority baseline (conf): {ms(maj_conf)}")
        print(f"Majority baseline (corr): {ms(maj_corr)}")
        if len(all_corrs) > 0:
            print(f"Overall correctness rate (pooled): {all_corrs.mean():.4f}")
        print(f"Best conf layers per run: {[r.get('best_layer_conf') for r in runs]}")
        print(f"Best corr layers per run: {[r.get('best_layer_corr') for r in runs]}")
except Exception as e:
    print(f"Error printing metrics: {e}")
