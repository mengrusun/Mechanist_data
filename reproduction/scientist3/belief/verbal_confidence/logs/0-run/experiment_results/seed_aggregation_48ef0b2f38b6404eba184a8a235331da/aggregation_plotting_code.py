import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_22-41-30_verbal_confidence_attempt_2/logs/0-run/experiment_results/experiment_ad2777dd214144869b53e6a79b6970a2_proc_329916/experiment_data.npy"
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

key = "triviaqa_gemma3_27b_pt"
runs = [ed.get(key, {}) for ed in all_experiment_data if key in ed]
n_runs = len(runs)
print(f"Number of runs aggregated: {n_runs}")


def sem(x, axis=0):
    x = np.asarray(x, dtype=float)
    n = x.shape[axis]
    if n <= 1:
        return np.zeros(x.shape[1:] if x.ndim > 1 else ())
    return x.std(axis=axis, ddof=1) / np.sqrt(n)


# Plot 1: Per-layer probe accuracy aggregated
try:
    layer_indices = None
    conf_stack, corr_stack, ctrl_stack = [], [], []
    maj_conf_list, maj_corr_list = [], []
    for r in runs:
        li = r.get("layer_indices", [])
        if len(li) == 0:
            continue
        if layer_indices is None:
            layer_indices = li
        if len(r.get("per_layer_confidence_acc", [])) == len(layer_indices):
            conf_stack.append(r["per_layer_confidence_acc"])
        if len(r.get("per_layer_correctness_acc", [])) == len(layer_indices):
            corr_stack.append(r["per_layer_correctness_acc"])
        if len(r.get("per_layer_control_conf_acc", [])) == len(layer_indices):
            ctrl_stack.append(r["per_layer_control_conf_acc"])
        if r.get("majority_baseline_conf") is not None:
            maj_conf_list.append(r["majority_baseline_conf"])
        if r.get("majority_baseline_corr") is not None:
            maj_corr_list.append(r["majority_baseline_corr"])

    plt.figure(figsize=(10, 6))
    if conf_stack:
        arr = np.array(conf_stack)
        m, s = arr.mean(0), sem(arr, 0)
        plt.errorbar(
            layer_indices,
            m,
            yerr=s,
            fmt="o-",
            label=f"Post-answer→Conf (mean±SEM, n={len(conf_stack)})",
        )
    if ctrl_stack:
        arr = np.array(ctrl_stack)
        m, s = arr.mean(0), sem(arr, 0)
        plt.errorbar(
            layer_indices,
            m,
            yerr=s,
            fmt="s--",
            label=f"Pre-answer(ctrl)→Conf (mean±SEM)",
        )
    if corr_stack:
        arr = np.array(corr_stack)
        m, s = arr.mean(0), sem(arr, 0)
        plt.errorbar(
            layer_indices, m, yerr=s, fmt="^-", label=f"Post-answer→Corr (mean±SEM)"
        )
    if maj_conf_list:
        mc = np.mean(maj_conf_list)
        plt.axhline(
            mc, color="gray", linestyle=":", label=f"Majority conf mean={mc:.3f}"
        )
    if maj_corr_list:
        mc = np.mean(maj_corr_list)
        plt.axhline(
            mc, color="black", linestyle=":", label=f"Majority corr mean={mc:.3f}"
        )
    plt.xlabel("Layer")
    plt.ylabel("Probe accuracy")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Aggregated Probe Accuracy vs Layer\nMean ± SEM across runs"
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
    all_conf = []
    for r in runs:
        rc = r.get("raw_confidences", [])
        if len(rc):
            all_conf.append(np.array(rc))
    plt.figure(figsize=(8, 5))
    if all_conf:
        combined = np.concatenate(all_conf)
        plt.hist(
            combined,
            bins=20,
            color="steelblue",
            edgecolor="black",
            alpha=0.7,
            label=f"All runs pooled (N={len(combined)})",
        )
        means = [a.mean() for a in all_conf]
        overall_mean = np.mean(means)
        overall_sem = sem(np.array(means))
        plt.axvline(
            overall_mean,
            color="red",
            linestyle="--",
            label=f"mean of run means={overall_mean:.2f} ± {overall_sem:.2f} (SEM)",
        )
        plt.legend()
    plt.xlabel("Continuous verbal confidence")
    plt.ylabel("Count")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Aggregated Distribution of Verbal Confidences\nPooled across runs"
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

# Plot 3: Aggregated calibration curve
try:
    # Determine common bin edges from pooled data
    all_conf, all_corr = [], []
    for r in runs:
        rc = np.array(r.get("raw_confidences", []))
        rcorr = np.array(r.get("raw_correctness", []))
        if len(rc) and len(rc) == len(rcorr):
            all_conf.append(rc)
            all_corr.append(rcorr)

    plt.figure(figsize=(8, 5))
    if all_conf:
        pooled = np.concatenate(all_conf)
        bins = np.linspace(pooled.min(), pooled.max() + 1e-6, 6)
        centers = (bins[:-1] + bins[1:]) / 2
        # per-run accuracy per bin
        per_run_accs = []
        for rc, rcorr in zip(all_conf, all_corr):
            bin_ids = np.clip(np.digitize(rc, bins) - 1, 0, len(bins) - 2)
            accs = []
            for b in range(len(bins) - 1):
                mask = bin_ids == b
                accs.append(rcorr[mask].mean() if mask.sum() > 0 else np.nan)
            per_run_accs.append(accs)
        per_run_accs = np.array(per_run_accs, dtype=float)
        mean_acc = np.nanmean(per_run_accs, axis=0)
        # SEM ignoring NaNs
        counts = np.sum(~np.isnan(per_run_accs), axis=0)
        std_acc = (
            np.nanstd(per_run_accs, axis=0, ddof=1)
            if per_run_accs.shape[0] > 1
            else np.zeros_like(mean_acc)
        )
        sem_acc = np.where(counts > 1, std_acc / np.sqrt(counts), 0)

        plt.errorbar(
            centers,
            mean_acc,
            yerr=sem_acc,
            fmt="o-",
            color="purple",
            label=f"Mean accuracy per bin ± SEM (n_runs={len(all_conf)})",
        )
        overall_acc = np.concatenate(all_corr).mean()
        plt.axhline(
            overall_acc,
            color="gray",
            linestyle=":",
            label=f"overall acc={overall_acc:.3f}",
        )
    plt.xlabel("Verbal confidence (bin center)")
    plt.ylabel("Empirical accuracy")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Aggregated Calibration Curve\nMean ± SEM across runs"
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
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Summary bar chart with error bars
try:
    best_conf_list, best_corr_list = [], []
    maj_conf_list, maj_corr_list = [], []
    for r in runs:
        vm = r.get("metrics", {}).get("val", [])
        if vm:
            if vm[0].get("cache_probe_accuracy") is not None:
                best_conf_list.append(vm[0]["cache_probe_accuracy"])
            if vm[0].get("corr_probe_accuracy") is not None:
                best_corr_list.append(vm[0]["corr_probe_accuracy"])
        if r.get("majority_baseline_conf") is not None:
            maj_conf_list.append(r["majority_baseline_conf"])
        if r.get("majority_baseline_corr") is not None:
            maj_corr_list.append(r["majority_baseline_corr"])

    def ms(lst):
        if not lst:
            return 0, 0
        a = np.array(lst, dtype=float)
        return a.mean(), sem(a) if len(a) > 1 else 0

    bc_m, bc_s = ms(best_conf_list)
    mc_m, mc_s = ms(maj_conf_list)
    br_m, br_s = ms(best_corr_list)
    mr_m, mr_s = ms(maj_corr_list)

    labels = [
        "Conf Probe\n(best layer)",
        "Conf Majority",
        "Corr Probe\n(best layer)",
        "Corr Majority",
    ]
    values = [bc_m, mc_m, br_m, mr_m]
    errors = [bc_s, mc_s, br_s, mr_s]
    colors = ["steelblue", "lightsteelblue", "seagreen", "lightgreen"]

    plt.figure(figsize=(8, 5))
    plt.bar(
        labels,
        values,
        yerr=errors,
        color=colors,
        edgecolor="black",
        capsize=6,
        label=f"Mean ± SEM (n_runs={n_runs})",
    )
    for i, (v, e) in enumerate(zip(values, errors)):
        plt.text(i, v + e + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    plt.ylim(0, 1.1)
    plt.ylabel("Accuracy")
    plt.title(
        "TriviaQA (Gemma-3-27b-pt): Aggregated Probe Accuracy vs Majority Baseline\nLeft Pair: Confidence, Right Pair: Correctness"
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
    print(f"Error creating plot4: {e}")
    plt.close()

# Print aggregated metrics
try:
    print("=== Aggregated Key Metrics ===")
    print(f"Number of runs: {n_runs}")

    def summ(name, lst):
        if lst:
            a = np.array(lst, dtype=float)
            print(f"{name}: mean={a.mean():.4f}, SEM={sem(a):.4f}, n={len(a)}")
        else:
            print(f"{name}: no data")

    summ(
        "Majority baseline (confidence)",
        [
            r.get("majority_baseline_conf")
            for r in runs
            if r.get("majority_baseline_conf") is not None
        ],
    )
    summ(
        "Majority baseline (correctness)",
        [
            r.get("majority_baseline_corr")
            for r in runs
            if r.get("majority_baseline_corr") is not None
        ],
    )
    summ(
        "Best cache probe accuracy (conf)",
        [
            r.get("metrics", {}).get("val", [{}])[0].get("cache_probe_accuracy")
            for r in runs
            if r.get("metrics", {}).get("val")
        ],
    )
    summ(
        "Best corr probe accuracy",
        [
            r.get("metrics", {}).get("val", [{}])[0].get("corr_probe_accuracy")
            for r in runs
            if r.get("metrics", {}).get("val")
        ],
    )
except Exception as e:
    print(f"Error printing metrics: {e}")
