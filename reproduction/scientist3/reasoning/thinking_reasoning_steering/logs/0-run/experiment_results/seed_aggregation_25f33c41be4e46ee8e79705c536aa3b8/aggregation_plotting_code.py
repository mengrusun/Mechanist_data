import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_07-29-12_thinking_reasoning_steering_attempt_1/logs/0-run/experiment_results/experiment_68d53627a52e4c5b8764bc078dc1b550_proc_846506/experiment_data.npy"
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        ed = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

key = "r1_distill_llama_8b_steering"
runs = [ed.get(key, {}) for ed in all_experiment_data if key in ed]
n_runs = len(runs)
print(f"Loaded {n_runs} runs")


def sem(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size <= 1:
        return 0.0
    return np.std(arr, ddof=1) / np.sqrt(arr.size)


# Collect behaviours & coeffs from first run
if runs:
    steering_results0 = runs[0].get("steering_results", {})
    behaviours = list(steering_results0.keys())
    coeffs = (
        sorted([float(k) for k in steering_results0[behaviours[0]].keys()])
        if behaviours
        else []
    )
else:
    behaviours, coeffs = [], []

# Plot 1: Aggregated baseline behaviour counts (mean ± SEM across runs)
try:
    all_bcs = [r.get("behaviour_counts", {}).get("baseline_total", {}) for r in runs]
    names = sorted({n for bc in all_bcs for n in bc.keys()})
    if names:
        means, sems = [], []
        for n in names:
            vals = [bc.get(n, 0) for bc in all_bcs]
            means.append(np.mean(vals))
            sems.append(sem(vals))
        plt.figure(figsize=(7, 4))
        plt.bar(
            names,
            means,
            yerr=sems,
            color="steelblue",
            capsize=4,
            label=f"Mean ± SEM (n={n_runs})",
        )
        plt.ylabel("Total occurrences across baseline chains")
        plt.title(
            "R1-Distill-Llama-8B: Baseline Behaviour Counts (Aggregated)\nMean ± SEM across runs"
        )
        plt.xticks(rotation=20)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "r1_llama8b_agg_baseline_behaviour_counts.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated baseline behaviour count plot: {e}")
    plt.close()

# Plot 2: Aggregated mean behaviour count per chain vs steering coefficient
try:
    if behaviours and coeffs:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(behaviours))
        w = 0.8 / len(coeffs)
        for i, c in enumerate(coeffs):
            means, errs = [], []
            for b in behaviours:
                vals = []
                for r in runs:
                    try:
                        vals.append(r["steering_results"][b][str(c)]["mean"])
                    except Exception:
                        pass
                means.append(np.mean(vals) if vals else 0)
                errs.append(sem(vals) if vals else 0)
            ax.bar(
                x + (i - (len(coeffs) - 1) / 2) * w,
                means,
                w,
                yerr=errs,
                capsize=3,
                label=f"coeff={c:+.1f}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(behaviours, rotation=20)
        ax.set_ylabel("Mean behaviour count / chain")
        ax.set_title(
            f"R1-Distill-Llama-8B: Behaviour Frequency vs Steering Coefficient\nMean ± SEM across {n_runs} run(s)"
        )
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "r1_llama8b_agg_steering_behaviour_counts.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated behaviour count vs coeff plot: {e}")
    plt.close()

# Plot 3: Aggregated task accuracy under steering
try:
    if behaviours and coeffs:
        fig, ax = plt.subplots(figsize=(9, 5))
        x = np.arange(len(behaviours))
        w = 0.8 / len(coeffs)
        for i, c in enumerate(coeffs):
            means, errs = [], []
            for b in behaviours:
                vals = []
                for r in runs:
                    try:
                        vals.append(r["steering_results"][b][str(c)]["acc"])
                    except Exception:
                        pass
                means.append(np.mean(vals) if vals else 0)
                errs.append(sem(vals) if vals else 0)
            ax.bar(
                x + (i - (len(coeffs) - 1) / 2) * w,
                means,
                w,
                yerr=errs,
                capsize=3,
                label=f"coeff={c:+.1f}",
            )
        baseline_accs = [
            r.get("baseline_accuracy")
            for r in runs
            if r.get("baseline_accuracy") is not None
        ]
        if baseline_accs:
            b_mean = np.mean(baseline_accs)
            b_sem = sem(baseline_accs)
            ax.axhline(
                b_mean,
                color="red",
                linestyle="--",
                label=f"baseline acc={b_mean:.2f}±{b_sem:.2f}",
            )
            ax.fill_between(
                [-0.5, len(behaviours) - 0.5],
                b_mean - b_sem,
                b_mean + b_sem,
                color="red",
                alpha=0.1,
            )
        ax.set_xticks(x)
        ax.set_xticklabels(behaviours, rotation=20)
        ax.set_ylabel("Task accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-0.5, len(behaviours) - 0.5)
        ax.set_title(
            f"R1-Distill-Llama-8B: Task Accuracy under Steering\nMean ± SEM across {n_runs} run(s)"
        )
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "r1_llama8b_agg_steering_task_accuracy.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated task accuracy plot: {e}")
    plt.close()

# Plot 4: Aggregated dose-response curves (mean ± SEM shaded)
try:
    if behaviours and coeffs:
        fig, ax = plt.subplots(figsize=(7, 5))
        for b in behaviours:
            means, errs = [], []
            for c in coeffs:
                vals = []
                for r in runs:
                    try:
                        vals.append(r["steering_results"][b][str(c)]["mean"])
                    except Exception:
                        pass
                means.append(np.mean(vals) if vals else 0)
                errs.append(sem(vals) if vals else 0)
            means = np.array(means)
            errs = np.array(errs)
            (line,) = ax.plot(coeffs, means, marker="o", label=f"{b} (mean)")
            ax.fill_between(
                coeffs, means - errs, means + errs, color=line.get_color(), alpha=0.2
            )
        ax.set_xlabel("Steering coefficient")
        ax.set_ylabel("Mean behaviour count / chain")
        ax.set_title(
            f"R1-Distill-Llama-8B: Dose-Response of Steering (Aggregated)\nLines: mean, Shaded: ±SEM across {n_runs} run(s)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "r1_llama8b_agg_steering_dose_response.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated dose-response plot: {e}")
    plt.close()

# Plot 5: Aggregated Pos/Neg pool sizes per behaviour
try:
    beh_names = ["hedging", "backtracking", "self_correction", "example_gen"]
    pos_all = {b: [] for b in beh_names}
    neg_all = {b: [] for b in beh_names}
    for r in runs:
        bc = r.get("behaviour_counts", {})
        for b in beh_names:
            pos_all[b].append(bc.get(f"{b}_pos", 0))
            neg_all[b].append(bc.get(f"{b}_neg", 0))
    pos_means = [np.mean(pos_all[b]) for b in beh_names]
    pos_sems = [sem(pos_all[b]) for b in beh_names]
    neg_means = [np.mean(neg_all[b]) for b in beh_names]
    neg_sems = [sem(neg_all[b]) for b in beh_names]
    if any(pos_means) or any(neg_means):
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(beh_names))
        w = 0.35
        ax.bar(
            x - w / 2,
            pos_means,
            w,
            yerr=pos_sems,
            capsize=3,
            label="Positive (mean ± SEM)",
            color="green",
        )
        ax.bar(
            x + w / 2,
            neg_means,
            w,
            yerr=neg_sems,
            capsize=3,
            label="Negative (mean ± SEM)",
            color="gray",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(beh_names, rotation=20)
        ax.set_ylabel("Number of sentence activations")
        ax.set_title(
            f"R1-Distill-Llama-8B: Pos/Neg Pool Sizes (Aggregated)\nMean ± SEM across {n_runs} run(s)"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "r1_llama8b_agg_pool_sizes.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated pool sizes plot: {e}")
    plt.close()

# Print aggregated summary metrics
try:
    rates = [
        r.get("behaviour_steering_success_rate")
        for r in runs
        if r.get("behaviour_steering_success_rate") is not None
    ]
    baseline_accs = [
        r.get("baseline_accuracy")
        for r in runs
        if r.get("baseline_accuracy") is not None
    ]
    if baseline_accs:
        print(
            f"Baseline accuracy: mean={np.mean(baseline_accs):.4f}, SEM={sem(baseline_accs):.4f}, n={len(baseline_accs)}"
        )
    if rates:
        print(
            f"Behaviour steering success rate: mean={np.mean(rates):.4f}, SEM={sem(rates):.4f}, n={len(rates)}"
        )
except Exception as e:
    print(f"Error printing summary: {e}")
