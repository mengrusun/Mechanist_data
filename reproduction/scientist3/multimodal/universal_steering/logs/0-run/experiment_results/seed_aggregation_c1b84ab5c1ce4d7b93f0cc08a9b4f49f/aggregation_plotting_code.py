import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_04-03-48_universal_steering_attempt_1/logs/0-run/experiment_results/experiment_9a4a69ccc2e245bc8284b17cfca3e1ef_proc_1647063/experiment_data.npy",
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


def sem(arr, axis=0):
    arr = np.asarray(arr, dtype=float)
    n = arr.shape[axis]
    if n <= 1:
        return np.zeros(arr.mean(axis=axis).shape)
    return arr.std(axis=axis, ddof=1) / np.sqrt(n)


# Gather per-run data
runs_per_alpha = []  # list of dicts per run: {alpha: {...}}
baselines = []
best_alphas = []
for ed in all_experiment_data:
    cb = ed.get("alpha_tuning", {}).get("concept_benchmark", {})
    runs_per_alpha.append(cb.get("per_alpha", {}))
    b = cb.get("overall_baseline_success_rate", None)
    if b is not None:
        baselines.append(b)
    ba = cb.get("best_alpha", None)
    if ba is not None:
        best_alphas.append(ba)

# Determine common alpha keys
if runs_per_alpha:
    common_keys = set(runs_per_alpha[0].keys())
    for r in runs_per_alpha[1:]:
        common_keys &= set(r.keys())
    alpha_keys = sorted(common_keys, key=lambda k: runs_per_alpha[0][k]["alpha"])
else:
    alpha_keys = []

alphas = [runs_per_alpha[0][k]["alpha"] for k in alpha_keys] if alpha_keys else []
baseline_mean = float(np.mean(baselines)) if baselines else None
baseline_sem = float(sem(baselines)) if len(baselines) > 1 else 0.0
n_runs = len(all_experiment_data)

# Plot 1: Overall success rate vs alpha (aggregated)
try:
    plt.figure(figsize=(6, 4))
    ys_mat = np.array(
        [
            [runs_per_alpha[i][k]["overall_steering_success_rate"] for k in alpha_keys]
            for i in range(n_runs)
        ]
    )
    means = ys_mat.mean(axis=0)
    errs = sem(ys_mat, axis=0)
    plt.errorbar(
        alphas,
        means,
        yerr=errs,
        marker="o",
        label=f"steered mean ± SEM (n={n_runs})",
        capsize=3,
    )
    if baseline_mean is not None:
        plt.axhline(
            baseline_mean,
            color="gray",
            linestyle="--",
            label=f"baseline mean={baseline_mean:.3f}",
        )
        if n_runs > 1:
            plt.fill_between(
                alphas,
                baseline_mean - baseline_sem,
                baseline_mean + baseline_sem,
                color="gray",
                alpha=0.2,
                label="baseline SEM",
            )
    plt.xlabel("α (steering strength)")
    plt.ylabel("Overall success rate")
    plt.title(
        "Concept Benchmark: Aggregated Overall Success Rate vs α\n(Mean ± SEM across runs)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "concept_benchmark_agg_overall_success_vs_alpha.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Per-class success rate across alphas (aggregated)
try:
    if alpha_keys and n_runs > 0:
        classes = sorted(
            set(r["class"] for r in runs_per_alpha[0][alpha_keys[0]]["per_concept"])
        )
        xs = np.arange(len(classes))
        width = 0.8 / (len(alphas) + 1)
        plt.figure(figsize=(11, 5))

        # baseline per class - across runs
        base_matrix = []  # runs x classes
        for i in range(n_runs):
            base_by_class = {c: [] for c in classes}
            for r in runs_per_alpha[i][alpha_keys[0]]["per_concept"]:
                base_by_class[r["class"]].append(r["baseline_success"] / r["total"])
            base_matrix.append(
                [
                    np.mean(base_by_class[c]) if base_by_class[c] else 0.0
                    for c in classes
                ]
            )
        base_matrix = np.array(base_matrix)
        base_means = base_matrix.mean(axis=0)
        base_errs = sem(base_matrix, axis=0)
        plt.bar(
            xs - 0.4 + width / 2,
            base_means,
            width=width,
            yerr=base_errs,
            capsize=2,
            label="baseline",
        )

        for j, a in enumerate(alphas):
            akey = f"alpha_{a}"
            mat = []
            for i in range(n_runs):
                cls_s = {c: [] for c in classes}
                for r in runs_per_alpha[i][akey]["per_concept"]:
                    cls_s[r["class"]].append(r["steered_success"] / r["total"])
                mat.append([np.mean(cls_s[c]) if cls_s[c] else 0.0 for c in classes])
            mat = np.array(mat)
            means = mat.mean(axis=0)
            errs = sem(mat, axis=0)
            plt.bar(
                xs - 0.4 + width * (j + 1.5),
                means,
                width=width,
                yerr=errs,
                capsize=2,
                label=f"α={a}",
            )
        plt.xticks(xs, classes, rotation=30)
        plt.ylabel("Success rate (mean ± SEM)")
        plt.title(
            "Concept Benchmark: Aggregated Success Rate by Concept Class\n(Baseline vs Steered α; error bars = SEM)"
        )
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "concept_benchmark_agg_success_by_class.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Val loss vs alpha (aggregated)
try:
    plt.figure(figsize=(6, 4))
    vl_mat = np.array(
        [[runs_per_alpha[i][k]["val_loss"] for k in alpha_keys] for i in range(n_runs)]
    )
    means = vl_mat.mean(axis=0)
    errs = sem(vl_mat, axis=0)
    plt.errorbar(
        alphas,
        means,
        yerr=errs,
        marker="s",
        color="red",
        capsize=3,
        label=f"val loss mean ± SEM (n={n_runs})",
    )
    plt.xlabel("α (steering strength)")
    plt.ylabel("Validation loss (1 - success)")
    plt.title(
        "Concept Benchmark: Aggregated Validation Loss vs α\n(Mean ± SEM across runs)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "concept_benchmark_agg_val_loss_vs_alpha.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Running steered success rate curves (aggregated per alpha)
try:
    plt.figure(figsize=(8, 5))
    for k in alpha_keys:
        # Collect running metrics across runs; align by concept_index
        series = []
        common_idx = None
        for i in range(n_runs):
            rm = runs_per_alpha[i][k].get("running_metrics", [])
            if not rm:
                continue
            idx = [r["concept_index"] for r in rm]
            vals = [r["steered_success_rate_running"] for r in rm]
            series.append((idx, vals))
            common_idx = (
                idx if common_idx is None else [x for x in common_idx if x in set(idx)]
            )
        if not series or not common_idx:
            continue
        common_idx = sorted(common_idx)
        aligned = []
        for idx, vals in series:
            m = dict(zip(idx, vals))
            aligned.append([m[c] for c in common_idx])
        aligned = np.array(aligned)
        means = aligned.mean(axis=0)
        errs = sem(aligned, axis=0)
        alpha_val = runs_per_alpha[0][k]["alpha"]
        (line,) = plt.plot(common_idx, means, marker=".", label=f"α={alpha_val} (mean)")
        plt.fill_between(
            common_idx, means - errs, means + errs, color=line.get_color(), alpha=0.2
        )
    if baseline_mean is not None:
        plt.axhline(baseline_mean, color="gray", linestyle="--", label="baseline mean")
    plt.xlabel("Concept index (evaluation order)")
    plt.ylabel("Running steered success rate")
    plt.title("Concept Benchmark: Aggregated Running Success Rate\n(Mean ± SEM shaded)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "concept_benchmark_agg_running_success.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Print aggregated metrics
print(f"Number of runs aggregated: {n_runs}")
if baseline_mean is not None:
    print(f"Baseline success rate: mean={baseline_mean:.4f}, SEM={baseline_sem:.4f}")
for j, k in enumerate(alpha_keys):
    succ = np.array(
        [runs_per_alpha[i][k]["overall_steering_success_rate"] for i in range(n_runs)]
    )
    vl = np.array([runs_per_alpha[i][k]["val_loss"] for i in range(n_runs)])
    print(
        f"{k} (α={alphas[j]}): steered success mean={succ.mean():.4f} SEM={sem(succ):.4f} | "
        f"val_loss mean={vl.mean():.4f} SEM={sem(vl):.4f}"
    )
if best_alphas:
    print(f"Best alphas per run: {best_alphas}")
