import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_04-03-48_universal_steering_attempt_1/logs/0-run/experiment_results/experiment_9aa5f7633ec04f038b7e05d7b99ab0a8_proc_1241075/experiment_data.npy",
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

# Collect per-run info
runs = []
for ed in all_experiment_data:
    d = ed.get("concept_benchmark", {})
    runs.append(d)

if not runs:
    print("No runs found.")

cfg = runs[0].get("config", {}) if runs else {}
alpha = cfg.get("alpha", "?")
layer = cfg.get("steer_layer", "?")
n_runs = len(runs)


def sem(arr):
    arr = np.array(arr, dtype=float)
    if len(arr) < 2:
        return 0.0
    return np.std(arr, ddof=1) / np.sqrt(len(arr))


# Plot 1: Aggregated success rate by concept class (mean +/- SEM across runs)
try:
    # For each run, compute mean success rate per class, then aggregate across runs
    classes_set = set()
    for d in runs:
        for r in d.get("per_concept", []):
            classes_set.add(r["class"])
    classes = sorted(classes_set)

    base_per_run = {c: [] for c in classes}
    steer_per_run = {c: [] for c in classes}
    for d in runs:
        cls_success = {c: [] for c in classes}
        cls_base = {c: [] for c in classes}
        for r in d.get("per_concept", []):
            cls_success[r["class"]].append(r["steered_success"] / r["total"])
            cls_base[r["class"]].append(r["baseline_success"] / r["total"])
        for c in classes:
            if cls_success[c]:
                steer_per_run[c].append(np.mean(cls_success[c]))
                base_per_run[c].append(np.mean(cls_base[c]))

    xs = np.arange(len(classes))
    base_means = [np.mean(base_per_run[c]) if base_per_run[c] else 0 for c in classes]
    steer_means = [
        np.mean(steer_per_run[c]) if steer_per_run[c] else 0 for c in classes
    ]
    base_sems = [sem(base_per_run[c]) for c in classes]
    steer_sems = [sem(steer_per_run[c]) for c in classes]

    plt.figure(figsize=(9, 5))
    plt.bar(
        xs - 0.2,
        base_means,
        width=0.4,
        yerr=base_sems,
        capsize=4,
        label=f"baseline (mean±SEM, n={n_runs})",
    )
    plt.bar(
        xs + 0.2,
        steer_means,
        width=0.4,
        yerr=steer_sems,
        capsize=4,
        label=f"steered (mean±SEM, n={n_runs})",
    )
    plt.xticks(xs, classes, rotation=30)
    plt.ylabel("Success rate")
    plt.title(
        f"Concept Benchmark (Aggregated): Success Rate by Class\n"
        f"Left bar: Baseline, Right bar: Steered (α={alpha}, layer={layer})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "concept_benchmark_agg_success_by_class.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Per-concept aggregated success rates (only for concepts appearing in all runs)
try:
    concept_data = {}  # concept -> {"base": [rates], "steer": [rates]}
    for d in runs:
        for r in d.get("per_concept", []):
            c = r["concept"]
            if c not in concept_data:
                concept_data[c] = {"base": [], "steer": []}
            concept_data[c]["base"].append(r["baseline_success"] / r["total"])
            concept_data[c]["steer"].append(r["steered_success"] / r["total"])

    concepts = sorted(concept_data.keys())
    base_means = [np.mean(concept_data[c]["base"]) for c in concepts]
    steer_means = [np.mean(concept_data[c]["steer"]) for c in concepts]
    base_sems = [sem(concept_data[c]["base"]) for c in concepts]
    steer_sems = [sem(concept_data[c]["steer"]) for c in concepts]

    xs = np.arange(len(concepts))
    plt.figure(figsize=(max(12, len(concepts) * 0.3), 6))
    plt.bar(
        xs - 0.2,
        base_means,
        width=0.4,
        yerr=base_sems,
        capsize=2,
        label=f"baseline (mean±SEM, n={n_runs})",
    )
    plt.bar(
        xs + 0.2,
        steer_means,
        width=0.4,
        yerr=steer_sems,
        capsize=2,
        label=f"steered (mean±SEM, n={n_runs})",
    )
    plt.xticks(xs, concepts, rotation=75, fontsize=8)
    plt.ylabel("Success rate")
    plt.title(
        f"Concept Benchmark (Aggregated): Per-Concept Success Rates\n"
        f"Left: Baseline, Right: Steered (α={alpha}, layer={layer})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "concept_benchmark_agg_per_concept_success.png")
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Aggregated running success rate over concepts
try:
    # Collect running curves; align by concept index
    all_idx = set()
    for d in runs:
        for m in d.get("metrics", {}).get("val", []):
            all_idx.add(m["concept_index"])
    idxs = sorted(all_idx)

    if idxs:
        base_matrix = []
        steer_matrix = []
        for d in runs:
            val_metrics = d.get("metrics", {}).get("val", [])
            idx_to_base = {
                m["concept_index"]: m["baseline_success_rate_running"]
                for m in val_metrics
            }
            idx_to_steer = {
                m["concept_index"]: m["steered_success_rate_running"]
                for m in val_metrics
            }
            base_row = [idx_to_base.get(i, np.nan) for i in idxs]
            steer_row = [idx_to_steer.get(i, np.nan) for i in idxs]
            base_matrix.append(base_row)
            steer_matrix.append(steer_row)
        base_matrix = np.array(base_matrix, dtype=float)
        steer_matrix = np.array(steer_matrix, dtype=float)

        base_mean = np.nanmean(base_matrix, axis=0)
        steer_mean = np.nanmean(steer_matrix, axis=0)
        base_sem = (
            np.nanstd(base_matrix, axis=0, ddof=1)
            / np.sqrt(np.sum(~np.isnan(base_matrix), axis=0))
            if n_runs > 1
            else np.zeros_like(base_mean)
        )
        steer_sem = (
            np.nanstd(steer_matrix, axis=0, ddof=1)
            / np.sqrt(np.sum(~np.isnan(steer_matrix), axis=0))
            if n_runs > 1
            else np.zeros_like(steer_mean)
        )

        plt.figure(figsize=(8, 5))
        plt.plot(idxs, base_mean, label=f"baseline mean (n={n_runs})", color="gray")
        plt.fill_between(
            idxs,
            base_mean - base_sem,
            base_mean + base_sem,
            alpha=0.3,
            color="gray",
            label="baseline ±SEM",
        )
        plt.plot(
            idxs, steer_mean, label=f"steered mean (n={n_runs})", color="steelblue"
        )
        plt.fill_between(
            idxs,
            steer_mean - steer_sem,
            steer_mean + steer_sem,
            alpha=0.3,
            color="steelblue",
            label="steered ±SEM",
        )
        plt.xlabel("Concept index")
        plt.ylabel("Running success rate")
        plt.title(
            "Concept Benchmark (Aggregated): Running Success Rate\n"
            "Baseline vs Steered across concepts evaluated"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "concept_benchmark_agg_running_success_rate.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Aggregated overall comparison
try:
    base_overall = [
        d.get("overall_baseline_success_rate")
        for d in runs
        if d.get("overall_baseline_success_rate") is not None
    ]
    steer_overall = [
        d.get("overall_steering_success_rate")
        for d in runs
        if d.get("overall_steering_success_rate") is not None
    ]

    if base_overall and steer_overall:
        base_mean = np.mean(base_overall)
        steer_mean = np.mean(steer_overall)
        base_sem_v = sem(base_overall)
        steer_sem_v = sem(steer_overall)

        plt.figure(figsize=(6, 5))
        labels = ["baseline", "steered"]
        means = [base_mean, steer_mean]
        sems = [base_sem_v, steer_sem_v]
        plt.bar(
            labels,
            means,
            yerr=sems,
            capsize=6,
            color=["gray", "steelblue"],
            label=f"mean±SEM (n={n_runs})",
        )
        plt.ylabel("Overall success rate")
        plt.ylim(0, 1)
        for i, (v, s) in enumerate(zip(means, sems)):
            plt.text(i, v + s + 0.02, f"{v:.3f}±{s:.3f}", ha="center")
        plt.title(
            f"Concept Benchmark (Aggregated): Overall Success Rate\n"
            f"Baseline vs Steered (α={alpha}, layer={layer})"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "concept_benchmark_agg_overall_success.png")
        )
        plt.close()

        print(
            f"Aggregated overall baseline success rate: {base_mean:.4f} ± {base_sem_v:.4f} (SEM, n={n_runs})"
        )
        print(
            f"Aggregated overall steered success rate: {steer_mean:.4f} ± {steer_sem_v:.4f} (SEM, n={n_runs})"
        )
        print(f"Individual baseline rates: {base_overall}")
        print(f"Individual steered rates: {steer_overall}")
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()
