import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_11-03-09_multi_modal_feature_description_attempt_0/logs/0-run/experiment_results/experiment_9ca9a381689441eb9a13ef8ca9a45553_proc_967289/experiment_data.npy"
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

key = "imagenet_val_resnet50"

# Gather per-run arrays and metrics
concept_runs = []
random_runs = []
metrics_runs = []
for ed in all_experiment_data:
    d = ed.get(key, {})
    c = np.asarray(d.get("predictions", []), dtype=float)
    r = np.asarray(d.get("ground_truth", []), dtype=float)
    if c.size:
        concept_runs.append(c)
    if r.size:
        random_runs.append(r)
    vm_list = d.get("metrics", {}).get("val", [{}])
    if vm_list:
        metrics_runs.append(vm_list[0])

n_runs = len(all_experiment_data)

# Plot 1: Aggregated histograms
try:
    if concept_runs and random_runs:
        concept_all = np.concatenate(concept_runs)
        random_all = np.concatenate(random_runs)
        plt.figure(figsize=(6, 4))
        bins = np.linspace(
            min(concept_all.min(), random_all.min()),
            max(concept_all.max(), random_all.max()),
            25,
        )
        plt.hist(
            concept_all,
            bins=bins,
            alpha=0.6,
            label=f"top-k activating (mean={concept_all.mean():.3f})",
            color="tab:blue",
        )
        plt.hist(
            random_all,
            bins=bins,
            alpha=0.6,
            label=f"random baseline (mean={random_all.mean():.3f})",
            color="tab:orange",
        )
        plt.axvline(concept_all.mean(), color="tab:blue", ls="--", lw=1)
        plt.axvline(random_all.mean(), color="tab:orange", ls="--", lw=1)
        plt.xlabel("Mean pairwise CLIP cosine similarity")
        plt.ylabel("# channels (aggregated across runs)")
        plt.title(
            f"ImageNet-val / ResNet-50 layer4\nAggregated Concept Consistency Histogram (n_runs={n_runs})"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "imagenetval_resnet50_agg_concept_hist.png"),
            dpi=140,
        )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Sorted per-channel concept scores with mean +/- SEM band across runs
try:
    if concept_runs:
        # sort each run's scores in descending order, then align by rank
        min_len = min(len(c) for c in concept_runs)
        sorted_arr = np.stack(
            [np.sort(c)[::-1][:min_len] for c in concept_runs], axis=0
        )
        mean_curve = sorted_arr.mean(axis=0)
        if sorted_arr.shape[0] > 1:
            sem_curve = sorted_arr.std(axis=0, ddof=1) / np.sqrt(sorted_arr.shape[0])
        else:
            sem_curve = np.zeros_like(mean_curve)
        x = np.arange(min_len)
        plt.figure(figsize=(7, 4))
        plt.plot(x, mean_curve, lw=1.2, color="tab:blue", label="mean concept score")
        plt.fill_between(
            x,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            alpha=0.3,
            color="tab:blue",
            label="± SEM across runs",
        )
        if random_runs:
            rand_means = [r.mean() for r in random_runs]
            rmean = float(np.mean(rand_means))
            rsem = (
                float(np.std(rand_means, ddof=1) / np.sqrt(len(rand_means)))
                if len(rand_means) > 1
                else 0.0
            )
            plt.axhline(
                rmean,
                color="r",
                ls="--",
                label=f"random baseline mean={rmean:.3f} ± {rsem:.3f}",
            )
        plt.xlabel("Channel rank")
        plt.ylabel("Mean pairwise CLIP cos sim")
        plt.title(
            f"ImageNet-val / ResNet-50 layer4\nSorted Per-Channel Concept Consistency (mean ± SEM, n_runs={n_runs})"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "imagenetval_resnet50_sorted_concept_scores_agg.png"
            ),
            dpi=140,
        )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Summary metrics bar chart with SEM error bars across runs
try:
    keys_of_interest = [
        "concept_consistency_score_mean",
        "concept_consistency_score_median",
        "random_baseline_mean",
        "delta",
    ]
    agg = {k: [] for k in keys_of_interest}
    for m in metrics_runs:
        for k in keys_of_interest:
            if k in m:
                try:
                    agg[k].append(float(m[k]))
                except Exception:
                    pass
    labels, means, sems = [], [], []
    for k in keys_of_interest:
        vals = agg[k]
        if vals:
            labels.append(k.replace("_", "\n"))
            means.append(np.mean(vals))
            sems.append(
                np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
            )
    if means:
        plt.figure(figsize=(7, 4))
        x = np.arange(len(labels))
        colors = ["tab:blue", "tab:cyan", "tab:orange", "tab:green"][: len(means)]
        plt.bar(
            x,
            means,
            yerr=sems,
            color=colors,
            capsize=5,
            label=f"mean ± SEM (n_runs={n_runs})",
        )
        for i, (m, s) in enumerate(zip(means, sems)):
            plt.text(i, m, f"{m:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=9)
        plt.xticks(x, labels)
        plt.ylabel("Score")
        plt.title(
            "ImageNet-val / ResNet-50 layer4\nAggregated Summary Metrics (CLIP-based Concept Consistency)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "imagenetval_resnet50_summary_metrics_agg.png"),
            dpi=140,
        )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Print aggregated evaluation metrics
print(f"Aggregated evaluation metrics over {n_runs} run(s):")
for k in [
    "concept_consistency_score_mean",
    "concept_consistency_score_median",
    "random_baseline_mean",
    "delta",
]:
    vals = [float(m[k]) for m in metrics_runs if k in m]
    if vals:
        mean_v = np.mean(vals)
        sem_v = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        print(f"  {k}: mean={mean_v:.4f}, SEM={sem_v:.4f}, values={vals}")
