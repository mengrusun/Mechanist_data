import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_11-03-09_multi_modal_feature_description_attempt_0/logs/0-run/experiment_results/experiment_833243039f564323960e7d346c43e8b2_proc_1176445/experiment_data.npy",
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


def sem(a, axis=0):
    a = np.asarray(a, dtype=float)
    n = a.shape[axis] if a.ndim else 1
    if n <= 1:
        return np.zeros_like(np.mean(a, axis=axis))
    return np.std(a, axis=axis, ddof=1) / np.sqrt(n)


# Gather aggregated structures
roots = []
for ed in all_experiment_data:
    r = ed.get("N_PROBE", {}).get("imagenet_val_resnet50", {})
    if r:
        roots.append(r)

# Determine common N_PROBE values (intersection)
n_probe_sets = [set(r.get("n_probe_values", [])) for r in roots]
if n_probe_sets:
    common_n = sorted(set.intersection(*n_probe_sets), key=lambda x: float(x))
else:
    common_n = []

# Build per-N_PROBE arrays of mean metrics per run
concept_mean_runs = {n: [] for n in common_n}
random_mean_runs = {n: [] for n in common_n}
delta_runs = {n: [] for n in common_n}
concept_scores_pool = {n: [] for n in common_n}
random_scores_pool = {n: [] for n in common_n}

for r in roots:
    vm = {m["N_PROBE"]: m for m in r.get("metrics", {}).get("val", [])}
    per_n = r.get("per_n_probe", {})
    for n in common_n:
        if n in vm:
            concept_mean_runs[n].append(vm[n]["concept_consistency_score_mean"])
            random_mean_runs[n].append(vm[n]["random_baseline_mean"])
            delta_runs[n].append(vm[n]["delta"])
        if n in per_n:
            concept_scores_pool[n].append(np.asarray(per_n[n]["concept_scores"]))
            random_scores_pool[n].append(np.asarray(per_n[n]["random_scores"]))

# Determine best N_PROBE by mean delta
best_n = None
if common_n:
    best_n = max(
        common_n, key=lambda n: np.mean(delta_runs[n]) if delta_runs[n] else -np.inf
    )

# Plot 1: N_PROBE sweep with mean ± SE
try:
    xs = list(common_n)
    c_mean = np.array([np.mean(concept_mean_runs[n]) for n in xs])
    c_se = np.array([sem(concept_mean_runs[n]) for n in xs])
    r_mean = np.array([np.mean(random_mean_runs[n]) for n in xs])
    r_se = np.array([sem(random_mean_runs[n]) for n in xs])
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        xs,
        c_mean,
        yerr=c_se,
        fmt="o-",
        capsize=3,
        label=f"Concept (mean ± SE, n={len(roots)})",
    )
    plt.errorbar(
        xs,
        r_mean,
        yerr=r_se,
        fmt="s--",
        capsize=3,
        label=f"Random baseline (mean ± SE, n={len(roots)})",
    )
    plt.xscale("log")
    plt.xlabel("N_PROBE")
    plt.ylabel("Mean pairwise CLIP cosine similarity")
    plt.title(
        "ImageNet-val ResNet-50: Aggregated Concept Consistency vs N_PROBE\n"
        "Lines: mean across runs, error bars: standard error"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "imagenet_val_resnet50_agg_nprobe_sweep.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Delta bar chart with SE error bars
try:
    xs = list(common_n)
    d_mean = np.array([np.mean(delta_runs[n]) for n in xs])
    d_se = np.array([sem(delta_runs[n]) for n in xs])
    plt.figure(figsize=(6, 4))
    plt.bar(
        [str(x) for x in xs],
        d_mean,
        yerr=d_se,
        capsize=4,
        color="steelblue",
        label=f"Delta mean ± SE (n={len(roots)})",
    )
    plt.xlabel("N_PROBE")
    plt.ylabel("Delta (concept - random)")
    plt.title(
        "ImageNet-val ResNet-50: Aggregated Concept Improvement Over Random\n"
        "Bars: mean(delta) across runs, error bars: standard error"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "imagenet_val_resnet50_agg_delta_bar.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Aggregated per-channel score histogram at best N_PROBE
try:
    if best_n is not None and concept_scores_pool[best_n]:
        cs = np.concatenate(concept_scores_pool[best_n])
        rs = np.concatenate(random_scores_pool[best_n])
        plt.figure(figsize=(6, 4))
        plt.hist(
            cs,
            bins=25,
            alpha=0.6,
            label=f"Concept (mean={cs.mean():.3f})",
            color="tab:blue",
        )
        plt.hist(
            rs,
            bins=25,
            alpha=0.6,
            label=f"Random (mean={rs.mean():.3f})",
            color="tab:orange",
        )
        plt.axvline(cs.mean(), color="tab:blue", linestyle="--", linewidth=1)
        plt.axvline(rs.mean(), color="tab:orange", linestyle="--", linewidth=1)
        plt.xlabel("Mean pairwise CLIP cosine similarity")
        plt.ylabel("Number of channels (pooled across runs)")
        plt.title(
            f"ImageNet-val ResNet-50: Aggregated Per-Channel Score Distribution "
            f"(N_PROBE={best_n})\nPooled across {len(roots)} run(s); dashed lines: means"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "imagenet_val_resnet50_agg_score_hist_best.png"),
            dpi=140,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Aggregated scatter concept vs random at best N_PROBE
try:
    if best_n is not None and concept_scores_pool[best_n]:
        cs = np.concatenate(concept_scores_pool[best_n])
        rs = np.concatenate(random_scores_pool[best_n])
        plt.figure(figsize=(5, 5))
        plt.scatter(rs, cs, alpha=0.5, s=15, label="Per-channel (pooled)")
        lo = float(min(cs.min(), rs.min()))
        hi = float(max(cs.max(), rs.max()))
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
        plt.xlabel("Random baseline score")
        plt.ylabel("Concept (top-k) score")
        plt.title(
            f"ImageNet-val ResNet-50: Aggregated Per-Channel Concept vs Random "
            f"(N_PROBE={best_n})\nPoints above diagonal indicate meaningful concept channels"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "imagenet_val_resnet50_agg_scatter_best.png"),
            dpi=140,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Boxplot of pooled per-channel concept scores across N_PROBE
try:
    data = [
        np.concatenate(concept_scores_pool[n])
        for n in common_n
        if concept_scores_pool[n]
    ]
    labels = [str(n) for n in common_n if concept_scores_pool[n]]
    if data:
        plt.figure(figsize=(6, 4))
        plt.boxplot(data, labels=labels, showmeans=True)
        means = [d.mean() for d in data]
        plt.plot(range(1, len(means) + 1), means, "r-o", label="Mean (pooled)")
        plt.xlabel("N_PROBE")
        plt.ylabel("Per-channel concept score")
        plt.title(
            "ImageNet-val ResNet-50: Aggregated Per-Channel Concept Score Distribution\n"
            f"Boxplots pooled across {len(roots)} run(s); red line: pooled means"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "imagenet_val_resnet50_agg_boxplot_nprobe.png"),
            dpi=140,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print aggregated summary metrics
try:
    print(f"Number of runs aggregated: {len(roots)}")
    for n in common_n:
        cm = np.mean(concept_mean_runs[n]) if concept_mean_runs[n] else float("nan")
        cse = sem(concept_mean_runs[n]) if concept_mean_runs[n] else float("nan")
        rm = np.mean(random_mean_runs[n]) if random_mean_runs[n] else float("nan")
        rse = sem(random_mean_runs[n]) if random_mean_runs[n] else float("nan")
        dm = np.mean(delta_runs[n]) if delta_runs[n] else float("nan")
        dse = sem(delta_runs[n]) if delta_runs[n] else float("nan")
        print(
            f"N_PROBE={n}: concept={cm:.4f}±{cse:.4f}, "
            f"random={rm:.4f}±{rse:.4f}, delta={dm:.4f}±{dse:.4f}"
        )
    print(f"Best N_PROBE (by mean delta): {best_n}")
except Exception as e:
    print(f"Error printing metrics: {e}")
