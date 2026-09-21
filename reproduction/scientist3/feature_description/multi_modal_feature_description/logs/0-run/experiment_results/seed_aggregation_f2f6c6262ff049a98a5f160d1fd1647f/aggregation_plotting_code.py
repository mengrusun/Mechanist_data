import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_11-03-09_multi_modal_feature_description_attempt_0/logs/0-run/experiment_results/experiment_8f4b9565dc6242319db7214a28c0389f_proc_1122355/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        root_env = os.getenv("AI_SCIENTIST_ROOT", "")
        full = os.path.join(root_env, p) if root_env else p
        ed = np.load(full, allow_pickle=True).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# Gather per-run roots
roots = []
for ed in all_experiment_data:
    r = ed.get("N_PROBE", {}).get("imagenet_val_resnet50", {})
    if r:
        roots.append(r)

n_runs = len(roots)
print(f"Loaded {n_runs} runs.")

# Determine common set of N_PROBE values (intersection preserving order)
if roots:
    n_probe_values = list(roots[0].get("n_probe_values", []))
    for r in roots[1:]:
        n_probe_values = [n for n in n_probe_values if n in r.get("n_probe_values", [])]
else:
    n_probe_values = []

# Aggregate scalar metrics per N_PROBE across runs
agg = {n: {"concept_mean": [], "random_mean": [], "delta": []} for n in n_probe_values}
for r in roots:
    vm = {m["N_PROBE"]: m for m in r.get("metrics", {}).get("val", [])}
    for n in n_probe_values:
        if n in vm:
            agg[n]["concept_mean"].append(vm[n]["concept_consistency_score_mean"])
            agg[n]["random_mean"].append(vm[n]["random_baseline_mean"])
            agg[n]["delta"].append(vm[n]["delta"])


def mean_se(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return np.nan, np.nan
    m = float(np.mean(x))
    se = float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0
    return m, se


concept_means = np.array([mean_se(agg[n]["concept_mean"])[0] for n in n_probe_values])
concept_ses = np.array([mean_se(agg[n]["concept_mean"])[1] for n in n_probe_values])
random_means = np.array([mean_se(agg[n]["random_mean"])[0] for n in n_probe_values])
random_ses = np.array([mean_se(agg[n]["random_mean"])[1] for n in n_probe_values])
delta_means = np.array([mean_se(agg[n]["delta"])[0] for n in n_probe_values])
delta_ses = np.array([mean_se(agg[n]["delta"])[1] for n in n_probe_values])

# Determine best N_PROBE by mean delta
best_n = None
if len(n_probe_values) > 0 and np.any(~np.isnan(delta_means)):
    best_idx = int(np.nanargmax(delta_means))
    best_n = n_probe_values[best_idx]

# Plot 1: N_PROBE sweep with SE error bars (concept vs random)
try:
    plt.figure(figsize=(6, 4))
    xs = n_probe_values
    plt.errorbar(
        xs,
        concept_means,
        yerr=concept_ses,
        fmt="o-",
        label=f"Concept mean ± SE (n={n_runs})",
        capsize=3,
    )
    plt.errorbar(
        xs,
        random_means,
        yerr=random_ses,
        fmt="s--",
        label=f"Random baseline mean ± SE (n={n_runs})",
        capsize=3,
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
        os.path.join(working_dir, "imagenet_val_resnet50_nprobe_sweep_agg.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Delta bar plot with SE error bars
try:
    plt.figure(figsize=(6, 4))
    labels = [str(n) for n in n_probe_values]
    plt.bar(
        labels,
        delta_means,
        yerr=delta_ses,
        color="steelblue",
        capsize=4,
        label=f"Mean delta ± SE (n={n_runs})",
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
        os.path.join(working_dir, "imagenet_val_resnet50_delta_bar_agg.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Pooled per-channel concept vs random histogram at best N_PROBE
try:
    if best_n is None:
        raise ValueError("No best N_PROBE available.")
    cs_all, rs_all = [], []
    for r in roots:
        d = r.get("per_n_probe", {}).get(best_n, {})
        if "concept_scores" in d:
            cs_all.append(np.asarray(d["concept_scores"]))
        if "random_scores" in d:
            rs_all.append(np.asarray(d["random_scores"]))
    cs_all = np.concatenate(cs_all) if cs_all else np.array([])
    rs_all = np.concatenate(rs_all) if rs_all else np.array([])
    plt.figure(figsize=(6, 4))
    plt.hist(
        cs_all,
        bins=25,
        alpha=0.6,
        label=f"Concept (mean={cs_all.mean():.3f})",
        color="tab:blue",
    )
    plt.hist(
        rs_all,
        bins=25,
        alpha=0.6,
        label=f"Random (mean={rs_all.mean():.3f})",
        color="tab:orange",
    )
    plt.axvline(cs_all.mean(), color="tab:blue", linestyle="--", linewidth=1)
    plt.axvline(rs_all.mean(), color="tab:orange", linestyle="--", linewidth=1)
    plt.xlabel("Mean pairwise CLIP cosine similarity")
    plt.ylabel("Number of channels (pooled across runs)")
    plt.title(
        f"ImageNet-val ResNet-50: Pooled Per-Channel Score Distribution (N_PROBE={best_n})\n"
        f"Aggregated across {n_runs} run(s); dashed lines: pooled means"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "imagenet_val_resnet50_score_hist_best_agg.png"),
        dpi=140,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Scatter concept vs random pooled across runs at best N_PROBE
try:
    if best_n is None:
        raise ValueError("No best N_PROBE available.")
    cs_all, rs_all = [], []
    for r in roots:
        d = r.get("per_n_probe", {}).get(best_n, {})
        if "concept_scores" in d and "random_scores" in d:
            cs_all.append(np.asarray(d["concept_scores"]))
            rs_all.append(np.asarray(d["random_scores"]))
    cs_all = np.concatenate(cs_all) if cs_all else np.array([])
    rs_all = np.concatenate(rs_all) if rs_all else np.array([])
    plt.figure(figsize=(5, 5))
    plt.scatter(rs_all, cs_all, alpha=0.5, s=15, label="Channels (pooled)")
    if cs_all.size and rs_all.size:
        lo = float(min(cs_all.min(), rs_all.min()))
        hi = float(max(cs_all.max(), rs_all.max()))
        plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    plt.xlabel("Random baseline score")
    plt.ylabel("Concept (top-k) score")
    plt.title(
        f"ImageNet-val ResNet-50: Per-Channel Concept vs Random (N_PROBE={best_n})\n"
        f"Pooled across {n_runs} run(s); above diagonal = concept > random"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "imagenet_val_resnet50_scatter_best_agg.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Boxplot of pooled per-channel concept scores across N_PROBE
try:
    data = []
    for n in n_probe_values:
        vals = []
        for r in roots:
            d = r.get("per_n_probe", {}).get(n, {})
            if "concept_scores" in d:
                vals.append(np.asarray(d["concept_scores"]))
        data.append(np.concatenate(vals) if vals else np.array([]))
    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=[str(n) for n in n_probe_values], showmeans=True)
    means = [float(d.mean()) if d.size else np.nan for d in data]
    plt.plot(range(1, len(n_probe_values) + 1), means, "r-o", label="Pooled mean")
    plt.xlabel("N_PROBE")
    plt.ylabel("Per-channel concept score (pooled)")
    plt.title(
        "ImageNet-val ResNet-50: Pooled Per-Channel Concept Score Distribution\n"
        f"Boxplots aggregated across {n_runs} run(s); red line: pooled means"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "imagenet_val_resnet50_boxplot_nprobe_agg.png"),
        dpi=140,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print aggregated summary
try:
    print("Aggregated metrics (mean ± SE across runs):")
    for i, n in enumerate(n_probe_values):
        print(
            f"  N_PROBE={n}: concept={concept_means[i]:.4f}±{concept_ses[i]:.4f}, "
            f"random={random_means[i]:.4f}±{random_ses[i]:.4f}, "
            f"delta={delta_means[i]:.4f}±{delta_ses[i]:.4f}"
        )
    print(f"Best N_PROBE (by mean delta): {best_n}")
except Exception as e:
    print(f"Error printing metrics: {e}")
