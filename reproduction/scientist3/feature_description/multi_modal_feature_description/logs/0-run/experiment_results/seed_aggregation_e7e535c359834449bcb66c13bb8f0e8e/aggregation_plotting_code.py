import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

# Load experiment data (list of runs)
try:
    experiment_data_path_list = [
        "experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        full = (
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
            if not os.path.isabs(p)
            else p
        )
        if not os.path.exists(full):
            alt = os.path.join(working_dir, p)
            if os.path.exists(alt):
                full = alt
        try:
            ed = np.load(full, allow_pickle=True).item()
            all_experiment_data.append(ed)
        except Exception as e:
            print(f"Skipping {full}: {e}")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# Collect per-run summaries
runs = []
for ed in all_experiment_data:
    root = ed.get("N_PROBE", {}).get("imagenet_val_resnet50", {})
    if not root:
        continue
    runs.append(root)

if len(runs) == 0:
    print("No runs available.")
else:
    # Union of N_PROBE values (assume all runs share)
    n_probe_values = runs[0].get("n_probe_values", [])
    n_runs = len(runs)

    # Aggregate scalars across runs per N_PROBE
    def collect_metric(key):
        arr = np.full((n_runs, len(n_probe_values)), np.nan)
        for i, r in enumerate(runs):
            vm = r.get("metrics", {}).get("val", [])
            m_by_n = {m["N_PROBE"]: m for m in vm}
            for j, n in enumerate(n_probe_values):
                if n in m_by_n and key in m_by_n[n]:
                    arr[i, j] = m_by_n[n][key]
        return arr

    concept_mat = collect_metric("concept_consistency_score_mean")
    random_mat = collect_metric("random_baseline_mean")
    delta_mat = collect_metric("delta")

    def mean_sem(mat):
        m = np.nanmean(mat, axis=0)
        n = np.sum(~np.isnan(mat), axis=0)
        s = np.nanstd(mat, axis=0, ddof=1) if mat.shape[0] > 1 else np.zeros_like(m)
        sem = np.where(n > 1, s / np.sqrt(np.maximum(n, 1)), 0.0)
        return m, sem

    c_mean, c_sem = mean_sem(concept_mat)
    r_mean, r_sem = mean_sem(random_mat)
    d_mean, d_sem = mean_sem(delta_mat)

    # Plot 1: aggregated N_PROBE sweep with SEM
    try:
        plt.figure(figsize=(6, 4))
        xs = np.array(n_probe_values, dtype=float)
        plt.errorbar(
            xs,
            c_mean,
            yerr=c_sem,
            fmt="o-",
            label=f"Concept (mean ± SEM, n={n_runs})",
            capsize=3,
        )
        plt.errorbar(
            xs,
            r_mean,
            yerr=r_sem,
            fmt="s--",
            label=f"Random (mean ± SEM, n={n_runs})",
            capsize=3,
        )
        plt.xscale("log")
        plt.xlabel("N_PROBE")
        plt.ylabel("Mean pairwise CLIP cosine similarity")
        plt.title(
            "ImageNet-val ResNet-50: Aggregated Concept Consistency vs N_PROBE\nMean over runs with SEM error bars"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "imagenet_val_resnet50_agg_nprobe_sweep.png"),
            dpi=140,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot1: {e}")
        plt.close()

    # Plot 2: aggregated delta bar with SEM
    try:
        plt.figure(figsize=(6, 4))
        labels = [str(n) for n in n_probe_values]
        x = np.arange(len(labels))
        plt.bar(
            x,
            d_mean,
            yerr=d_sem,
            color="steelblue",
            capsize=4,
            label=f"Delta mean ± SEM (n={n_runs})",
        )
        plt.xticks(x, labels)
        plt.xlabel("N_PROBE")
        plt.ylabel("Delta (concept - random)")
        plt.title(
            "ImageNet-val ResNet-50: Aggregated Concept Score Improvement\nMean delta across runs with SEM error bars"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "imagenet_val_resnet50_agg_delta_bar.png"),
            dpi=140,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot2: {e}")
        plt.close()

    # Determine common best N_PROBE (by aggregated concept mean)
    try:
        best_idx = int(np.nanargmax(c_mean))
        best_n = n_probe_values[best_idx]
    except Exception:
        best_n = runs[0].get("best_n_probe", n_probe_values[0])

    # Plot 3: pooled per-channel histogram at best N_PROBE
    try:
        cs_all, rs_all = [], []
        for r in runs:
            d = r.get("per_n_probe", {}).get(best_n, {})
            if "concept_scores" in d:
                cs_all.append(np.asarray(d["concept_scores"]))
            if "random_scores" in d:
                rs_all.append(np.asarray(d["random_scores"]))
        if cs_all and rs_all:
            cs_all = np.concatenate(cs_all)
            rs_all = np.concatenate(rs_all)
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
            plt.axvline(cs_all.mean(), color="tab:blue", linestyle="--")
            plt.axvline(rs_all.mean(), color="tab:orange", linestyle="--")
            plt.xlabel("Mean pairwise CLIP cosine similarity")
            plt.ylabel("Number of channels (pooled across runs)")
            plt.title(
                f"ImageNet-val ResNet-50: Pooled Per-Channel Scores (N_PROBE={best_n})\nAggregated over {n_runs} runs, dashed lines = means"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, "imagenet_val_resnet50_agg_score_hist_best.png"
                ),
                dpi=140,
            )
            plt.close()
    except Exception as e:
        print(f"Error creating plot3: {e}")
        plt.close()

    # Plot 4: aggregated boxplot per N_PROBE (pooled per-channel concept scores)
    try:
        data = []
        for n in n_probe_values:
            pooled = []
            for r in runs:
                d = r.get("per_n_probe", {}).get(n, {})
                if "concept_scores" in d:
                    pooled.append(np.asarray(d["concept_scores"]))
            if pooled:
                data.append(np.concatenate(pooled))
            else:
                data.append(np.array([]))
        plt.figure(figsize=(6, 4))
        plt.boxplot(data, labels=[str(n) for n in n_probe_values], showmeans=True)
        # overlay mean line
        means = [d.mean() if d.size else np.nan for d in data]
        plt.plot(np.arange(1, len(means) + 1), means, "r-o", label="Pooled mean")
        plt.xlabel("N_PROBE")
        plt.ylabel("Per-channel concept score (pooled)")
        plt.title(
            f"ImageNet-val ResNet-50: Pooled Per-Channel Concept Score Distribution\nBoxplots across N_PROBE sweep (aggregated over {n_runs} runs)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "imagenet_val_resnet50_agg_boxplot_nprobe.png"),
            dpi=140,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot4: {e}")
        plt.close()

    # Print aggregated metrics
    try:
        print(f"Aggregated over {n_runs} run(s):")
        for j, n in enumerate(n_probe_values):
            print(
                f"N_PROBE={n}: concept_mean={c_mean[j]:.4f}±{c_sem[j]:.4f}, "
                f"random_mean={r_mean[j]:.4f}±{r_sem[j]:.4f}, "
                f"delta={d_mean[j]:.4f}±{d_sem[j]:.4f}"
            )
        print(f"Best N_PROBE (by aggregated concept mean): {best_n}")
    except Exception as e:
        print(f"Error printing metrics: {e}")
