import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_02-20-28_interplm_attempt_1/logs/0-run/experiment_results/experiment_e7e6c90713a44b17b9a5c55dcba6a77b_proc_395414/experiment_data.npy",
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

ds_key = "swissprot_esm2_8m"
F1_THRESH = 0.5

# Collect per-run data
runs_per_concept = []
runs_concept_counts = []
for ed in all_experiment_data:
    data = ed.get(ds_key, {})
    runs_per_concept.append(data.get("per_concept_f1", {}))
    runs_concept_counts.append(data.get("concept_counts", {}))

n_runs = len(all_experiment_data)
print(f"Number of runs: {n_runs}")


def sem(a):
    a = np.asarray(a, dtype=float)
    if len(a) < 2:
        return 0.0
    return a.std(ddof=1) / np.sqrt(len(a))


# Plot 1: F1 histogram with mean ± SE across runs per bin
try:
    bins = np.linspace(0, 1, 21)
    neu_counts_runs = []
    sae_counts_runs = []
    for pc in runs_per_concept:
        f1n = [v["f1_neuron"] for v in pc.values()]
        f1s = [v["f1_sae"] for v in pc.values()]
        nc, _ = np.histogram(f1n, bins=bins)
        sc, _ = np.histogram(f1s, bins=bins)
        neu_counts_runs.append(nc)
        sae_counts_runs.append(sc)
    neu_counts_runs = np.array(neu_counts_runs)
    sae_counts_runs = np.array(sae_counts_runs)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = (bins[1] - bins[0]) * 0.4

    neu_mean = neu_counts_runs.mean(axis=0)
    sae_mean = sae_counts_runs.mean(axis=0)
    neu_se = (
        neu_counts_runs.std(axis=0, ddof=1) / np.sqrt(n_runs)
        if n_runs > 1
        else np.zeros_like(neu_mean)
    )
    sae_se = (
        sae_counts_runs.std(axis=0, ddof=1) / np.sqrt(n_runs)
        if n_runs > 1
        else np.zeros_like(sae_mean)
    )

    plt.figure(figsize=(9, 5))
    plt.bar(
        centers - width / 2,
        neu_mean,
        width=width,
        yerr=neu_se,
        alpha=0.7,
        label="Raw neurons (mean±SE)",
        capsize=2,
        color="tab:blue",
    )
    plt.bar(
        centers + width / 2,
        sae_mean,
        width=width,
        yerr=sae_se,
        alpha=0.7,
        label="SAE features (mean±SE)",
        capsize=2,
        color="tab:orange",
    )
    plt.axvline(F1_THRESH, color="r", linestyle="--", label=f"F1={F1_THRESH}")
    plt.xlabel("Best F1 per concept")
    plt.ylabel("# concepts (mean over runs)")
    plt.title(
        f"Swiss-Prot ESM-2-8M: Aggregated F1 Distribution across {n_runs} runs\nRaw Neurons vs SAE Features"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_f1_histogram_agg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Scatter per-concept mean F1 with error bars
try:
    # Aggregate per concept across runs
    concept_ids = set()
    for pc in runs_per_concept:
        concept_ids.update(pc.keys())
    neu_by_c = {c: [] for c in concept_ids}
    sae_by_c = {c: [] for c in concept_ids}
    npos_by_c = {c: [] for c in concept_ids}
    for pc in runs_per_concept:
        for c, v in pc.items():
            neu_by_c[c].append(v["f1_neuron"])
            sae_by_c[c].append(v["f1_sae"])
            npos_by_c[c].append(v.get("n_pos", np.nan))

    neu_mean = np.array([np.mean(neu_by_c[c]) for c in concept_ids])
    sae_mean = np.array([np.mean(sae_by_c[c]) for c in concept_ids])
    neu_err = np.array([sem(neu_by_c[c]) for c in concept_ids])
    sae_err = np.array([sem(sae_by_c[c]) for c in concept_ids])

    plt.figure(figsize=(7, 7))
    plt.errorbar(
        neu_mean,
        sae_mean,
        xerr=neu_err,
        yerr=sae_err,
        fmt="o",
        alpha=0.5,
        ecolor="gray",
        label="Concepts (mean±SE)",
    )
    lim = max(
        neu_mean.max() if len(neu_mean) else 1,
        sae_mean.max() if len(sae_mean) else 1,
        0.1,
    )
    plt.plot([0, lim], [0, lim], "k--", alpha=0.5, label="y = x")
    plt.axhline(F1_THRESH, color="r", linestyle=":", alpha=0.5)
    plt.axvline(F1_THRESH, color="r", linestyle=":", alpha=0.5)
    plt.xlabel("Mean Best F1 (Raw Neurons)")
    plt.ylabel("Mean Best F1 (SAE Features)")
    plt.title(
        f"Swiss-Prot ESM-2-8M: Per-Concept F1 Comparison\nAggregated over {n_runs} runs (mean±SE)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_f1_scatter_agg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Bar chart of concept alignment counts with error bars
try:
    neu_counts = [cc.get("neuron", 0) for cc in runs_concept_counts]
    sae_counts = [cc.get("sae", 0) for cc in runs_concept_counts]
    totals = [cc.get("total_evaluated", 0) for cc in runs_concept_counts]
    means = [np.mean(neu_counts), np.mean(sae_counts)]
    errs = [sem(neu_counts), sem(sae_counts)]
    labels = ["Raw Neurons", "SAE Features"]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(
        labels,
        means,
        yerr=errs,
        capsize=5,
        color=["tab:blue", "tab:orange"],
        label="Mean ± SE",
    )
    for b, m in zip(bars, means):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{m:.1f}",
            ha="center",
            va="bottom",
        )
    plt.ylabel(f"# concepts with F1 >= {F1_THRESH}")
    plt.title(
        f"Swiss-Prot ESM-2-8M: Concept Alignment Counts\nAggregated over {n_runs} runs (total evaluated≈{np.mean(totals):.0f})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_alignment_counts_agg.png"))
    plt.close()

    print(
        f"Neuron counts across runs: {neu_counts} -> mean={np.mean(neu_counts):.2f} SE={sem(neu_counts):.2f}"
    )
    print(
        f"SAE counts across runs: {sae_counts} -> mean={np.mean(sae_counts):.2f} SE={sem(sae_counts):.2f}"
    )
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Top concepts by mean SAE F1 with error bars
try:
    concept_ids_l = list(concept_ids)
    sae_mean_map = {c: np.mean(sae_by_c[c]) for c in concept_ids_l}
    top = sorted(concept_ids_l, key=lambda c: -sae_mean_map[c])[:15]
    sae_vals = [np.mean(sae_by_c[c]) for c in top]
    neu_vals = [np.mean(neu_by_c[c]) for c in top]
    sae_errs = [sem(sae_by_c[c]) for c in top]
    neu_errs = [sem(neu_by_c[c]) for c in top]
    names = [str(c)[:40] for c in top]
    y = np.arange(len(names))

    plt.figure(figsize=(9, 7))
    plt.barh(
        y - 0.2,
        sae_vals,
        xerr=sae_errs,
        height=0.4,
        label="SAE (mean±SE)",
        capsize=3,
        color="tab:orange",
    )
    plt.barh(
        y + 0.2,
        neu_vals,
        xerr=neu_errs,
        height=0.4,
        label="Neuron (mean±SE)",
        capsize=3,
        color="tab:blue",
    )
    plt.yticks(y, names, fontsize=8)
    plt.xlabel("Best F1 (mean over runs)")
    plt.title(
        f"Swiss-Prot ESM-2-8M: Top 15 Concepts by SAE F1\nAggregated over {n_runs} runs"
    )
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_top_concepts_agg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: F1 vs concept size (mean±SE per concept)
try:
    n_pos_mean = np.array([np.nanmean(npos_by_c[c]) for c in concept_ids_l])
    sae_m = np.array([np.mean(sae_by_c[c]) for c in concept_ids_l])
    neu_m = np.array([np.mean(neu_by_c[c]) for c in concept_ids_l])
    sae_e = np.array([sem(sae_by_c[c]) for c in concept_ids_l])
    neu_e = np.array([sem(neu_by_c[c]) for c in concept_ids_l])

    plt.figure(figsize=(9, 5))
    plt.errorbar(
        n_pos_mean,
        sae_m,
        yerr=sae_e,
        fmt="o",
        alpha=0.6,
        label="SAE (mean±SE)",
        color="tab:orange",
    )
    plt.errorbar(
        n_pos_mean,
        neu_m,
        yerr=neu_e,
        fmt="o",
        alpha=0.6,
        label="Neuron (mean±SE)",
        color="tab:blue",
    )
    plt.xscale("log")
    plt.xlabel("# positive residues per concept (log)")
    plt.ylabel("Best F1 (mean over runs)")
    plt.title(f"Swiss-Prot ESM-2-8M: F1 vs Concept Size\nAggregated over {n_runs} runs")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_f1_vs_size_agg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

print("Done plotting.")
