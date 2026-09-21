import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_02-20-28_interplm_attempt_1/logs/0-run/experiment_results/experiment_dfc3bce9566d4aa9b352f5561c235210_proc_795126/experiment_data.npy"
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

ds_key = "swissprot_esm2_8m"
F1_THRESH = 0.5

# Collect per-concept data across runs
runs_per_concept = []
runs_concept_counts = []
for ed in all_experiment_data:
    data = ed.get(ds_key, {})
    runs_per_concept.append(data.get("per_concept_f1", {}))
    runs_concept_counts.append(data.get("concept_counts", {}))

n_runs = len(runs_per_concept)
print(f"Number of runs loaded: {n_runs}")

# Plot 1: Aggregated F1 histogram (mean±SEM of bin counts across runs)
try:
    bins = np.linspace(0, 1, 21)
    neuron_hists = []
    sae_hists = []
    for pc in runs_per_concept:
        f1n = [v["f1_neuron"] for v in pc.values()]
        f1s = [v["f1_sae"] for v in pc.values()]
        hn, _ = np.histogram(f1n, bins=bins)
        hs, _ = np.histogram(f1s, bins=bins)
        neuron_hists.append(hn)
        sae_hists.append(hs)
    neuron_hists = np.array(neuron_hists)
    sae_hists = np.array(sae_hists)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = (bins[1] - bins[0]) * 0.4

    mean_n = neuron_hists.mean(axis=0)
    sem_n = neuron_hists.std(axis=0, ddof=0) / np.sqrt(max(n_runs, 1))
    mean_s = sae_hists.mean(axis=0)
    sem_s = sae_hists.std(axis=0, ddof=0) / np.sqrt(max(n_runs, 1))

    plt.figure(figsize=(9, 5))
    plt.bar(
        centers - width / 2,
        mean_n,
        width=width,
        yerr=sem_n,
        alpha=0.7,
        label=f"Raw neurons (mean±SEM, n={n_runs})",
        capsize=3,
        color="tab:blue",
    )
    plt.bar(
        centers + width / 2,
        mean_s,
        width=width,
        yerr=sem_s,
        alpha=0.7,
        label=f"SAE features (mean±SEM, n={n_runs})",
        capsize=3,
        color="tab:orange",
    )
    plt.axvline(F1_THRESH, color="r", linestyle="--", label=f"F1={F1_THRESH}")
    plt.xlabel("Best F1 per concept")
    plt.ylabel("# concepts (mean across runs)")
    plt.title(
        "Swiss-Prot ESM-2-8M: Aggregated F1 Distribution\nRaw Neurons vs SAE Features (mean ± SEM across runs)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_agg_f1_histogram.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Mean F1 per concept across runs, with error bars (scatter)
try:
    # Union of concepts
    concept_keys = set()
    for pc in runs_per_concept:
        concept_keys.update(pc.keys())
    concept_keys = sorted(concept_keys)

    neuron_means, neuron_sems = [], []
    sae_means, sae_sems = [], []
    for c in concept_keys:
        nvals = [pc[c]["f1_neuron"] for pc in runs_per_concept if c in pc]
        svals = [pc[c]["f1_sae"] for pc in runs_per_concept if c in pc]
        if len(nvals) == 0 or len(svals) == 0:
            continue
        neuron_means.append(np.mean(nvals))
        neuron_sems.append(np.std(nvals, ddof=0) / np.sqrt(len(nvals)))
        sae_means.append(np.mean(svals))
        sae_sems.append(np.std(svals, ddof=0) / np.sqrt(len(svals)))

    neuron_means = np.array(neuron_means)
    sae_means = np.array(sae_means)
    neuron_sems = np.array(neuron_sems)
    sae_sems = np.array(sae_sems)

    plt.figure(figsize=(7, 7))
    plt.errorbar(
        neuron_means,
        sae_means,
        xerr=neuron_sems,
        yerr=sae_sems,
        fmt="o",
        alpha=0.6,
        ecolor="gray",
        capsize=2,
        label=f"Concepts (mean±SEM across {n_runs} runs)",
    )
    lim = max(
        neuron_means.max() if len(neuron_means) else 1,
        sae_means.max() if len(sae_means) else 1,
        0.1,
    )
    plt.plot([0, lim], [0, lim], "k--", alpha=0.5, label="y=x")
    plt.axhline(F1_THRESH, color="r", linestyle=":", alpha=0.5)
    plt.axvline(F1_THRESH, color="r", linestyle=":", alpha=0.5)
    plt.xlabel("Mean Best F1 (Raw Neurons)")
    plt.ylabel("Mean Best F1 (SAE Features)")
    plt.title(
        "Swiss-Prot ESM-2-8M: Per-Concept F1 Comparison\nAggregated across runs (mean ± SEM)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_agg_f1_scatter.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Aggregated alignment counts bar chart
try:
    neuron_counts = [cc.get("neuron", 0) for cc in runs_concept_counts]
    sae_counts = [cc.get("sae", 0) for cc in runs_concept_counts]
    totals = [cc.get("total_evaluated", 0) for cc in runs_concept_counts]

    labels = ["Raw Neurons", "SAE Features"]
    means = [
        np.mean(neuron_counts) if neuron_counts else 0,
        np.mean(sae_counts) if sae_counts else 0,
    ]
    sems = [
        (
            np.std(neuron_counts, ddof=0) / np.sqrt(max(len(neuron_counts), 1))
            if neuron_counts
            else 0
        ),
        (
            np.std(sae_counts, ddof=0) / np.sqrt(max(len(sae_counts), 1))
            if sae_counts
            else 0
        ),
    ]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(
        labels,
        means,
        yerr=sems,
        color=["tab:blue", "tab:orange"],
        capsize=5,
        label=f"Mean ± SEM (n={n_runs} runs)",
    )
    for b, m in zip(bars, means):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{m:.1f}",
            ha="center",
            va="bottom",
        )
    total_mean = np.mean(totals) if totals else 0
    plt.ylabel(f"# concepts with F1 >= {F1_THRESH}")
    plt.title(
        f"Swiss-Prot ESM-2-8M: Aggregated Concept Alignment Counts\nMean total evaluated: {total_mean:.1f} (n={n_runs} runs)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_agg_alignment_counts.png"))
    plt.close()

    print(
        f"Neuron aligned counts per run: {neuron_counts}, mean={means[0]:.2f}±{sems[0]:.2f}"
    )
    print(
        f"SAE aligned counts per run:    {sae_counts}, mean={means[1]:.2f}±{sems[1]:.2f}"
    )
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Top concepts by mean SAE F1 with error bars
try:
    concept_stats = []
    concept_keys = set()
    for pc in runs_per_concept:
        concept_keys.update(pc.keys())
    for c in sorted(concept_keys):
        nvals = [pc[c]["f1_neuron"] for pc in runs_per_concept if c in pc]
        svals = [pc[c]["f1_sae"] for pc in runs_per_concept if c in pc]
        if not nvals or not svals:
            continue
        concept_stats.append(
            (
                c,
                np.mean(svals),
                np.std(svals, ddof=0) / np.sqrt(len(svals)),
                np.mean(nvals),
                np.std(nvals, ddof=0) / np.sqrt(len(nvals)),
            )
        )
    concept_stats.sort(key=lambda x: -x[1])
    top = concept_stats[:15]
    names = [t[0][:40] for t in top]
    sae_m = [t[1] for t in top]
    sae_e = [t[2] for t in top]
    neu_m = [t[3] for t in top]
    neu_e = [t[4] for t in top]
    y = np.arange(len(names))

    plt.figure(figsize=(10, 7))
    plt.barh(
        y - 0.2,
        sae_m,
        xerr=sae_e,
        height=0.4,
        label=f"SAE (mean±SEM, n={n_runs})",
        capsize=3,
        color="tab:orange",
    )
    plt.barh(
        y + 0.2,
        neu_m,
        xerr=neu_e,
        height=0.4,
        label=f"Neuron (mean±SEM, n={n_runs})",
        capsize=3,
        color="tab:blue",
    )
    plt.yticks(y, names, fontsize=8)
    plt.xlabel("Best F1")
    plt.title(
        "Swiss-Prot ESM-2-8M: Top 15 Concepts by Mean SAE F1\nSAE vs Raw Neuron (aggregated across runs)"
    )
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_agg_top_concepts.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: F1 vs concept size (binned mean ± SEM)
try:
    all_npos = []
    all_sae = []
    all_neu = []
    for pc in runs_per_concept:
        for v in pc.values():
            all_npos.append(v["n_pos"])
            all_sae.append(v["f1_sae"])
            all_neu.append(v["f1_neuron"])
    all_npos = np.array(all_npos)
    all_sae = np.array(all_sae)
    all_neu = np.array(all_neu)

    if len(all_npos) > 0:
        # log-spaced bins
        lo = max(all_npos.min(), 1)
        hi = all_npos.max()
        bins = np.logspace(np.log10(lo), np.log10(hi + 1), 8)
        centers = np.sqrt(bins[:-1] * bins[1:])

        sae_means, sae_sems = [], []
        neu_means, neu_sems = [], []
        for i in range(len(bins) - 1):
            mask = (all_npos >= bins[i]) & (all_npos < bins[i + 1])
            if mask.sum() > 0:
                sae_means.append(all_sae[mask].mean())
                sae_sems.append(all_sae[mask].std(ddof=0) / np.sqrt(mask.sum()))
                neu_means.append(all_neu[mask].mean())
                neu_sems.append(all_neu[mask].std(ddof=0) / np.sqrt(mask.sum()))
            else:
                sae_means.append(np.nan)
                sae_sems.append(0)
                neu_means.append(np.nan)
                neu_sems.append(0)

        plt.figure(figsize=(8, 5))
        plt.errorbar(
            centers,
            sae_means,
            yerr=sae_sems,
            marker="o",
            label="SAE (mean±SEM)",
            color="tab:orange",
            capsize=3,
        )
        plt.errorbar(
            centers,
            neu_means,
            yerr=neu_sems,
            marker="s",
            label="Neuron (mean±SEM)",
            color="tab:blue",
            capsize=3,
        )
        plt.xscale("log")
        plt.xlabel("# positive residues per concept (log, binned)")
        plt.ylabel("Best F1")
        plt.title(
            "Swiss-Prot ESM-2-8M: F1 vs Concept Size\nBinned mean ± SEM (aggregated across runs)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_agg_f1_vs_size.png"))
        plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

print("Done plotting.")
