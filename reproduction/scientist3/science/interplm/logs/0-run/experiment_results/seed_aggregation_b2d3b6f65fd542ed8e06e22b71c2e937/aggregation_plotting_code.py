import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_02-20-28_interplm_attempt_1/logs/0-run/experiment_results/experiment_f6bd8bf4111d4b1ba757b6f28c6197d9_proc_1311622/experiment_data.npy"
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

# Collect per_concept across runs
runs_per_concept = []
runs_concept_counts = []
for exp in all_experiment_data:
    data = exp.get(ds_key, {})
    runs_per_concept.append(data.get("per_concept_f1", {}))
    runs_concept_counts.append(data.get("concept_counts", {}))

n_runs = len(all_experiment_data)
print(f"Loaded {n_runs} runs")


def sem(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < 2:
        return 0.0
    return arr.std(ddof=1) / np.sqrt(len(arr))


# Plot 1: Aggregated F1 histograms with mean lines
try:
    all_f1_neurons = []
    all_f1_saes = []
    for pc in runs_per_concept:
        all_f1_neurons.extend([v["f1_neuron"] for v in pc.values()])
        all_f1_saes.extend([v["f1_sae"] for v in pc.values()])
    all_f1_neurons = np.array(all_f1_neurons)
    all_f1_saes = np.array(all_f1_saes)

    plt.figure(figsize=(8, 5))
    plt.hist(
        all_f1_neurons,
        bins=20,
        alpha=0.5,
        label=f"Raw neurons (mean={all_f1_neurons.mean():.3f})",
    )
    plt.hist(
        all_f1_saes,
        bins=20,
        alpha=0.5,
        label=f"SAE features (mean={all_f1_saes.mean():.3f})",
    )
    plt.axvline(all_f1_neurons.mean(), color="tab:blue", linestyle="-", alpha=0.8)
    plt.axvline(all_f1_saes.mean(), color="tab:orange", linestyle="-", alpha=0.8)
    plt.axvline(F1_THRESH, color="r", linestyle="--", label=f"F1={F1_THRESH}")
    plt.xlabel("Best F1 per concept")
    plt.ylabel("# concepts (aggregated across runs)")
    plt.title(
        f"Swiss-Prot ESM-2-8M: Aggregated F1 Distribution ({n_runs} runs)\nRaw Neurons vs SAE Features"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_f1_histogram_agg.png"))
    plt.close()
    print(f"Mean F1 neurons: {all_f1_neurons.mean():.4f} +/- {sem(all_f1_neurons):.4f}")
    print(f"Mean F1 SAE: {all_f1_saes.mean():.4f} +/- {sem(all_f1_saes):.4f}")
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Scatter per-concept mean F1 with error bars (concepts across runs)
try:
    # Aggregate by concept key
    concept_neuron = {}
    concept_sae = {}
    for pc in runs_per_concept:
        for k, v in pc.items():
            concept_neuron.setdefault(k, []).append(v["f1_neuron"])
            concept_sae.setdefault(k, []).append(v["f1_sae"])
    keys = sorted(set(concept_neuron.keys()) & set(concept_sae.keys()))
    mn = np.array([np.mean(concept_neuron[k]) for k in keys])
    ms = np.array([np.mean(concept_sae[k]) for k in keys])
    en = np.array([sem(concept_neuron[k]) for k in keys])
    es = np.array([sem(concept_sae[k]) for k in keys])

    plt.figure(figsize=(6, 6))
    plt.errorbar(
        mn,
        ms,
        xerr=en,
        yerr=es,
        fmt="o",
        alpha=0.5,
        ecolor="gray",
        label=f"Concepts (n={len(keys)}, err=SEM across {n_runs} runs)",
    )
    lim = max(mn.max() if len(mn) else 1, ms.max() if len(ms) else 1, 0.1)
    plt.plot([0, lim], [0, lim], "k--", alpha=0.5, label="y=x")
    plt.axhline(F1_THRESH, color="r", linestyle=":", alpha=0.5)
    plt.axvline(F1_THRESH, color="r", linestyle=":", alpha=0.5)
    plt.xlabel("Mean Best F1 (Raw Neurons)")
    plt.ylabel("Mean Best F1 (SAE Features)")
    plt.title(
        "Swiss-Prot ESM-2-8M: Per-Concept F1 (Mean ± SEM)\nRaw Neurons vs SAE Features"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_f1_scatter_agg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Bar chart of concept alignment counts with SEM error bars
try:
    neuron_counts = [cc.get("neuron", 0) for cc in runs_concept_counts]
    sae_counts = [cc.get("sae", 0) for cc in runs_concept_counts]
    totals = [cc.get("total_evaluated", 0) for cc in runs_concept_counts]

    means = [
        np.mean(neuron_counts) if neuron_counts else 0,
        np.mean(sae_counts) if sae_counts else 0,
    ]
    errs = [sem(neuron_counts), sem(sae_counts)]
    labels = ["Raw Neurons", "SAE Features"]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(
        labels,
        means,
        yerr=errs,
        capsize=8,
        color=["tab:blue", "tab:orange"],
        label=f"Mean ± SEM ({n_runs} runs)",
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
        f"Swiss-Prot ESM-2-8M: Concept Alignment Counts (Mean ± SEM, {n_runs} runs)\nMean total concepts evaluated: {total_mean:.1f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_alignment_counts_agg.png"))
    plt.close()
    print(f"Neuron aligned: {means[0]:.2f} +/- {errs[0]:.2f}")
    print(f"SAE aligned: {means[1]:.2f} +/- {errs[1]:.2f}")
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Top concepts by mean SAE F1 with error bars
try:
    concept_neuron = {}
    concept_sae = {}
    for pc in runs_per_concept:
        for k, v in pc.items():
            concept_neuron.setdefault(k, []).append(v["f1_neuron"])
            concept_sae.setdefault(k, []).append(v["f1_sae"])
    keys = list(set(concept_neuron.keys()) & set(concept_sae.keys()))
    key_stats = [
        (
            k,
            np.mean(concept_sae[k]),
            sem(concept_sae[k]),
            np.mean(concept_neuron[k]),
            sem(concept_neuron[k]),
        )
        for k in keys
    ]
    key_stats.sort(key=lambda x: -x[1])
    top = key_stats[:15]
    names = [k[:40] for k, *_ in top]
    sae_m = [x[1] for x in top]
    sae_e = [x[2] for x in top]
    neu_m = [x[3] for x in top]
    neu_e = [x[4] for x in top]
    y = np.arange(len(names))

    plt.figure(figsize=(9, 7))
    plt.barh(
        y - 0.2, sae_m, xerr=sae_e, height=0.4, label="SAE (mean ± SEM)", capsize=3
    )
    plt.barh(
        y + 0.2, neu_m, xerr=neu_e, height=0.4, label="Neuron (mean ± SEM)", capsize=3
    )
    plt.yticks(y, names, fontsize=8)
    plt.xlabel("Best F1")
    plt.title(
        f"Swiss-Prot ESM-2-8M: Top 15 Concepts by Mean SAE F1 ({n_runs} runs)\nSAE vs Raw Neuron"
    )
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_top_concepts_agg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: F1 vs concept size, binned means with SEM
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

    plt.figure(figsize=(8, 5))
    plt.scatter(
        all_npos, all_sae, alpha=0.3, label="SAE (individual)", color="tab:orange", s=15
    )
    plt.scatter(
        all_npos,
        all_neu,
        alpha=0.3,
        label="Neuron (individual)",
        color="tab:blue",
        s=15,
    )

    # Log-spaced bins with means
    if len(all_npos) > 0 and all_npos.max() > 0:
        bins = np.logspace(
            np.log10(max(all_npos.min(), 1)), np.log10(all_npos.max()), 8
        )
        bin_centers = []
        sae_means, sae_errs = [], []
        neu_means, neu_errs = [], []
        for i in range(len(bins) - 1):
            mask = (all_npos >= bins[i]) & (all_npos < bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append(np.sqrt(bins[i] * bins[i + 1]))
                sae_means.append(all_sae[mask].mean())
                sae_errs.append(sem(all_sae[mask]))
                neu_means.append(all_neu[mask].mean())
                neu_errs.append(sem(all_neu[mask]))
        plt.errorbar(
            bin_centers,
            sae_means,
            yerr=sae_errs,
            fmt="o-",
            color="darkorange",
            label="SAE bin mean ± SEM",
            capsize=4,
            linewidth=2,
        )
        plt.errorbar(
            bin_centers,
            neu_means,
            yerr=neu_errs,
            fmt="s-",
            color="darkblue",
            label="Neuron bin mean ± SEM",
            capsize=4,
            linewidth=2,
        )

    plt.xscale("log")
    plt.xlabel("# positive residues per concept (log)")
    plt.ylabel("Best F1")
    plt.title(
        f"Swiss-Prot ESM-2-8M: F1 vs Concept Size (Aggregated, {n_runs} runs)\nSAE Features vs Raw Neurons"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_f1_vs_size_agg.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

print("Done plotting.")
