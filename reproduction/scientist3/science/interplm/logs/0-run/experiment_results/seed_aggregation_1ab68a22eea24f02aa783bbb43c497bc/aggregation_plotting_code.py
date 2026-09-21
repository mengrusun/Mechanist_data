import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_02-20-28_interplm_attempt_1/logs/0-run/experiment_results/experiment_6b67405e6ad443afbb04484cdf70c2a4_proc_1677598/experiment_data.npy",
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

ds_key = "swissprot_esm2_8m"
F1_THRESH = 0.5

# Collect per-run stats
runs_f1_neuron = []
runs_f1_sae = []
runs_n_pos = []
runs_counts_neuron = []
runs_counts_sae = []
runs_total = []
per_concept_agg = {}  # concept -> {"neuron": [...], "sae": [...], "n_pos": [...]}

for exp in all_experiment_data:
    data = exp.get(ds_key, {})
    per_concept = data.get("per_concept_f1", {})
    concept_counts = data.get("concept_counts", {})
    fn = np.array([v["f1_neuron"] for v in per_concept.values()])
    fs = np.array([v["f1_sae"] for v in per_concept.values()])
    npos = np.array([v["n_pos"] for v in per_concept.values()])
    runs_f1_neuron.append(fn)
    runs_f1_sae.append(fs)
    runs_n_pos.append(npos)
    runs_counts_neuron.append(concept_counts.get("neuron", 0))
    runs_counts_sae.append(concept_counts.get("sae", 0))
    runs_total.append(concept_counts.get("total_evaluated", 0))
    for k, v in per_concept.items():
        per_concept_agg.setdefault(k, {"neuron": [], "sae": [], "n_pos": []})
        per_concept_agg[k]["neuron"].append(v["f1_neuron"])
        per_concept_agg[k]["sae"].append(v["f1_sae"])
        per_concept_agg[k]["n_pos"].append(v["n_pos"])

n_runs = len(all_experiment_data)
print(f"Number of runs aggregated: {n_runs}")


def sem(x):
    x = np.asarray(x)
    if len(x) <= 1:
        return 0.0
    return np.std(x, ddof=1) / np.sqrt(len(x))


# Plot 1: Aggregated F1 histogram with mean +/- SEM lines
try:
    plt.figure(figsize=(8, 5))
    all_neuron = np.concatenate(runs_f1_neuron) if runs_f1_neuron else np.array([])
    all_sae = np.concatenate(runs_f1_sae) if runs_f1_sae else np.array([])
    bins = np.linspace(
        0,
        max(
            all_neuron.max() if len(all_neuron) else 1,
            all_sae.max() if len(all_sae) else 1,
            1.0,
        ),
        21,
    )
    # Per-run histograms -> mean and SEM per bin
    hist_neuron = (
        np.stack([np.histogram(r, bins=bins)[0] for r in runs_f1_neuron])
        if n_runs
        else np.zeros((1, len(bins) - 1))
    )
    hist_sae = (
        np.stack([np.histogram(r, bins=bins)[0] for r in runs_f1_sae])
        if n_runs
        else np.zeros((1, len(bins) - 1))
    )
    centers = (bins[:-1] + bins[1:]) / 2
    width = (bins[1] - bins[0]) * 0.4

    mn_n, se_n = hist_neuron.mean(0), np.array(
        [sem(hist_neuron[:, i]) for i in range(hist_neuron.shape[1])]
    )
    mn_s, se_s = hist_sae.mean(0), np.array(
        [sem(hist_sae[:, i]) for i in range(hist_sae.shape[1])]
    )

    plt.bar(
        centers - width / 2,
        mn_n,
        width=width,
        yerr=se_n,
        alpha=0.7,
        label=f"Raw neurons (mean ± SE, n={n_runs})",
        capsize=3,
    )
    plt.bar(
        centers + width / 2,
        mn_s,
        width=width,
        yerr=se_s,
        alpha=0.7,
        label=f"SAE features (mean ± SE, n={n_runs})",
        capsize=3,
    )
    plt.axvline(F1_THRESH, color="r", linestyle="--", label=f"F1={F1_THRESH}")
    plt.xlabel("Best F1 per concept")
    plt.ylabel("# concepts (mean across runs)")
    plt.title(
        "Swiss-Prot ESM-2-8M: Aggregated F1 Distribution\nRaw Neurons vs SAE Features (mean ± SE)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_agg_f1_histogram.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Aggregated alignment counts bar with SEM
try:
    labels = ["Raw Neurons", "SAE Features"]
    means = [
        np.mean(runs_counts_neuron) if runs_counts_neuron else 0,
        np.mean(runs_counts_sae) if runs_counts_sae else 0,
    ]
    ses = [sem(runs_counts_neuron), sem(runs_counts_sae)]
    total_mean = np.mean(runs_total) if runs_total else 0
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, means, yerr=ses, capsize=6, color=["tab:blue", "tab:orange"])
    for b, m, s in zip(bars, means, ses):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + s,
            f"{m:.1f}±{s:.1f}",
            ha="center",
            va="bottom",
        )
    plt.ylabel(f"# concepts with F1 ≥ {F1_THRESH} (mean ± SE)")
    plt.title(
        f"Swiss-Prot ESM-2-8M: Aggregated Concept Alignment Counts\nMean total evaluated: {total_mean:.1f}, n_runs={n_runs}"
    )
    plt.legend(["Mean ± SE"])
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_agg_alignment_counts.png"))
    plt.close()
    print(f"Neuron aligned: mean={means[0]:.2f} SE={ses[0]:.2f}")
    print(f"SAE aligned:    mean={means[1]:.2f} SE={ses[1]:.2f}")
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Per-concept mean F1 scatter (SAE vs Neuron) with error bars
try:
    concepts = list(per_concept_agg.keys())
    mn_neuron = np.array([np.mean(per_concept_agg[c]["neuron"]) for c in concepts])
    se_neuron = np.array([sem(per_concept_agg[c]["neuron"]) for c in concepts])
    mn_sae = np.array([np.mean(per_concept_agg[c]["sae"]) for c in concepts])
    se_sae = np.array([sem(per_concept_agg[c]["sae"]) for c in concepts])
    plt.figure(figsize=(6, 6))
    plt.errorbar(
        mn_neuron,
        mn_sae,
        xerr=se_neuron,
        yerr=se_sae,
        fmt="o",
        alpha=0.6,
        capsize=2,
        label=f"Concepts (mean ± SE, n_runs={n_runs})",
    )
    lim = max(
        mn_neuron.max() if len(mn_neuron) else 1,
        mn_sae.max() if len(mn_sae) else 1,
        0.1,
    )
    plt.plot([0, lim], [0, lim], "k--", alpha=0.5, label="y=x")
    plt.axhline(F1_THRESH, color="r", linestyle=":", alpha=0.5)
    plt.axvline(F1_THRESH, color="r", linestyle=":", alpha=0.5)
    plt.xlabel("Best F1 (Raw Neurons, mean)")
    plt.ylabel("Best F1 (SAE Features, mean)")
    plt.title(
        "Swiss-Prot ESM-2-8M: Per-Concept Mean F1 Comparison\nRaw Neurons vs SAE Features"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_agg_f1_scatter.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Top 15 concepts by mean SAE F1 with SE error bars
try:
    items = sorted(per_concept_agg.items(), key=lambda x: -np.mean(x[1]["sae"]))[:15]
    names = [k[:40] for k, _ in items]
    sae_m = [np.mean(v["sae"]) for _, v in items]
    sae_e = [sem(v["sae"]) for _, v in items]
    neu_m = [np.mean(v["neuron"]) for _, v in items]
    neu_e = [sem(v["neuron"]) for _, v in items]
    y = np.arange(len(names))
    plt.figure(figsize=(9, 7))
    plt.barh(y - 0.2, sae_m, xerr=sae_e, height=0.4, label="SAE (mean ± SE)", capsize=3)
    plt.barh(
        y + 0.2, neu_m, xerr=neu_e, height=0.4, label="Neuron (mean ± SE)", capsize=3
    )
    plt.yticks(y, names, fontsize=8)
    plt.xlabel("Best F1 (mean across runs)")
    plt.title(
        f"Swiss-Prot ESM-2-8M: Top 15 Concepts by Mean SAE F1\nSAE vs Raw Neuron (n_runs={n_runs})"
    )
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_agg_top_concepts.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Binned F1 vs concept size (mean ± SE)
try:
    all_npos = np.concatenate(runs_n_pos) if runs_n_pos else np.array([])
    all_sae = np.concatenate(runs_f1_sae) if runs_f1_sae else np.array([])
    all_neu = np.concatenate(runs_f1_neuron) if runs_f1_neuron else np.array([])
    if len(all_npos):
        bins = np.logspace(
            np.log10(max(all_npos.min(), 1)), np.log10(all_npos.max() + 1), 8
        )
        idx = np.digitize(all_npos, bins)
        centers, mean_s, se_s, mean_n, se_n = [], [], [], [], []
        for b in range(1, len(bins)):
            mask = idx == b
            if mask.sum() > 0:
                centers.append(np.sqrt(bins[b - 1] * bins[b]))
                mean_s.append(np.mean(all_sae[mask]))
                se_s.append(sem(all_sae[mask]))
                mean_n.append(np.mean(all_neu[mask]))
                se_n.append(sem(all_neu[mask]))
        plt.figure(figsize=(8, 5))
        plt.errorbar(
            centers, mean_s, yerr=se_s, marker="o", label="SAE (mean ± SE)", capsize=3
        )
        plt.errorbar(
            centers,
            mean_n,
            yerr=se_n,
            marker="s",
            label="Neuron (mean ± SE)",
            capsize=3,
        )
        plt.xscale("log")
        plt.xlabel("# positive residues per concept (log, binned)")
        plt.ylabel("Best F1 (mean per bin)")
        plt.title(
            "Swiss-Prot ESM-2-8M: F1 vs Concept Size (binned)\nSAE Features vs Raw Neurons"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_agg_f1_vs_size.png"))
        plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

print("Done plotting.")
