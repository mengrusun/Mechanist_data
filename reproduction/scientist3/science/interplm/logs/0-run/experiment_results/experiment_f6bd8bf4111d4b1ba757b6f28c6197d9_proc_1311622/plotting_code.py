import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

ds_key = "swissprot_esm2_8m"
data = experiment_data.get(ds_key, {})
per_concept = data.get("per_concept_f1", {})
concept_counts = data.get("concept_counts", {})

F1_THRESH = 0.5

# Plot 1: F1 histogram
try:
    f1_neurons = [v["f1_neuron"] for v in per_concept.values()]
    f1_saes = [v["f1_sae"] for v in per_concept.values()]
    plt.figure(figsize=(8, 5))
    plt.hist(f1_neurons, bins=20, alpha=0.6, label="Raw neurons")
    plt.hist(f1_saes, bins=20, alpha=0.6, label="SAE features")
    plt.axvline(F1_THRESH, color="r", linestyle="--", label=f"F1={F1_THRESH}")
    plt.xlabel("Best F1 per concept")
    plt.ylabel("# concepts")
    plt.title(f"Swiss-Prot ESM-2-8M: F1 Distribution\nRaw Neurons vs SAE Features")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_f1_histogram.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Scatter neuron vs SAE F1
try:
    f1_neurons = np.array([v["f1_neuron"] for v in per_concept.values()])
    f1_saes = np.array([v["f1_sae"] for v in per_concept.values()])
    plt.figure(figsize=(6, 6))
    plt.scatter(f1_neurons, f1_saes, alpha=0.6)
    lim = max(
        f1_neurons.max() if len(f1_neurons) else 1,
        f1_saes.max() if len(f1_saes) else 1,
        0.1,
    )
    plt.plot([0, lim], [0, lim], "k--", alpha=0.5)
    plt.axhline(F1_THRESH, color="r", linestyle=":", alpha=0.5)
    plt.axvline(F1_THRESH, color="r", linestyle=":", alpha=0.5)
    plt.xlabel("Best F1 (Raw Neurons)")
    plt.ylabel("Best F1 (SAE Features)")
    plt.title(
        "Swiss-Prot ESM-2-8M: Per-Concept F1 Comparison\nRaw Neurons vs SAE Features"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_f1_scatter.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Bar chart of concept alignment counts
try:
    labels = ["Raw Neurons", "SAE Features"]
    counts = [concept_counts.get("neuron", 0), concept_counts.get("sae", 0)]
    total = concept_counts.get("total_evaluated", 0)
    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, counts, color=["tab:blue", "tab:orange"])
    for b, c in zip(bars, counts):
        plt.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            str(c),
            ha="center",
            va="bottom",
        )
    plt.ylabel(f"# concepts with F1 >= {F1_THRESH}")
    plt.title(
        f"Swiss-Prot ESM-2-8M: Concept Alignment Counts\nTotal concepts evaluated: {total}"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_alignment_counts.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Top concepts by SAE F1
try:
    items = sorted(per_concept.items(), key=lambda x: -x[1]["f1_sae"])[:15]
    names = [k[:40] for k, _ in items]
    sae_vals = [v["f1_sae"] for _, v in items]
    neu_vals = [v["f1_neuron"] for _, v in items]
    y = np.arange(len(names))
    plt.figure(figsize=(9, 7))
    plt.barh(y - 0.2, sae_vals, height=0.4, label="SAE")
    plt.barh(y + 0.2, neu_vals, height=0.4, label="Neuron")
    plt.yticks(y, names, fontsize=8)
    plt.xlabel("Best F1")
    plt.title("Swiss-Prot ESM-2-8M: Top 15 Concepts by SAE F1\nSAE vs Raw Neuron")
    plt.legend()
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_top_concepts.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: F1 vs concept size
try:
    n_pos = np.array([v["n_pos"] for v in per_concept.values()])
    f1_saes = np.array([v["f1_sae"] for v in per_concept.values()])
    f1_neurons = np.array([v["f1_neuron"] for v in per_concept.values()])
    plt.figure(figsize=(8, 5))
    plt.scatter(n_pos, f1_saes, alpha=0.6, label="SAE", color="tab:orange")
    plt.scatter(n_pos, f1_neurons, alpha=0.6, label="Neuron", color="tab:blue")
    plt.xscale("log")
    plt.xlabel("# positive residues per concept (log)")
    plt.ylabel("Best F1")
    plt.title("Swiss-Prot ESM-2-8M: F1 vs Concept Size\nSAE Features vs Raw Neurons")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "swissprot_esm2_8m_f1_vs_size.png"))
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

print("Done plotting.")
