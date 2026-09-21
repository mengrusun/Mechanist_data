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

ed = experiment_data.get("steering_vector_construction", {}).get(
    "r1_distill_llama_8b_logreg", {}
)
dose_response = ed.get("dose_response", {})
bse_scores = ed.get("bse_scores", {})
layer_ablation = ed.get("layer_ablation", {}).get("hedging", {})
interference = ed.get("cross_interference", {})
baseline = ed.get("baseline", {})
baseline_acc = baseline.get("accuracy", 0.0)

BEHAVIOURS = ["hedging", "backtracking", "self_correction", "example_gen"]

# Plot 1: Dose-response curves
try:
    plt.figure(figsize=(8, 5))
    for b, r in dose_response.items():
        cs = sorted([float(k) for k in r.keys()])
        ys = [r[str(c)]["target_mean"] for c in cs]
        plt.plot(cs, ys, "o-", label=b)
    plt.xlabel("coefficient (× BASE_SCALE)")
    plt.ylabel("Mean behaviour count in CoT")
    plt.title(
        "Dose-response curves (LogReg steering)\nDataset: R1-Distill-Llama-8B math tasks"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "r1_llama8b_dose_response_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating dose-response plot: {e}")
    plt.close()

# Plot 2: Accuracy vs coefficient
try:
    plt.figure(figsize=(8, 5))
    for b, r in dose_response.items():
        cs = sorted([float(k) for k in r.keys()])
        ys = [r[str(c)]["acc"] for c in cs]
        plt.plot(cs, ys, "s-", label=b)
    plt.axhline(baseline_acc, color="k", ls="--", label=f"baseline={baseline_acc:.2f}")
    plt.xlabel("coefficient")
    plt.ylabel("Task accuracy")
    plt.title(
        "Accuracy vs Steering Coefficient (LogReg)\nDataset: R1-Distill-Llama-8B math tasks"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "r1_llama8b_accuracy_vs_coefficient.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot 3: BSE scores bar chart
try:
    if bse_scores:
        plt.figure(figsize=(7, 4))
        bs = list(bse_scores.keys())
        ys_a1 = [bse_scores[b]["bse_a1"] for b in bs]
        ys_a2 = [
            (
                bse_scores[b].get("bse_a2")
                if bse_scores[b].get("bse_a2") is not None
                else 0
            )
            for b in bs
        ]
        x = np.arange(len(bs))
        w = 0.35
        plt.bar(x - w / 2, ys_a1, w, label="BSE(α=1)", color="darkorange")
        plt.bar(x + w / 2, ys_a2, w, label="BSE(α=2)", color="steelblue")
        plt.xticks(x, bs, rotation=15)
        plt.ylabel("BSE score")
        plt.title(
            "Behaviour Steering Efficacy (LogReg construction)\nDataset: R1-Distill-Llama-8B"
        )
        plt.axhline(0, color="k", lw=0.5)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "r1_llama8b_bse_scores.png"))
        plt.close()
except Exception as e:
    print(f"Error creating BSE plot: {e}")
    plt.close()

# Plot 4: Layer ablation for hedging
try:
    if layer_ablation:
        plt.figure(figsize=(7, 4))
        lis = sorted([int(k) for k in layer_ablation.keys()])
        deltas = [layer_ablation[str(li)]["delta"] for li in lis]
        bses = [layer_ablation[str(li)]["bse"] for li in lis]
        plt.plot(lis, deltas, "o-", label="Δ frequency (+coeff vs -coeff)")
        plt.plot(lis, bses, "s-", label="BSE")
        plt.xlabel("Layer index")
        plt.ylabel("Value")
        plt.title("Per-Layer Ablation: Hedging (LogReg)\nDataset: R1-Distill-Llama-8B")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "r1_llama8b_layer_ablation_hedging.png"))
        plt.close()
except Exception as e:
    print(f"Error creating layer ablation plot: {e}")
    plt.close()

# Plot 5: Cross-interference heatmap
try:
    if interference:
        src = list(interference.keys())
        tgt = BEHAVIOURS
        M = np.array([[interference[s].get(t, 0) for t in tgt] for s in src])
        plt.figure(figsize=(6, 5))
        vmax = max(np.max(np.abs(M)), 1e-6)
        im = plt.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.xticks(range(len(tgt)), tgt, rotation=30)
        plt.yticks(range(len(src)), src)
        plt.xlabel("Target behaviour (measured)")
        plt.ylabel("Source behaviour (steered)")
        plt.title("Cross-Behaviour Interference (LogReg)\nDataset: R1-Distill-Llama-8B")
        for i in range(len(src)):
            for j in range(len(tgt)):
                plt.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "r1_llama8b_cross_interference_heatmap.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating interference heatmap: {e}")
    plt.close()

# Plot 6: Baseline behaviour counts
try:
    counts = baseline.get("counts", {})
    if counts:
        plt.figure(figsize=(7, 4))
        bs = list(counts.keys())
        ys = [counts[b] for b in bs]
        plt.bar(bs, ys, color="seagreen")
        plt.ylabel("Total occurrences in baseline CoTs")
        plt.title(
            f"Baseline Behaviour Counts (acc={baseline_acc:.2f})\nDataset: R1-Distill-Llama-8B math tasks"
        )
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "r1_llama8b_baseline_behaviour_counts.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating baseline counts plot: {e}")
    plt.close()

# Print evaluation metrics
print(f"Baseline accuracy: {baseline_acc:.3f}")
for b in bse_scores:
    print(
        f"  [{b}] BSE(α=1)={bse_scores[b]['bse_a1']:.3f}, BSE(α=2)={bse_scores[b].get('bse_a2')}"
    )
