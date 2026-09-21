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

ds_key = "propositional_logic_circuit"
data = experiment_data.get(ds_key, {})

# Plot 1: Attention head patching recovery heatmap
try:
    head_effects = np.array(data.get("head_effects", []))
    plt.figure(figsize=(10, 6))
    vmax = np.abs(head_effects).max() if head_effects.size else 1
    im = plt.imshow(head_effects, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    plt.colorbar(im, label="Recovery")
    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(
        "Propositional Logic Dataset: Attention Head Patching Recovery\n(Clean-into-Corrupted Activation Patching)"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "propositional_logic_head_patching_heatmap.png"),
        dpi=100,
    )
    plt.close()
except Exception as e:
    print(f"Error creating head patching heatmap: {e}")
    plt.close()

# Plot 2: MLP patching recovery per layer
try:
    mlp_effects = np.array(data.get("mlp_effects", []))
    plt.figure(figsize=(8, 4))
    plt.bar(range(len(mlp_effects)), mlp_effects, color="steelblue")
    plt.xlabel("Layer")
    plt.ylabel("Recovery")
    plt.title("Propositional Logic Dataset: MLP Patching Recovery per Layer")
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "propositional_logic_mlp_recovery.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating MLP recovery plot: {e}")
    plt.close()

# Plot 3: Faithfulness vs circuit size K
try:
    faith = data.get("faithfulness", {})
    Ks = sorted(faith.keys(), key=lambda k: int(k))
    matches = [faith[k]["match"] for k in Ks]
    spars = [faith[k]["sparsity"] for k in Ks]
    faiths = [faith[k]["faithfulness"] for k in Ks]
    Ks_int = [int(k) for k in Ks]
    plt.figure(figsize=(7, 4.5))
    plt.plot(Ks_int, matches, "o-", label="Match (circuit vs full)")
    plt.plot(Ks_int, spars, "^-", label="Sparsity")
    plt.plot(Ks_int, faiths, "s-", label="Faithfulness (match*sparsity)")
    plt.xlabel("Circuit Size K")
    plt.ylabel("Score")
    plt.title(
        "Propositional Logic Dataset: Circuit Faithfulness vs Size\nMean-Ablation of Non-Circuit Components"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "propositional_logic_faithfulness_vs_K.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating faithfulness plot: {e}")
    plt.close()

# Plot 4: Top-20 components bar chart
try:
    top_comps = data.get("top_components", [])[:20]
    labels = []
    effects = []
    colors = []
    for kind, L, H, eff in top_comps:
        if kind == "attn":
            labels.append(f"A L{L}H{H}")
            colors.append("tab:blue")
        else:
            labels.append(f"M L{L}")
            colors.append("tab:orange")
        effects.append(eff)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(effects)), effects, color=colors)
    plt.xticks(range(len(effects)), labels, rotation=60, ha="right", fontsize=8)
    plt.ylabel("Patching Recovery")
    plt.title(
        "Propositional Logic Dataset: Top-20 Components by Patching Recovery\n(Blue: Attention Head, Orange: MLP)"
    )
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "propositional_logic_top_components.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating top components plot: {e}")
    plt.close()

# Plot 5: Role distribution of top-20 components (pie chart)
try:
    role_counts = data.get("role_counts_top20", {})
    labels_r = list(role_counts.keys())
    vals_r = [role_counts[k] for k in labels_r]
    plt.figure(figsize=(6, 6))
    plt.pie(
        vals_r,
        labels=labels_r,
        autopct="%1.0f%%",
        startangle=90,
        colors=["#8ecae6", "#ffb703", "#fb8500"],
    )
    plt.title(
        "Propositional Logic Dataset: Functional Role Distribution\nof Top-20 Circuit Components (by Layer Bucket)"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "propositional_logic_role_distribution.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating role distribution plot: {e}")
    plt.close()

# Plot 6: Confusion matrix from predictions vs ground truth
try:
    preds = np.array(data.get("predictions", []), dtype=bool)
    gts = np.array(data.get("ground_truth", []), dtype=bool)
    cm = np.zeros((2, 2), dtype=int)
    for p, g in zip(preds, gts):
        cm[int(g), int(p)] += 1
    plt.figure(figsize=(5, 4.5))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im)
    plt.xticks([0, 1], ["False", "True"])
    plt.yticks([0, 1], ["False", "True"])
    plt.xlabel("Predicted")
    plt.ylabel("Ground Truth")
    plt.title(
        "Propositional Logic Dataset: Full-Model Confusion Matrix\n(Clean+Corrupted Prompts)"
    )
    for r in range(2):
        for c in range(2):
            plt.text(
                c,
                r,
                str(cm[r, c]),
                ha="center",
                va="center",
                color="white" if cm[r, c] > cm.max() / 2 else "black",
            )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "propositional_logic_confusion_matrix.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating confusion matrix: {e}")
    plt.close()

# Plot 7: Validation metrics summary bar
try:
    val = data.get("metrics", {}).get("val", [])
    if val:
        m = val[0]
        keys = [
            "clean_acc",
            "corr_acc",
            "mean_clean_logit_diff",
            "mean_corr_logit_diff",
        ]
        vals = [m.get(k, 0) for k in keys]
        plt.figure(figsize=(7, 4))
        colors = ["green", "green", "steelblue", "steelblue"]
        plt.bar(keys, vals, color=colors)
        plt.xticks(rotation=20, ha="right")
        plt.ylabel("Value")
        plt.title(
            "Propositional Logic Dataset: Baseline Validation Metrics\n(Clean vs Corrupted Prompts)"
        )
        plt.axhline(0, color="black", linewidth=0.7)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "propositional_logic_val_metrics.png"), dpi=100
        )
        plt.close()
except Exception as e:
    print(f"Error creating val metrics plot: {e}")
    plt.close()

# Print summary
print("===== Summary =====")
val = data.get("metrics", {}).get("val", [])
if val:
    print(
        f"Clean acc: {val[0].get('clean_acc'):.3f}, Corr acc: {val[0].get('corr_acc'):.3f}"
    )
    print(f"Mean clean logit_diff: {val[0].get('mean_clean_logit_diff'):.3f}")
    print(f"Mean corr  logit_diff: {val[0].get('mean_corr_logit_diff'):.3f}")
faith = data.get("faithfulness", {})
for k in sorted(faith.keys(), key=lambda x: int(x)):
    f = faith[k]
    print(
        f"K={k}: match={f['match']:.3f} sparsity={f['sparsity']:.3f} faithfulness={f['faithfulness']:.3f}"
    )
print("Role counts (top-20):", data.get("role_counts_top20", {}))
