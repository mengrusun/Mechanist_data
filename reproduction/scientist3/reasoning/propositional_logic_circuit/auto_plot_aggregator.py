The prior scripts kept getting truncated. I'll produce a compact, complete, self-contained script that fits well within the response and covers the key figures.

```python
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300,
})

os.makedirs("figures", exist_ok=True)

BASELINE_NPY = "experiment_results/experiment_af208288b9eb4e4d84e7b58f4d740b60_proc_3610328/experiment_data.npy"
RESEARCH_NPY = "experiment_results/experiment_ad0d47e64ad84722b41506f461bea070_proc_3928150/experiment_data.npy"


def _safe_load(path):
    try:
        return np.load(path, allow_pickle=True).item()
    except Exception as e:
        print(f"[WARN] load {path}: {e}")
        return {}


baseline_data = _safe_load(BASELINE_NPY)
research_data = _safe_load(RESEARCH_NPY)
tuning = baseline_data.get("max_length_tuning", {})
mls_all = sorted([int(k) for k in tuning.keys()])
research = research_data.get("propositional_logic_circuit", {})


# --- Figure 1: Circuit overview (heatmap, MLP, top-20) ---
try:
    head_effects = np.array(research.get("head_effects", []))
    mlp_effects = np.array(research.get("mlp_effects", []))
    top_comps = research.get("top_components", [])[:20]

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))

    ax = axes[0]
    if head_effects.size:
        vmax = float(np.abs(head_effects).max())
        im = ax.imshow(head_effects, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Patching Recovery")
    ax.set_xlabel("Head Index"); ax.set_ylabel("Layer Index")
    ax.set_title("(a) Attention-Head Patching Recovery")

    ax = axes[1]
    if mlp_effects.size:
        ax.bar(range(len(mlp_effects)), mlp_effects, color="steelblue")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Layer Index"); ax.set_ylabel("Patching Recovery")
    ax.set_title("(b) MLP Patching Recovery by Layer")
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[2]
    labels, effects, colors = [], [], []
    for kind, L, H, eff in top_comps:
        if kind == "attn":
            labels.append(f"A L{L}H{H}"); colors.append("tab:blue")
        else:
            labels.append(f"M L{L}"); colors.append("tab:orange")
        effects.append(eff)
    if effects:
        ax.bar(range(len(effects)), effects, color=colors)
        ax.set_xticks(range(len(effects)))
        ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("Patching Recovery")
    ax.set_title("(c) Top-20 Components (blue Attn, orange MLP)")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Mechanistic Circuit for Propositional-Logic Reasoning in Mistral-7B",
                 fontsize=15, y=1.03)
    plt.tight_layout()
    plt.savefig("figures/fig1_circuit_overview.png", bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"[ERROR] Fig1: {e}"); plt.close()


# --- Figure 2: Faithfulness curves & functional roles ---
try:
    faith = research.get("faithfulness", {})
    role_counts = research.get("role_counts_top20", {})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    if faith:
        Ks_sorted = sorted(faith.keys(), key=lambda k: int(k))
        Ks_int = [int(k) for k in Ks_sorted]
        matches = [faith[k]["match"] for k in Ks_sorted]
        spars = [faith[k]["sparsity"] for k in Ks_sorted]
        faiths = [faith[k]["faithfulness"] for k in Ks_sorted]
        ax.plot(Ks_int, matches, "o-", label="Match (circuit vs full)", color="tab:green")
        ax.plot(Ks_int, spars, "^-", label="Sparsity", color="tab:purple")
        ax.plot(Ks_int, faiths, "s-", label="Faithfulness (match x sparsity)", color="tab:red")
    ax.set_xlabel("Circuit Size K"); ax.set_ylabel("Score")
    ax.set_title("(a) Circuit Faithfulness vs Size")
    ax.legend(loc="best"); ax.grid(True, alpha=0.3)

    ax = axes[1]
    if role_counts:
        labels_r = [k.replace("_", " ") for k in role_counts.keys()]
        vals_r = [role_counts[k] for k in role_counts.keys()]
        ax.pie(vals_r, labels=labels_r, autopct="%1.0f%%", startangle=90,
               colors=["#8ecae6", "#ffb703", "#fb8500", "#219ebc"],
               textprops={"fontsize": 12})
    ax.set_title("(b) Functional Role Distribution (Top-20)")

    fig.suptitle("Circuit Faithfulness and Functional Modularity",
                 fontsize=15, y=1.03)
    plt.tight_layout()
    plt.savefig("figures/fig2_faithfulness_and_roles.png", bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"[ERROR] Fig2: {e}"); plt.close()


# --- Figure 3: Baseline behavior (confusion + val metrics) ---
try:
    preds = np.array(research.get("predictions", []), dtype=bool)
    gts = np.array(research.get("ground_truth", []), dtype=bool)
    val = research.get("metrics", {}).get("val", [])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    cm = np.zeros((2, 2), dtype=int)
    for p, g in zip(preds, gts):
        cm[int(g), int(p)] += 1
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["False", "True"]); ax.set_yticklabels(["False", "True"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Ground Truth")
    ax.set_title("(a) Confusion Matrix on Propositional-Logic Prompts")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                    color="white" if cm[r, c] > cm.max() / 2 else "black",
                    fontsize=14, fontweight="bold")

    ax = axes[1]
    if val:
        m = val[0]
        keys = ["clean_acc", "corr_acc", "mean_clean_logit_diff", "mean_corr_logit_diff"]
        pretty = ["Clean Acc", "Corrupted Acc", "Mean Clean Delta-logit", "Mean Corr Delta-logit"]
        vals = [m.get(k, 0) for k in keys]
        colors = ["seagreen", "salmon", "steelblue", "goldenrod"]
        bars = ax.bar(pretty, vals, color=colors)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=10)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_ylabel("Value")
        ax.set_title("(b) Baseline Validation Metrics (Clean vs Corrupted)")
        ax.grid(True, alpha=0.3, axis="y")
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    fig.suptitle("Mistral-7B Baseline Behavior on Propositional-Logic Task",
                 fontsize=15, y=1.03)
    plt.tight_layout()
    plt.savefig("figures/fig3_baseline_behavior.png", bbox_inches="tight")
    plt.close()
except Exception as e:
    print(f"[ERROR] Fig3: {e}"); plt.close()


# --- Figure 4: Depth-wise distribution ---
try:
    head_effects = np.array(research.get("head_effects", []))
    mlp_effects = np.array(research.get("mlp_effects", []))
    if head_effects.size and mlp_effects.size:
        L = np.arange(head_effects.shape[0])
        layer_max_attn = np.max(np.abs(head_effects), axis=1)
        layer_sum_attn = np.sum(np.abs(head_effects), axis=1)
        thr = 0.5 * float(np.abs(head_effects).max())
        n_important = np.sum(np.abs(head_effects) > thr, axis=1)

        fig, axes = plt.subplots(1, 3, figsize=(19, 5))

        ax = axes[0]
        ax.plot(L, layer_max_attn, "o-", label="Max Attn Head Effect", color="tab:blue")
        ax.plot(L, np.abs(mlp_effects), "s-", label="MLP Effect", color="tab:orange")
        ax.set_xlabel("Layer Index"); ax.set_ylabel("Absolute Patching Recovery")
        ax.set_title("(a) Per-Layer Maximum Component Effect")
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[1]
        ax.plot(L, layer_sum_attn, "o-", label="Sum Attn Heads", color="tab:blue")
        ax.plot(L, np.abs(mlp_effects), "s-", label="MLP Effect", color="tab:orange")
        ax.set_xlabel("Layer Index"); ax.set_ylabel("Aggregate Absolute Recovery")
        ax.set_title("(b) Per-Layer Aggregate Effect")
        ax.legend(); ax.grid(True, alpha=0.3)

        ax = axes[2]
        ax.bar(L, n_important, color="tab:purple")
        ax.set_xlabel("Layer Index"); ax.set_ylabel("Number of Heads")
        ax.set_title("(c) Heads with Effect > 50% of Max")
        ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle("Depth-wise Distribution of Circuit Components",
                     fontsize=15, y=