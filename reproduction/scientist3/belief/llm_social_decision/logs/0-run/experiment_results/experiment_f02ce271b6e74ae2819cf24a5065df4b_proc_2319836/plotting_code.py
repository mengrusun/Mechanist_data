import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

data = experiment_data.get("multi_scenario_behavioral_generalization", {})
scenarios = list(data.keys())
variables = ["gender", "age", "instr", "meeting"]

# Plot 1: Within-scenario effect sizes (grouped bar chart)
try:
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(variables))
    w = 0.25
    colors = ["steelblue", "salmon", "seagreen"]
    for i, sc in enumerate(scenarios):
        vals = [data[sc]["effect_sizes_within"].get(v, 0.0) for v in variables]
        ax.bar(x + (i - 1) * w, vals, w, label=sc, color=colors[i % len(colors)])
    ax.set_xticks(x)
    ax.set_xticklabels(variables)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Cohen's d")
    ax.set_title(
        "Behavioral Economics: Within-Scenario Intervention Effect Sizes\nSubtitle: Cohen's d per variable across scenarios"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "behavioral_within_scenario_effects.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating within-scenario plot: {e}")
    plt.close()

# Plot 2: Cross-scenario transfer heatmap per variable
try:
    fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 4))
    if len(variables) == 1:
        axes = [axes]
    im = None
    for ax, var in zip(axes, variables):
        mat = np.zeros((len(scenarios), len(scenarios)))
        for i, src in enumerate(scenarios):
            for j, tgt in enumerate(scenarios):
                if src == tgt:
                    v = data[tgt]["effect_sizes_within"].get(var, np.nan)
                else:
                    v = data[tgt]["effect_sizes_cross"].get(src, {}).get(var, np.nan)
                mat[i, j] = v
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-2, vmax=2)
        ax.set_xticks(range(len(scenarios)))
        ax.set_xticklabels(scenarios, rotation=45)
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels(scenarios)
        ax.set_xlabel("Target")
        ax.set_ylabel("Source")
        ax.set_title(var)
        for i in range(len(scenarios)):
            for j in range(len(scenarios)):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8)
    if im is not None:
        plt.colorbar(im, ax=axes, shrink=0.8)
    plt.suptitle(
        "Behavioral Economics: Cross-Scenario Direction Transfer\nSubtitle: Rows=Source directions, Cols=Target prompts (Cohen's d)"
    )
    plt.savefig(
        os.path.join(working_dir, "behavioral_cross_scenario_transfer.png"),
        dpi=120,
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating cross-scenario plot: {e}")
    plt.close()

# Plot 3: Baseline amount distributions per scenario
try:
    fig, ax = plt.subplots(figsize=(8, 4))
    box_data = []
    labels = []
    for sc in scenarios:
        amts = [a for a in data[sc].get("baseline_amounts", []) if a is not None]
        if amts:
            box_data.append(amts)
            labels.append(sc)
    if box_data:
        ax.boxplot(box_data, labels=labels)
        ax.set_ylabel("Dollar amount")
        ax.set_title(
            "Behavioral Economics: Baseline Response Distributions\nSubtitle: No-intervention amounts across scenarios"
        )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "behavioral_baseline_distributions.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating baseline plot: {e}")
    plt.close()

# Plot 4: Mean |d| within vs cross per scenario
try:
    fig, ax = plt.subplots(figsize=(7, 4))
    within_vals, cross_vals = [], []
    for sc in scenarios:
        train_metrics = data[sc]["metrics"].get("train", [])
        if train_metrics:
            within_vals.append(train_metrics[0].get("mean_abs_d_within", np.nan))
            cross_vals.append(train_metrics[0].get("mean_abs_d_cross", np.nan))
        else:
            within_vals.append(np.nan)
            cross_vals.append(np.nan)
    x = np.arange(len(scenarios))
    w = 0.35
    ax.bar(x - w / 2, within_vals, w, label="Within", color="steelblue")
    ax.bar(x + w / 2, cross_vals, w, label="Cross", color="salmon")
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Mean |Cohen's d|")
    ax.set_title(
        "Behavioral Economics: Mean |d| Within vs Cross-Scenario\nSubtitle: Aggregated across variables"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "behavioral_within_vs_cross_meand.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating within-vs-cross plot: {e}")
    plt.close()

# Plot 5: Per-variable pos vs neg intervention means (within-scenario)
try:
    fig, axes = plt.subplots(
        1, len(scenarios), figsize=(5 * len(scenarios), 4), sharey=True
    )
    if len(scenarios) == 1:
        axes = [axes]
    for ax, sc in zip(axes, scenarios):
        transfers = data[sc].get("per_variable_transfers", {})
        pos_means, neg_means = [], []
        for var in variables:
            t = transfers.get(var, {})
            pos = [a for a in t.get("pos", []) if a is not None]
            neg = [a for a in t.get("neg", []) if a is not None]
            pos_means.append(np.mean(pos) if pos else np.nan)
            neg_means.append(np.mean(neg) if neg else np.nan)
        x = np.arange(len(variables))
        w = 0.35
        ax.bar(x - w / 2, pos_means, w, label="+alpha", color="seagreen")
        ax.bar(x + w / 2, neg_means, w, label="-alpha", color="indianred")
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=30)
        ax.set_title(sc)
        ax.legend()
    axes[0].set_ylabel("Mean dollar amount")
    plt.suptitle(
        "Behavioral Economics: Positive vs Negative Injection Means\nSubtitle: Per-variable within-scenario intervention outcomes"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "behavioral_pos_vs_neg_injection.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating pos-vs-neg plot: {e}")
    plt.close()

# Print summary metrics
try:
    print("=== Summary ===")
    for sc in scenarios:
        tm = data[sc]["metrics"].get("train", [])
        if tm:
            print(
                f"[{sc}] mean |d| within={tm[0].get('mean_abs_d_within'):.3f}, cross={tm[0].get('mean_abs_d_cross'):.3f}"
            )
except Exception as e:
    print(f"Error printing summary: {e}")
