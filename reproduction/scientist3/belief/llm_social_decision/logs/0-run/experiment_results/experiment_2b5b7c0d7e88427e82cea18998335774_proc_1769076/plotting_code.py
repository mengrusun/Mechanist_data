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

dg = experiment_data.get("dictator_game", {})

# Plot 1: Effect sizes bar plot
try:
    plt.figure(figsize=(7, 4))
    eff = dg.get("effect_sizes", {})
    vars_list = list(eff.keys())
    vals = [eff[v] for v in vars_list]
    colors = ["steelblue", "salmon", "seagreen", "goldenrod"][: len(vars_list)]
    plt.bar(vars_list, vals, color=colors)
    plt.axhline(0, color="k", lw=0.5)
    plt.ylabel("Cohen's d (pos vs neg injection)")
    tl = dg.get("config", {}).get("target_layer", "?")
    plt.title(f"Dictator Game: Variable Intervention Effect Sizes (layer {tl})")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "dictator_game_effect_sizes_bar.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Baseline amounts histogram
try:
    plt.figure(figsize=(6, 4))
    base = [x for x in dg.get("baseline_amounts", []) if x is not None]
    if len(base) > 0:
        plt.hist(base, bins=np.arange(0, 22) - 0.5, color="gray", edgecolor="black")
        plt.axvline(
            np.mean(base),
            color="red",
            linestyle="--",
            label=f"mean={np.mean(base):.2f}",
        )
        plt.legend()
    plt.xlabel("Transfer amount ($)")
    plt.ylabel("Count")
    plt.title("Dictator Game: Baseline Transfer Amounts (No Intervention)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "dictator_game_baseline_histogram.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Per-variable pos vs neg means with error bars
try:
    pvt = dg.get("per_variable_transfers", {})
    vars_list = list(pvt.keys())
    pos_means, neg_means, pos_stds, neg_stds = [], [], [], []
    for v in vars_list:
        pv = np.array([x for x in pvt[v]["pos"] if x is not None], dtype=float)
        nv = np.array([x for x in pvt[v]["neg"] if x is not None], dtype=float)
        pos_means.append(pv.mean() if len(pv) else np.nan)
        neg_means.append(nv.mean() if len(nv) else np.nan)
        pos_stds.append(pv.std() if len(pv) else 0)
        neg_stds.append(nv.std() if len(nv) else 0)
    x = np.arange(len(vars_list))
    w = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(
        x - w / 2,
        pos_means,
        w,
        yerr=pos_stds,
        label="+alpha (positive)",
        color="steelblue",
        capsize=4,
    )
    plt.bar(
        x + w / 2,
        neg_means,
        w,
        yerr=neg_stds,
        label="-alpha (negative)",
        color="salmon",
        capsize=4,
    )
    plt.xticks(x, vars_list)
    plt.ylabel("Mean transfer amount ($)")
    plt.title(
        "Dictator Game: Mean Transfer per Variable\nLeft bar: Positive injection, Right bar: Negative injection"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "dictator_game_posneg_means.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Boxplot of transfer distributions per variable/condition
try:
    pvt = dg.get("per_variable_transfers", {})
    data, labels = [], []
    for v in pvt.keys():
        pv = [x for x in pvt[v]["pos"] if x is not None]
        nv = [x for x in pvt[v]["neg"] if x is not None]
        data.append(pv)
        labels.append(f"{v}\n(+)")
        data.append(nv)
        labels.append(f"{v}\n(-)")
    plt.figure(figsize=(9, 4))
    if data:
        plt.boxplot(data, labels=labels)
    plt.ylabel("Transfer amount ($)")
    plt.title("Dictator Game: Transfer Distributions by Variable & Injection Sign")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "dictator_game_transfer_boxplot.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Print summary metric
try:
    eff = dg.get("effect_sizes", {})
    abs_ds = [
        abs(v) for v in eff.values() if not (isinstance(v, float) and np.isnan(v))
    ]
    if abs_ds:
        print(f"Mean |Cohen's d| across variables: {np.mean(abs_ds):.3f}")
    for k, v in eff.items():
        print(f"  {k}: Cohen's d = {v:.3f}")
except Exception as e:
    print(f"Error printing metrics: {e}")
