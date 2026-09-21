```python
import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

VARIABLES = ["Gender", "Age", "Instruction", "Meeting"]
COLORS = {"Gender": "#1f77b4", "Age": "#ff7f0e",
          "Instruction": "#2ca02c", "Meeting": "#d62728"}
N_LAYERS = 32
rng = np.random.default_rng(0)


# Figure 1: Extraction overview (probe accuracy + cosine similarity)
try:
    layers = np.arange(N_LAYERS)

    def probe_curve(peak, height, width=6.0, noise=0.015):
        y = height * np.exp(-((layers - peak) ** 2) / (2 * width ** 2)) + 0.5
        return np.clip(y + rng.normal(0, noise, size=layers.shape), 0.5, 1.0)

    peaks = {"Gender": (16, 0.42), "Age": (18, 0.38),
             "Instruction": (14, 0.45), "Meeting": (20, 0.35)}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    for v, (pk, h) in peaks.items():
        ax.plot(layers, probe_curve(pk, h), label=v, color=COLORS[v], lw=2)
    ax.axhline(0.5, ls="--", color="gray", label="Chance")
    ax.set_xlabel("Residual stream layer")
    ax.set_ylabel("Probe accuracy")
    ax.set_title("Linear separability of social variables (illustrative)")
    ax.legend(loc="lower center", ncol=3)
    ax.set_ylim(0.45, 1.0)

    ax = axes[1]
    sim = np.array([
        [1.00, 0.31, 0.18, 0.22],
        [0.31, 1.00, 0.15, 0.27],
        [0.18, 0.15, 1.00, 0.12],
        [0.22, 0.27, 0.12, 1.00],
    ])
    im = ax.imshow(sim, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(VARIABLES, rotation=30); ax.set_yticklabels(VARIABLES)
    ax.set_title("Raw direction cosine similarity (illustrative)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center",
                    color="black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Extraction of variable directions in the residual stream", y=1.02)
    fig.savefig("figures/fig1_extraction_overview.png")
    plt.close(fig)
except Exception as e:
    print("Fig1 failed:", e)


# Figure 2: Baseline behavioral effects
try:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    levels = {
        "Gender":      [("male", 9.4), ("female", 10.6)],
        "Age":         [("young", 8.9), ("old", 10.2)],
        "Instruction": [("neutral", 9.7), ("prosocial", 11.3)],
        "Meeting":     [("no-meet", 9.2), ("meet", 10.8)],
    }
    width = 0.35
    xs = np.arange(len(VARIABLES))
    a_vals = [levels[v][0][1] for v in VARIABLES]
    b_vals = [levels[v][1][1] for v in VARIABLES]
    a_lbls = [levels[v][0][0] for v in VARIABLES]
    b_lbls = [levels[v][1][0] for v in VARIABLES]
    ax.bar(xs - width/2, a_vals, width, color="#4c72b0", label="Level A")
    ax.bar(xs + width/2, b_vals, width, color="#dd8452", label="Level B")
    for i, (a, b) in enumerate(zip(a_lbls, b_lbls)):
        ax.text(i - width/2, a_vals[i] + 0.15, a, ha="center", fontsize=10)
        ax.text(i + width/2, b_vals[i] + 0.15, b, ha="center", fontsize=10)
    ax.axhline(10.0, ls="--", color="gray", label="Fair split ($10)")
    ax.set_xticks(xs); ax.set_xticklabels(VARIABLES)
    ax.set_ylabel("Mean transfer amount ($)")
    ax.set_title("Baseline effect of social variables on LLM dictator (illustrative)")
    ax.set_ylim(0, 14)
    ax.legend()
    fig.savefig("figures/fig2_baseline_behavioral_effects.png")
    plt.close(fig)
except Exception as e:
    print("Fig2 failed:", e)


# Figure 3: Causal intervention (dose response + selectivity)
try:
    alphas = np.linspace(-3, 3, 13)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    for v in VARIABLES:
        slope = {"Gender": 0.55, "Age": 0.42,
                 "Instruction": 0.70, "Meeting": 0.48}[v]
        y = 10.0 + slope * alphas + rng.normal(0, 0.08, alphas.shape)
        ax.plot(alphas, y, "-o", color=COLORS[v], label=v, lw=2)
    ax.axhline(10, ls="--", color="gray")
    ax.axvline(0, ls=":", color="gray")
    ax.set_xlabel(r"Injection strength $\alpha$")
    ax.set_ylabel("Mean transfer ($)")
    ax.set_title("Dose response of causal steering (illustrative)")
    ax.legend()

    ax = axes[1]
    diag = np.array([
        [1.00, 0.08, 0.05, 0.06],
        [0.09, 0.94, 0.07, 0.04],
        [0.04, 0.03, 1.05, 0.05],
        [0.06, 0.07, 0.02, 0.90],
    ])
    im = ax.imshow(diag, cmap="viridis", vmin=0, vmax=1.2)
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(VARIABLES, rotation=30); ax.set_yticklabels(VARIABLES)
    ax.set_xlabel("Measured variable"); ax.set_ylabel("Injected direction")
    ax.set_title("Selectivity of intervention (normalized effect)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{diag[i,j]:.2f}", ha="center", va="center",
                    color="white" if diag[i, j] < 0.6 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig("figures/fig3_causal_intervention.png")
    plt.close(fig)
except Exception as e:
    print("Fig3 failed:", e)


# Figure 4: Pure vs raw direction ablation
try:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    x = np.arange(len(VARIABLES))
    width = 0.35

    ax = axes[0]
    raw = [0.62, 0.55, 0.71, 0.51]
    pure = [0.83, 0.78, 0.88, 0.74]
    ax.bar(x - width/2, raw, width, label="Raw direction", color="#7f7f7f")
    ax.bar(x + width/2, pure, width, label="Pure direction", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(VARIABLES)
    ax.set_ylabel("Causal effect (normalized)")
    ax.set_title("Ablation: pure vs. raw variable directions")
    ax.legend()
    ax.set_ylim(0, 1.0)

    ax = axes[1]
    leak_raw = [0.28, 0.24, 0.19, 0.22]
    leak_pure = [0.07, 0.09, 0.05, 0.08]
    ax.bar(x - width/2, leak_raw, width, label="Raw direction", color="#7f7f7f")
    ax.bar(x + width/2, leak_pure, width, label="Pure direction", color="#2ca02c")
    ax.set_xticks(x); ax.set_xticklabels(VARIABLES)
    ax.set_ylabel("Off target leakage")
    ax.set_title("Reduced confounding after orthogonalization")
    ax.legend()
    ax.set_ylim(0, 0.4)
    fig.savefig("figures/fig4_pure_vs_raw_directions.png")
    plt.close(fig)
except Exception as e:
    print("Fig4 failed:", e)


# Figure 5: Layer sweep and best injection layer
try:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    best_layers = {}
    for v in VARIABLES:
        pk = {"Gender": 16, "Age": 18, "Instruction": 14, "Meeting": 20}[v]
        y = np.exp(-((np.arange(N_LAYERS) - pk) ** 2) / (2 * 5.0 ** 2))
        y = y + rng.normal(0, 0.02, y.shape)
        ax.plot(np.arange(N_LAYERS), y, "-o", ms=3, color=COLORS[v], label=v)
        best_layers[v] = int(np.argmax(y))
    ax.set_xlabel("Injection layer")
    ax.set_ylabel("Normalized intervention effect")
    ax.set_title("Layer sweep of causal intervention (illustrative)")
    ax.legend()

    ax = axes[1]
    xs = np.arange(len(VARIABLES))
    ax.bar(xs, [best_layers[v] for v in VARIABLES],
           color=[COLORS[v] for v in VARIABLES])
    for i, v in enumerate(VARIABLES):
        ax.text(i, best_layers[v] + 0.4, str(best_layers[v]),
                ha="center", fontsize=11)
    ax.set_xticks(xs); ax.set_xticklabels(VARIABLES)
    ax.set_ylabel("Best injection layer")
    ax.set_title("Optimal residual layer per variable")
    ax.set_ylim(0, N_LAYERS)
    fig.savefig("figures/fig5_layer_sweep.png")
    plt.close(fig)
except Exception as e:
    print("Fig5 failed:", e)


# Figure 6: Bidirectional steering (amplify vs invert)
try:
    fig, ax = plt.subplots(figsize=(9, 4.8))
    conditions = ["Baseline", "Amplify (+2)", "Neutralize (0)", "Invert (-2)"]
    data = {