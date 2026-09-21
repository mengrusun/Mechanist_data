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

dg = experiment_data.get("dictator_game_v2", {})
variables = list(dg.get("per_variable_effect_dollars", {}).keys())

# Plot 1: Main effects (argmax vs EV)
try:
    per_var_effect = dg.get("per_variable_effect_dollars", {})
    per_var_ev = dg.get("per_variable_effect_ev", {})
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].bar(
        variables, [per_var_effect.get(v, 0) for v in variables], color="steelblue"
    )
    ax[0].set_ylabel("Mean |Δ transfer| ($)")
    ax[0].set_title("Argmax causal steering")
    ax[1].bar(variables, [per_var_ev.get(v, 0) for v in variables], color="seagreen")
    ax[1].set_ylabel("Mean |ΔEV| ($)")
    ax[1].set_title("Logit-level (EV) causal steering")
    fig.suptitle("Dictator Game v2: Main Causal Steering Effects")
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "dictator_v2_main_effects.png"), dpi=120)
    plt.close(fig)
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Alpha sweep
try:
    alpha_sweep = dg.get("alpha_sweep", {})
    if alpha_sweep:
        any_var = next(iter(alpha_sweep))
        alphas = sorted(alpha_sweep[any_var].keys())
        fig, ax = plt.subplots(figsize=(7, 5))
        for v in variables:
            if v in alpha_sweep:
                ys = [alpha_sweep[v][a]["ev_frac_mean"] for a in alphas]
                ax.plot(alphas, ys, marker="o", label=v)
        ax.set_xlabel("α (multiples of unit)")
        ax.set_ylabel("Expected transfer / endowment")
        ax.set_title(
            "Dictator Game v2: α Sweep (Logit-level EV, best layer per variable)"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "dictator_v2_alpha_sweep.png"), dpi=120)
        plt.close(fig)
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Specificity heatmap
try:
    specificity = dg.get("specificity", {})
    if specificity and variables:
        mat = np.array(
            [[specificity[X][Y]["delta_gap"] for Y in variables] for X in variables]
        )
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        vmax = max(abs(mat).max(), 1e-6)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(variables)))
        ax.set_xticklabels(variables)
        ax.set_yticks(range(len(variables)))
        ax.set_yticklabels(variables)
        ax.set_xlabel("probed variable Y")
        ax.set_ylabel("injected variable X")
        for i in range(len(variables)):
            for j in range(len(variables)):
                ax.text(
                    j,
                    i,
                    f"{mat[i,j]:+.2f}",
                    ha="center",
                    va="center",
                    color="k",
                    fontsize=8,
                )
        plt.colorbar(im, ax=ax)
        ax.set_title("Dictator Game v2: Specificity\nΔ(EV gap of Y) when injecting X")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "dictator_v2_specificity.png"), dpi=120)
        plt.close(fig)
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Layer scan
try:
    layer_scan = dg.get("layer_scan", {})
    if layer_scan:
        any_var = next(iter(layer_scan))
        layers = sorted(layer_scan[any_var].keys())
        fig, ax = plt.subplots(figsize=(6, 4))
        for v in variables:
            if v in layer_scan:
                ys = [layer_scan[v][L]["abs"] for L in layers]
                ax.plot(layers, ys, marker="o", label=v)
        ax.set_xlabel("layer")
        ax.set_ylabel("|EV shift| under ±α")
        ax.set_title(
            "Dictator Game v2: Per-variable Direction Sensitivity Across Layers"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "dictator_v2_layer_scan.png"), dpi=120)
        plt.close(fig)
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Random control comparison
try:
    random_control = dg.get("random_control", {})
    per_var_ev = dg.get("per_variable_effect_ev", {})
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(variables))
    w = 0.35
    ax.bar(
        x - w / 2,
        [per_var_ev.get(v, 0) for v in variables],
        w,
        label="pure direction",
        color="steelblue",
    )
    ax.bar(
        x + w / 2,
        [random_control.get(v, {}).get("mean_abs_delta_ev", 0) for v in variables],
        w,
        label="random direction",
        color="salmon",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(variables)
    ax.set_ylabel("|ΔEV| ($)")
    ax.set_title("Dictator Game v2: Pure vs Random-Direction Control")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "dictator_v2_random_control.png"), dpi=120)
    plt.close(fig)
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Plot 6: Baseline transfer fraction histogram
try:
    base_amounts = dg.get("baseline_amounts", [])
    fractions = [a / e for a, e in base_amounts if a is not None]
    fig, ax = plt.subplots(figsize=(6, 4))
    if fractions:
        ax.hist(fractions, bins=20, color="gray", edgecolor="black")
        ax.axvline(
            np.mean(fractions),
            color="red",
            linestyle="--",
            label=f"mean={np.mean(fractions):.3f}",
        )
        ax.legend()
    ax.set_xlabel("Transfer / Endowment")
    ax.set_ylabel("Count")
    ax.set_title("Dictator Game v2: Baseline Transfer Fractions (No Intervention)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "dictator_v2_baseline_histogram.png"), dpi=120
    )
    plt.close(fig)
except Exception as e:
    print(f"Error creating plot6: {e}")
    plt.close()

# Plot 7: Pos vs neg injection means from main_results
try:
    main_results = dg.get("main_results", {})
    pos_means, neg_means, pos_stds, neg_stds = [], [], [], []
    for v in variables:
        mr = main_results.get(v, {})
        pv = np.array(mr.get("pos_amts", []), dtype=float)
        nv = np.array(mr.get("neg_amts", []), dtype=float)
        pos_means.append(pv.mean() if len(pv) else np.nan)
        neg_means.append(nv.mean() if len(nv) else np.nan)
        pos_stds.append(pv.std() if len(pv) else 0)
        neg_stds.append(nv.std() if len(nv) else 0)
    x = np.arange(len(variables))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(
        x - w / 2, pos_means, w, yerr=pos_stds, label="+α", color="steelblue", capsize=4
    )
    ax.bar(
        x + w / 2, neg_means, w, yerr=neg_stds, label="-α", color="salmon", capsize=4
    )
    ax.set_xticks(x)
    ax.set_xticklabels(variables)
    ax.set_ylabel("Mean transfer ($)")
    ax.set_title(
        "Dictator Game v2: Mean Transfer per Variable\nLeft: +α injection, Right: -α injection"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "dictator_v2_posneg_means.png"), dpi=120)
    plt.close(fig)
except Exception as e:
    print(f"Error creating plot7: {e}")
    plt.close()

# Plot 8: Interaction / additivity residuals
try:
    interaction = dg.get("interaction", {})
    keys = [k for k in interaction.keys() if k.startswith("g")]
    if keys:
        both_vals = [interaction[k]["ev_both"] for k in keys]
        pred_vals = [interaction[k]["predicted_additive"] for k in keys]
        residuals = [interaction[k]["residual"] for k in keys]
        x = np.arange(len(keys))
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        w = 0.35
        ax[0].bar(x - w / 2, both_vals, w, label="observed both", color="steelblue")
        ax[0].bar(x + w / 2, pred_vals, w, label="predicted additive", color="orange")
        ax[0].set_xticks(x)
        ax[0].set_xticklabels(keys, rotation=30)
        ax[0].set_ylabel("EV ($)")
        ax[0].set_title("Observed vs Predicted (gender+meeting)")
        ax[0].legend()
        ax[1].bar(x, residuals, color="purple")
        ax[1].axhline(0, color="k", lw=0.5)
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(keys, rotation=30)
        ax[1].set_ylabel("Residual ($)")
        ax[1].set_title("Interaction Residual (observed - additive)")
        fig.suptitle("Dictator Game v2: Additivity Test for Gender × Meeting")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "dictator_v2_interaction.png"), dpi=120)
        plt.close(fig)
except Exception as e:
    print(f"Error creating plot8: {e}")
    plt.close()

# Print summary
try:
    print(f"Mean |Δ$|: {dg.get('mean_abs_delta_dollars', float('nan')):.3f}")
    print(f"Mean |ΔEV|: {dg.get('mean_abs_delta_ev', float('nan')):.3f}")
    for v, val in dg.get("per_variable_effect_dollars", {}).items():
        print(
            f"  {v}: |Δ$|={val:.3f}, |ΔEV|={dg.get('per_variable_effect_ev', {}).get(v, float('nan')):.3f}"
        )
except Exception as e:
    print(f"Error printing metrics: {e}")
