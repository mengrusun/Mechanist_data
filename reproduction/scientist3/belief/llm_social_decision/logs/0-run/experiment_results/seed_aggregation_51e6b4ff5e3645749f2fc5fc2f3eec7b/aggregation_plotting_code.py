import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_14-15-51_llm_social_decision_attempt_2/logs/0-run/experiment_results/experiment_7d20ddb2522144bca3229ed238294655_proc_2319836/experiment_data.npy",
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


def sem(x):
    x = np.array(
        [v for v in x if v is not None and not (isinstance(v, float) and np.isnan(v))]
    )
    if len(x) < 2:
        return 0.0
    return np.std(x, ddof=1) / np.sqrt(len(x))


# Collect scenarios and variables across runs
scenarios = set()
variables = ["gender", "age", "instr", "meeting"]
for ed in all_experiment_data:
    data = ed.get("multi_scenario_behavioral_generalization", {})
    scenarios.update(data.keys())
scenarios = sorted(scenarios)
n_runs = len(all_experiment_data)

# Plot 1: Aggregated within-scenario effect sizes with SEM error bars
try:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(variables))
    w = 0.8 / max(len(scenarios), 1)
    colors = ["steelblue", "salmon", "seagreen", "orange", "purple"]
    for i, sc in enumerate(scenarios):
        means, sems = [], []
        for v in variables:
            vals = []
            for ed in all_experiment_data:
                d = ed.get("multi_scenario_behavioral_generalization", {}).get(sc, {})
                val = d.get("effect_sizes_within", {}).get(v, None)
                if val is not None:
                    vals.append(val)
            means.append(np.mean(vals) if vals else 0.0)
            sems.append(sem(vals))
        ax.bar(
            x + (i - (len(scenarios) - 1) / 2) * w,
            means,
            w,
            yerr=sems,
            label=f"{sc} (mean±SEM, n={n_runs})",
            color=colors[i % len(colors)],
            capsize=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(variables)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("Cohen's d (mean across runs)")
    ax.set_title(
        "Behavioral Economics (Aggregated): Within-Scenario Intervention Effect Sizes\n"
        f"Subtitle: Mean ± SEM across {n_runs} run(s), per variable and scenario"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "agg_behavioral_within_scenario_effects.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated within-scenario plot: {e}")
    plt.close()

# Plot 2: Aggregated cross-scenario transfer heatmap (mean across runs) per variable
try:
    fig, axes = plt.subplots(1, len(variables), figsize=(4 * len(variables), 4))
    if len(variables) == 1:
        axes = [axes]
    im = None
    for ax, var in zip(axes, variables):
        mat = np.zeros((len(scenarios), len(scenarios)))
        for i, src in enumerate(scenarios):
            for j, tgt in enumerate(scenarios):
                vals = []
                for ed in all_experiment_data:
                    d = ed.get("multi_scenario_behavioral_generalization", {}).get(
                        tgt, {}
                    )
                    if src == tgt:
                        v = d.get("effect_sizes_within", {}).get(var, None)
                    else:
                        v = d.get("effect_sizes_cross", {}).get(src, {}).get(var, None)
                    if v is not None:
                        vals.append(v)
                mat[i, j] = np.mean(vals) if vals else np.nan
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
                val = mat[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
    if im is not None:
        plt.colorbar(im, ax=axes, shrink=0.8)
    plt.suptitle(
        "Behavioral Economics (Aggregated): Cross-Scenario Direction Transfer\n"
        f"Subtitle: Mean Cohen's d across {n_runs} run(s). Rows=Source, Cols=Target"
    )
    plt.savefig(
        os.path.join(working_dir, "agg_behavioral_cross_scenario_transfer.png"),
        dpi=120,
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated cross-scenario plot: {e}")
    plt.close()

# Plot 3: Aggregated baseline distributions - mean and SEM per scenario
try:
    fig, ax = plt.subplots(figsize=(8, 4))
    means, sems, labels = [], [], []
    for sc in scenarios:
        all_amts = []
        for ed in all_experiment_data:
            d = ed.get("multi_scenario_behavioral_generalization", {}).get(sc, {})
            amts = [a for a in d.get("baseline_amounts", []) if a is not None]
            all_amts.extend(amts)
        if all_amts:
            means.append(np.mean(all_amts))
            sems.append(sem(all_amts))
            labels.append(sc)
    x = np.arange(len(labels))
    ax.bar(
        x,
        means,
        yerr=sems,
        capsize=5,
        color="steelblue",
        label=f"Mean ± SEM (n_runs={n_runs})",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Dollar amount")
    ax.set_title(
        "Behavioral Economics (Aggregated): Baseline Response Means\n"
        f"Subtitle: No-intervention amounts, mean ± SEM aggregated across {n_runs} run(s)"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "agg_behavioral_baseline_means.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating aggregated baseline plot: {e}")
    plt.close()

# Plot 4: Mean |d| within vs cross per scenario, aggregated
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    within_means, cross_means, within_sems, cross_sems = [], [], [], []
    for sc in scenarios:
        w_vals, c_vals = [], []
        for ed in all_experiment_data:
            d = ed.get("multi_scenario_behavioral_generalization", {}).get(sc, {})
            tm = d.get("metrics", {}).get("train", [])
            if tm:
                w = tm[0].get("mean_abs_d_within", None)
                c = tm[0].get("mean_abs_d_cross", None)
                if w is not None:
                    w_vals.append(w)
                if c is not None:
                    c_vals.append(c)
        within_means.append(np.mean(w_vals) if w_vals else np.nan)
        cross_means.append(np.mean(c_vals) if c_vals else np.nan)
        within_sems.append(sem(w_vals))
        cross_sems.append(sem(c_vals))
    x = np.arange(len(scenarios))
    w = 0.35
    ax.bar(
        x - w / 2,
        within_means,
        w,
        yerr=within_sems,
        label="Within (mean±SEM)",
        color="steelblue",
        capsize=4,
    )
    ax.bar(
        x + w / 2,
        cross_means,
        w,
        yerr=cross_sems,
        label="Cross (mean±SEM)",
        color="salmon",
        capsize=4,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(scenarios)
    ax.set_ylabel("Mean |Cohen's d|")
    ax.set_title(
        "Behavioral Economics (Aggregated): Mean |d| Within vs Cross-Scenario\n"
        f"Subtitle: Aggregated across {n_runs} run(s), error bars = SEM"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "agg_behavioral_within_vs_cross_meand.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated within-vs-cross plot: {e}")
    plt.close()

# Plot 5: Aggregated pos vs neg injection means per scenario with SEM
try:
    fig, axes = plt.subplots(
        1, max(len(scenarios), 1), figsize=(5 * max(len(scenarios), 1), 4), sharey=True
    )
    if len(scenarios) == 1:
        axes = [axes]
    elif len(scenarios) == 0:
        axes = [plt.gca()]
    for ax, sc in zip(axes, scenarios):
        pos_means, neg_means, pos_sems, neg_sems = [], [], [], []
        for var in variables:
            pos_all, neg_all = [], []
            for ed in all_experiment_data:
                d = ed.get("multi_scenario_behavioral_generalization", {}).get(sc, {})
                t = d.get("per_variable_transfers", {}).get(var, {})
                pos_all.extend([a for a in t.get("pos", []) if a is not None])
                neg_all.extend([a for a in t.get("neg", []) if a is not None])
            pos_means.append(np.mean(pos_all) if pos_all else np.nan)
            neg_means.append(np.mean(neg_all) if neg_all else np.nan)
            pos_sems.append(sem(pos_all))
            neg_sems.append(sem(neg_all))
        x = np.arange(len(variables))
        w = 0.35
        ax.bar(
            x - w / 2,
            pos_means,
            w,
            yerr=pos_sems,
            label="+alpha (mean±SEM)",
            color="seagreen",
            capsize=3,
        )
        ax.bar(
            x + w / 2,
            neg_means,
            w,
            yerr=neg_sems,
            label="-alpha (mean±SEM)",
            color="indianred",
            capsize=3,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(variables, rotation=30)
        ax.set_title(sc)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean dollar amount")
    plt.suptitle(
        "Behavioral Economics (Aggregated): Positive vs Negative Injection Means\n"
        f"Subtitle: Per-variable within-scenario outcomes, mean ± SEM across {n_runs} run(s)"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "agg_behavioral_pos_vs_neg_injection.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated pos-vs-neg plot: {e}")
    plt.close()

# Print aggregated summary metrics
try:
    print(f"=== Aggregated Summary (n_runs={n_runs}) ===")
    for sc in scenarios:
        w_vals, c_vals = [], []
        for ed in all_experiment_data:
            d = ed.get("multi_scenario_behavioral_generalization", {}).get(sc, {})
            tm = d.get("metrics", {}).get("train", [])
            if tm:
                if tm[0].get("mean_abs_d_within") is not None:
                    w_vals.append(tm[0]["mean_abs_d_within"])
                if tm[0].get("mean_abs_d_cross") is not None:
                    c_vals.append(tm[0]["mean_abs_d_cross"])
        wm = np.mean(w_vals) if w_vals else float("nan")
        cm = np.mean(c_vals) if c_vals else float("nan")
        ws = sem(w_vals)
        cs = sem(c_vals)
        print(f"[{sc}] mean |d| within={wm:.3f}±{ws:.3f}, cross={cm:.3f}±{cs:.3f}")
except Exception as e:
    print(f"Error printing aggregated summary: {e}")
