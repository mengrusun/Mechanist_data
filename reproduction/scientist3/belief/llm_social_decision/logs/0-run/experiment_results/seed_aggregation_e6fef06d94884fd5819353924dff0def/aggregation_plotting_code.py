import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_14-15-51_llm_social_decision_attempt_2/logs/0-run/experiment_results/experiment_4ee7a4e7c8df47ab8d6e0fe3267adf12_proc_1769076/experiment_data.npy"
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

n_runs = len(all_experiment_data)

# Collect variables across runs
all_vars = []
for ed in all_experiment_data:
    dg = ed.get("dictator_game", {})
    for v in dg.get("effect_sizes", {}).keys():
        if v not in all_vars:
            all_vars.append(v)


def sem(a):
    a = np.asarray(a, dtype=float)
    if len(a) < 2:
        return 0.0
    return a.std(ddof=1) / np.sqrt(len(a))


# Plot 1: Aggregated effect sizes
try:
    plt.figure(figsize=(7, 4))
    means, sems = [], []
    for v in all_vars:
        vals = []
        for ed in all_experiment_data:
            eff = ed.get("dictator_game", {}).get("effect_sizes", {})
            if (
                v in eff
                and eff[v] is not None
                and not (isinstance(eff[v], float) and np.isnan(eff[v]))
            ):
                vals.append(eff[v])
        means.append(np.mean(vals) if vals else np.nan)
        sems.append(sem(vals))
    x = np.arange(len(all_vars))
    plt.bar(
        x,
        means,
        yerr=sems,
        color="steelblue",
        capsize=4,
        label=f"Mean ± SEM (n={n_runs} runs)",
    )
    plt.axhline(0, color="k", lw=0.5)
    plt.xticks(x, all_vars)
    plt.ylabel("Cohen's d (pos vs neg injection)")
    plt.title("Dictator Game: Aggregated Effect Sizes across Runs\nMean ± SEM")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "dictator_game_agg_effect_sizes.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Aggregated baseline transfer histogram (pooled) with mean ± SEM line
try:
    plt.figure(figsize=(6, 4))
    pooled = []
    run_means = []
    for ed in all_experiment_data:
        base = [
            x
            for x in ed.get("dictator_game", {}).get("baseline_amounts", [])
            if x is not None
        ]
        pooled.extend(base)
        if base:
            run_means.append(np.mean(base))
    if pooled:
        plt.hist(
            pooled,
            bins=np.arange(0, 22) - 0.5,
            color="gray",
            edgecolor="black",
            label="Pooled samples",
        )
        m = np.mean(run_means) if run_means else np.mean(pooled)
        s = sem(run_means)
        plt.axvline(
            m,
            color="red",
            linestyle="--",
            label=f"Mean of run-means={m:.2f} ± {s:.2f} SEM",
        )
        plt.legend()
    plt.xlabel("Transfer amount ($)")
    plt.ylabel("Count")
    plt.title(f"Dictator Game: Aggregated Baseline Transfers (n={n_runs} runs)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "dictator_game_agg_baseline_histogram.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Aggregated per-variable pos vs neg means with SEM across runs
try:
    pos_means, neg_means, pos_sems, neg_sems = [], [], [], []
    for v in all_vars:
        pos_run_means, neg_run_means = [], []
        for ed in all_experiment_data:
            pvt = ed.get("dictator_game", {}).get("per_variable_transfers", {})
            if v in pvt:
                pv = [x for x in pvt[v].get("pos", []) if x is not None]
                nv = [x for x in pvt[v].get("neg", []) if x is not None]
                if pv:
                    pos_run_means.append(np.mean(pv))
                if nv:
                    neg_run_means.append(np.mean(nv))
        pos_means.append(np.mean(pos_run_means) if pos_run_means else np.nan)
        neg_means.append(np.mean(neg_run_means) if neg_run_means else np.nan)
        pos_sems.append(sem(pos_run_means))
        neg_sems.append(sem(neg_run_means))
    x = np.arange(len(all_vars))
    w = 0.35
    plt.figure(figsize=(7, 4))
    plt.bar(
        x - w / 2,
        pos_means,
        w,
        yerr=pos_sems,
        label="+alpha (positive) Mean ± SEM",
        color="steelblue",
        capsize=4,
    )
    plt.bar(
        x + w / 2,
        neg_means,
        w,
        yerr=neg_sems,
        label="-alpha (negative) Mean ± SEM",
        color="salmon",
        capsize=4,
    )
    plt.xticks(x, all_vars)
    plt.ylabel("Mean transfer amount ($)")
    plt.title(
        f"Dictator Game: Aggregated Mean Transfer per Variable (n={n_runs} runs)\n"
        "Left bar: Positive injection, Right bar: Negative injection"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "dictator_game_agg_posneg_means.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Aggregated difference (pos - neg) per variable across runs
try:
    diffs_mean, diffs_sem = [], []
    for v in all_vars:
        diffs = []
        for ed in all_experiment_data:
            pvt = ed.get("dictator_game", {}).get("per_variable_transfers", {})
            if v in pvt:
                pv = [x for x in pvt[v].get("pos", []) if x is not None]
                nv = [x for x in pvt[v].get("neg", []) if x is not None]
                if pv and nv:
                    diffs.append(np.mean(pv) - np.mean(nv))
        diffs_mean.append(np.mean(diffs) if diffs else np.nan)
        diffs_sem.append(sem(diffs))
    x = np.arange(len(all_vars))
    plt.figure(figsize=(7, 4))
    plt.bar(
        x,
        diffs_mean,
        yerr=diffs_sem,
        color="seagreen",
        capsize=4,
        label=f"Mean ± SEM (n={n_runs} runs)",
    )
    plt.axhline(0, color="k", lw=0.5)
    plt.xticks(x, all_vars)
    plt.ylabel("Δ Mean Transfer (pos − neg) [$]")
    plt.title("Dictator Game: Aggregated Pos−Neg Transfer Differences per Variable")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "dictator_game_agg_posneg_diff.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Print summary metrics
try:
    print(f"Number of runs aggregated: {n_runs}")
    for v in all_vars:
        vals = []
        for ed in all_experiment_data:
            eff = ed.get("dictator_game", {}).get("effect_sizes", {})
            if v in eff and eff[v] is not None:
                vals.append(eff[v])
        if vals:
            print(
                f"  {v}: Cohen's d mean={np.mean(vals):.3f}, SEM={sem(vals):.3f}, n={len(vals)}"
            )
    all_abs = []
    for ed in all_experiment_data:
        eff = ed.get("dictator_game", {}).get("effect_sizes", {})
        for v, d in eff.items():
            if d is not None and not (isinstance(d, float) and np.isnan(d)):
                all_abs.append(abs(d))
    if all_abs:
        print(
            f"Pooled mean |Cohen's d|: {np.mean(all_abs):.3f} (SEM={sem(all_abs):.3f})"
        )
except Exception as e:
    print(f"Error printing metrics: {e}")
