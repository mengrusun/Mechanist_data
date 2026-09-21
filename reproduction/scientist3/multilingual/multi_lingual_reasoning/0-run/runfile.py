import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_21-35-56_multi_lingual_reasoning_attempt_3/logs/0-run/experiment_results/experiment_8f134d50edfa4d07816d075486b05dda_proc_1445608/experiment_data.npy"
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        experiment_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

conds = ["baseline", "lang_suppression", "random_control"]
colors = {
    "baseline": "tab:blue",
    "lang_suppression": "tab:orange",
    "random_control": "tab:green",
}

# Gather mnt grid and languages from the first run
try:
    mgsm0 = all_experiment_data[0].get("max_new_tokens", {}).get("mgsm", {})
    mnt_grid = mgsm0.get("hyperparam_values", [])
    r0 = mgsm0.get("results_by_value", {})
    best_mnt = mnt_grid[-1] if mnt_grid else None
    langs = (
        list(r0[str(best_mnt)]["baseline"]["per_lang"].keys())
        if best_mnt is not None
        else []
    )
except Exception as e:
    print(f"Error extracting grid: {e}")
    mnt_grid, langs, best_mnt = [], [], None


def sem(a, axis=0):
    a = np.asarray(a, dtype=float)
    n = a.shape[axis]
    if n <= 1:
        return np.zeros(a.shape[1:]) if a.ndim > 1 else 0.0
    return a.std(axis=axis, ddof=1) / np.sqrt(n)


# Aggregate overall accuracy: shape (n_runs, n_mnt) per condition
overall_agg = {c: [] for c in conds}
for exp in all_experiment_data:
    rbv = exp.get("max_new_tokens", {}).get("mgsm", {}).get("results_by_value", {})
    for c in conds:
        overall_agg[c].append([rbv[str(m)][c]["overall"] for m in mnt_grid])
for c in conds:
    overall_agg[c] = np.array(overall_agg[c], dtype=float)  # (n_runs, n_mnt)

# Aggregate per-language accuracy at best_mnt: (n_runs, n_langs)
perlang_agg = {c: [] for c in conds}
for exp in all_experiment_data:
    rbv = exp.get("max_new_tokens", {}).get("mgsm", {}).get("results_by_value", {})
    for c in conds:
        perlang_agg[c].append([rbv[str(best_mnt)][c]["per_lang"][l] for l in langs])
for c in conds:
    perlang_agg[c] = np.array(perlang_agg[c], dtype=float)

n_runs = len(all_experiment_data)

# Plot 1: Overall accuracy line plot with SEM error bars
try:
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in conds:
        mean = overall_agg[c].mean(axis=0)
        err = sem(overall_agg[c], axis=0)
        ax.errorbar(
            mnt_grid,
            mean,
            yerr=err,
            marker="o",
            label=f"{c} (mean ± SEM)",
            color=colors[c],
            capsize=3,
        )
    ax.set_xlabel("max_new_tokens")
    ax.set_ylabel("Overall Accuracy")
    ax.set_title(
        f"MGSM Dataset: Aggregated Overall Accuracy vs max_new_tokens\nMean ± SEM across {n_runs} run(s)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "mgsm_agg_overall_accuracy_line.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Overall accuracy bar plot with error bars
try:
    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.25
    x = np.arange(len(mnt_grid))
    for i, c in enumerate(conds):
        mean = overall_agg[c].mean(axis=0)
        err = sem(overall_agg[c], axis=0)
        ax.bar(
            x + (i - 1) * width,
            mean,
            width,
            yerr=err,
            capsize=3,
            label=f"{c} (mean ± SEM)",
            color=colors[c],
        )
    ax.set_xticks(x)
    ax.set_xticklabels([str(m) for m in mnt_grid])
    ax.set_xlabel("max_new_tokens")
    ax.set_ylabel("Overall Accuracy")
    ax.set_title(
        f"MGSM Dataset: Aggregated Overall Accuracy (Bar)\nMean ± SEM across {n_runs} run(s)"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "mgsm_agg_overall_accuracy_bar.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Per-language accuracy at best_mnt with SEM error bars
try:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(langs))
    width = 0.25
    for i, c in enumerate(conds):
        mean = perlang_agg[c].mean(axis=0)
        err = sem(perlang_agg[c], axis=0)
        ax.bar(
            x + (i - 1) * width,
            mean,
            width,
            yerr=err,
            capsize=3,
            label=f"{c} (mean ± SEM)",
            color=colors[c],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_xlabel("Language")
    ax.set_ylabel("Accuracy")
    ax.set_title(
        f"MGSM Dataset: Per-Language Accuracy at max_new_tokens={best_mnt}\nMean ± SEM across {n_runs} run(s)"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "mgsm_agg_per_language_accuracy.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Per-language delta (condition - baseline) with SEM propagated (paired runs)
try:
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(langs))
    width = 0.35
    delta_supp = perlang_agg["lang_suppression"] - perlang_agg["baseline"]
    delta_rand = perlang_agg["random_control"] - perlang_agg["baseline"]
    ax.bar(
        x - width / 2,
        delta_supp.mean(axis=0),
        width,
        yerr=sem(delta_supp, axis=0),
        capsize=3,
        label="lang_suppression - baseline (mean ± SEM)",
        color="tab:orange",
    )
    ax.bar(
        x + width / 2,
        delta_rand.mean(axis=0),
        width,
        yerr=sem(delta_rand, axis=0),
        capsize=3,
        label="random_control - baseline (mean ± SEM)",
        color="tab:green",
    )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(langs)
    ax.set_xlabel("Language")
    ax.set_ylabel("Accuracy Delta vs Baseline")
    ax.set_title(
        f"MGSM Dataset: Accuracy Delta vs Baseline at max_new_tokens={best_mnt}\nMean ± SEM across {n_runs} run(s)"
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "mgsm_agg_accuracy_delta_per_language.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Print aggregated summary metrics
try:
    print(f"=== Aggregated Overall Accuracy (mean ± SEM across {n_runs} run(s)) ===")
    for j, m in enumerate(mnt_grid):
        parts = []
        for c in conds:
            mu = overall_agg[c][:, j].mean()
            se = sem(overall_agg[c][:, j])
            parts.append(f"{c}={mu:.4f}±{se:.4f}")
        print(f"mnt={m}: " + ", ".join(parts))

    print(f"\n=== Per-Language Accuracy at max_new_tokens={best_mnt} ===")
    for i, l in enumerate(langs):
        parts = []
        for c in conds:
            mu = perlang_agg[c][:, i].mean()
            se = sem(perlang_agg[c][:, i])
            parts.append(f"{c}={mu:.4f}±{se:.4f}")
        print(f"{l}: " + ", ".join(parts))
except Exception as e:
    print(f"Error printing summary: {e}")
