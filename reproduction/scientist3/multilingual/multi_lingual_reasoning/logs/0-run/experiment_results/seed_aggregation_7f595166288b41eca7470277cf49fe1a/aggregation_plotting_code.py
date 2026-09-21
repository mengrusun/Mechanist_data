import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_21-35-56_multi_lingual_reasoning_attempt_3/logs/0-run/experiment_results/experiment_794eab6e03c84c59bbd70ab5bdddeafb_proc_3625922/experiment_data.npy"
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

# Aggregate conditions across runs
cond_names = ["baseline", "lang_suppression", "random_control"]
colors = ["gray", "tab:blue", "tab:orange"]

# Collect overall accuracies per condition per run
overall_by_cond = {c: [] for c in cond_names}
per_lang_by_cond = {c: {} for c in cond_names}  # cond -> lang -> list of accs

for exp in all_experiment_data:
    mgsm = exp.get("mgsm", {})
    conditions = mgsm.get("conditions", {})
    for c in cond_names:
        if c in conditions:
            overall_by_cond[c].append(conditions[c].get("overall", 0.0))
            per_lang = conditions[c].get("per_lang", {})
            for lang, val in per_lang.items():
                per_lang_by_cond[c].setdefault(lang, []).append(val)

n_runs = len(all_experiment_data)
present_conds = [c for c in cond_names if len(overall_by_cond[c]) > 0]

# Plot 1: Aggregated overall accuracy bar chart with SE error bars
try:
    plt.figure(figsize=(6, 4))
    means = [np.mean(overall_by_cond[c]) for c in present_conds]
    ses = [
        (
            (np.std(overall_by_cond[c], ddof=1) / np.sqrt(len(overall_by_cond[c])))
            if len(overall_by_cond[c]) > 1
            else 0.0
        )
        for c in present_conds
    ]
    x = np.arange(len(present_conds))
    plt.bar(
        x,
        means,
        yerr=ses,
        color=colors[: len(present_conds)],
        capsize=5,
        label=f"Mean ± SE (n={n_runs})",
    )
    for i, (m, s) in enumerate(zip(means, ses)):
        plt.text(i, m + s + 0.01, f"{m:.3f}", ha="center")
    plt.xticks(x, present_conds)
    plt.ylabel("multilingual_reasoning_accuracy")
    plt.title(
        "MGSM Overall Accuracy by Condition (Aggregated)\nDataset: MGSM (Qwen3-4B-Thinking), Mean ± SE"
    )
    plt.ylim(0, max([m + s for m, s in zip(means, ses)]) * 1.3 + 0.05 if means else 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "mgsm_overall_accuracy_aggregated_bar.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated overall accuracy plot: {e}")
    plt.close()

# Plot 2: Aggregated per-language grouped bar chart with error bars
try:
    langs = sorted({l for c in present_conds for l in per_lang_by_cond[c].keys()})
    if langs:
        x = np.arange(len(langs))
        width = 0.8 / max(1, len(present_conds))
        plt.figure(figsize=(max(8, len(langs) * 0.9), 5))
        for i, c in enumerate(present_conds):
            means = [
                (
                    np.mean(per_lang_by_cond[c].get(l, [0.0]))
                    if per_lang_by_cond[c].get(l)
                    else 0.0
                )
                for l in langs
            ]
            ses = [
                (
                    (
                        np.std(per_lang_by_cond[c][l], ddof=1)
                        / np.sqrt(len(per_lang_by_cond[c][l]))
                    )
                    if per_lang_by_cond[c].get(l) and len(per_lang_by_cond[c][l]) > 1
                    else 0.0
                )
                for l in langs
            ]
            plt.bar(
                x + i * width - 0.4 + width / 2,
                means,
                width,
                yerr=ses,
                capsize=3,
                label=f"{c} (Mean ± SE)",
                color=colors[i],
            )
        plt.xticks(x, langs)
        plt.ylabel("accuracy")
        plt.xlabel("language")
        plt.title(
            "MGSM Per-Language Accuracy by Condition (Aggregated)\nDataset: MGSM (Qwen3-4B-Thinking), Mean ± SE"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "mgsm_per_language_aggregated_grouped_bar.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated per-language grouped bar: {e}")
    plt.close()

# Plot 3: Aggregated heatmap of mean per-language accuracy
try:
    langs = sorted({l for c in present_conds for l in per_lang_by_cond[c].keys()})
    if langs:
        mat = np.array(
            [
                [
                    (
                        np.mean(per_lang_by_cond[c][l])
                        if per_lang_by_cond[c].get(l)
                        else 0.0
                    )
                    for l in langs
                ]
                for c in present_conds
            ]
        )
        plt.figure(figsize=(max(8, len(langs) * 0.7), 3 + 0.4 * len(present_conds)))
        im = plt.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(im, label="mean accuracy")
        plt.yticks(range(len(present_conds)), present_conds)
        plt.xticks(range(len(langs)), langs)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                plt.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if mat[i, j] < 0.5 else "black",
                    fontsize=8,
                )
        plt.title(
            "MGSM Per-Language Mean Accuracy Heatmap (Aggregated)\nRows: Conditions, Cols: Languages (Qwen3-4B-Thinking)"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "mgsm_per_language_aggregated_heatmap.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated heatmap: {e}")
    plt.close()

# Plot 4: Difference from baseline per language (with SE)
try:
    if "baseline" in present_conds:
        langs = sorted(per_lang_by_cond["baseline"].keys())
        other_conds = [c for c in present_conds if c != "baseline"]
        if langs and other_conds:
            x = np.arange(len(langs))
            width = 0.8 / max(1, len(other_conds))
            plt.figure(figsize=(max(8, len(langs) * 0.9), 5))
            for i, c in enumerate(other_conds):
                diffs_mean = []
                diffs_se = []
                for l in langs:
                    b_vals = np.array(per_lang_by_cond["baseline"].get(l, []))
                    c_vals = np.array(per_lang_by_cond[c].get(l, []))
                    n = min(len(b_vals), len(c_vals))
                    if n > 0:
                        d = c_vals[:n] - b_vals[:n]
                        diffs_mean.append(np.mean(d))
                        diffs_se.append(
                            np.std(d, ddof=1) / np.sqrt(n) if n > 1 else 0.0
                        )
                    else:
                        diffs_mean.append(0.0)
                        diffs_se.append(0.0)
                plt.bar(
                    x + i * width - 0.4 + width / 2,
                    diffs_mean,
                    width,
                    yerr=diffs_se,
                    capsize=3,
                    label=f"{c} - baseline (Mean ± SE)",
                    color=colors[cond_names.index(c)],
                )
            plt.axhline(0, color="black", linewidth=0.8)
            plt.xticks(x, langs)
            plt.ylabel("Δ accuracy (condition - baseline)")
            plt.xlabel("language")
            plt.title(
                "MGSM Per-Language Accuracy Difference from Baseline (Aggregated)\nDataset: MGSM (Qwen3-4B-Thinking), Mean ± SE"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "mgsm_per_language_diff_from_baseline.png"),
                dpi=120,
            )
            plt.close()
except Exception as e:
    print(f"Error creating diff-from-baseline plot: {e}")
    plt.close()

# Print aggregated evaluation metrics
try:
    print(
        f"=== MGSM multilingual_reasoning_accuracy (aggregated over {n_runs} runs) ==="
    )
    for c in present_conds:
        vals = overall_by_cond[c]
        m = np.mean(vals)
        s = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
        print(f"  {c}: mean={m:.4f}, SE={s:.4f}, n={len(vals)}")
except Exception as e:
    print(f"Error printing metrics: {e}")
