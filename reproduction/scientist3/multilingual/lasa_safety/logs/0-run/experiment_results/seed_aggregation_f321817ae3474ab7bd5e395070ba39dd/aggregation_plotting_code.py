import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2026-07-13_23-35-02_lasa_safety_attempt_2/logs/0-run/experiment_results/experiment_1297b6c39edb4b81b90bb28ec1d2e67f_proc_2606607/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        ed = np.load(full, allow_pickle=True).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")

RESOURCE_LEVEL = {
    "en": "high",
    "zh": "high",
    "it": "high",
    "vi": "medium",
    "ar": "medium",
    "ko": "medium",
    "th": "low",
    "bn": "low",
    "sw": "low",
    "jv": "low",
}


# Aggregate across runs
def sem(arr):
    arr = np.array(arr, dtype=float)
    if len(arr) <= 1:
        return 0.0
    return np.std(arr, ddof=1) / np.sqrt(len(arr))


# Collect cfg names union
cfg_names_set = set()
for ed in all_experiment_data:
    mj = ed.get("sampling_strategy", {}).get("MultiJail", {})
    cfg_names_set.update(mj.get("runs", {}).keys())
cfg_names = sorted(cfg_names_set)

# Aggregate overall ASR per cfg
agg_overall_any = {c: [] for c in cfg_names}
agg_overall_mean = {c: [] for c in cfg_names}
for ed in all_experiment_data:
    ov = (
        ed.get("sampling_strategy", {})
        .get("MultiJail", {})
        .get("overall_asr_per_cfg", {})
    )
    for c in cfg_names:
        if c in ov:
            agg_overall_any[c].append(ov[c].get("any", np.nan))
            agg_overall_mean[c].append(ov[c].get("mean", np.nan))

# Plot 1: Overall ASR with SEM
try:
    if cfg_names:
        any_means = [
            np.nanmean(agg_overall_any[c]) if agg_overall_any[c] else 0
            for c in cfg_names
        ]
        any_sems = [sem(agg_overall_any[c]) for c in cfg_names]
        mean_means = [
            np.nanmean(agg_overall_mean[c]) if agg_overall_mean[c] else 0
            for c in cfg_names
        ]
        mean_sems = [sem(agg_overall_mean[c]) for c in cfg_names]
        x = np.arange(len(cfg_names))
        w = 0.35
        plt.figure(figsize=(8, 4.5))
        plt.bar(
            x - w / 2,
            any_means,
            w,
            yerr=any_sems,
            label="any-of-N (mean ± SEM)",
            color="#4c72b0",
            capsize=4,
        )
        plt.bar(
            x + w / 2,
            mean_means,
            w,
            yerr=mean_sems,
            label="mean-of-N (mean ± SEM)",
            color="#dd8452",
            capsize=4,
        )
        plt.xticks(x, cfg_names, rotation=20, ha="right")
        plt.ylabel("Overall ASR")
        plt.ylim(0, 1)
        plt.title(
            f"MultiJail Dataset: Aggregated Overall ASR across sampling strategies\n(N runs = {len(all_experiment_data)})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "MultiJail_agg_overall_ASR.png"), dpi=120)
        plt.close()
except Exception as e:
    print(f"Error creating overall ASR plot: {e}")
    plt.close()

# Plot 2: Per-language ASR (any-of-N) aggregated across runs, per cfg (heatmap of means)
try:
    all_langs = set()
    for ed in all_experiment_data:
        runs = ed.get("sampling_strategy", {}).get("MultiJail", {}).get("runs", {})
        for c in runs:
            all_langs.update(runs[c].get("asr_any_by_language", {}).keys())
    langs_sorted = sorted(all_langs, key=lambda l: (RESOURCE_LEVEL.get(l, "z"), l))
    if langs_sorted and cfg_names:
        mat_mean = np.zeros((len(cfg_names), len(langs_sorted)))
        mat_sem = np.zeros((len(cfg_names), len(langs_sorted)))
        for i, c in enumerate(cfg_names):
            for j, lg in enumerate(langs_sorted):
                vals = []
                for ed in all_experiment_data:
                    runs = (
                        ed.get("sampling_strategy", {})
                        .get("MultiJail", {})
                        .get("runs", {})
                    )
                    if c in runs:
                        v = runs[c].get("asr_any_by_language", {}).get(lg)
                        if v is not None:
                            vals.append(v)
                mat_mean[i, j] = np.mean(vals) if vals else 0
                mat_sem[i, j] = sem(vals)
        fig, ax = plt.subplots(figsize=(11, 4.5))
        im = ax.imshow(mat_mean, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(langs_sorted)))
        ax.set_xticklabels(
            [f"{l}\n({RESOURCE_LEVEL.get(l,'?')})" for l in langs_sorted]
        )
        ax.set_yticks(np.arange(len(cfg_names)))
        ax.set_yticklabels(cfg_names)
        for i in range(mat_mean.shape[0]):
            for j in range(mat_mean.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{mat_mean[i,j]:.2f}\n±{mat_sem[i,j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="black",
                )
        plt.colorbar(im, ax=ax, label="Mean ASR (any-of-N)")
        ax.set_title(
            f"MultiJail Dataset: Aggregated Per-language ASR heatmap\n(cells: mean ± SEM, N runs = {len(all_experiment_data)})"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_agg_per_language_ASR_heatmap.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating heatmap: {e}")
    plt.close()

# Plot 3: ASR by resource level aggregated
try:
    levels = ["high", "medium", "low"]
    if cfg_names:
        x = np.arange(len(levels))
        n_cfg = len(cfg_names)
        w = 0.8 / max(n_cfg, 1)
        plt.figure(figsize=(9, 4.5))
        for i, c in enumerate(cfg_names):
            means = []
            sems = []
            for lvl in levels:
                vals = []
                for ed in all_experiment_data:
                    runs = (
                        ed.get("sampling_strategy", {})
                        .get("MultiJail", {})
                        .get("runs", {})
                    )
                    if c in runs:
                        v = runs[c].get("asr_any_by_resource", {}).get(lvl)
                        if v is not None:
                            vals.append(v)
                means.append(np.mean(vals) if vals else 0)
                sems.append(sem(vals))
            plt.bar(
                x + (i - n_cfg / 2 + 0.5) * w,
                means,
                w,
                yerr=sems,
                capsize=3,
                label=f"{c} (mean ± SEM)",
            )
        plt.xticks(x, levels)
        plt.ylabel("ASR (any-of-N)")
        plt.ylim(0, 1)
        plt.title(
            f"MultiJail Dataset: Aggregated ASR by resource level\n(N runs = {len(all_experiment_data)})"
        )
        plt.legend(fontsize=7)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_agg_ASR_by_resource_level.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating resource-level plot: {e}")
    plt.close()

# Plot 4: Best config per-language ASR aggregated (any vs mean) with SEM
try:
    # Determine most frequent best_cfg across runs
    best_cfgs = []
    for ed in all_experiment_data:
        bc = ed.get("sampling_strategy", {}).get("MultiJail", {}).get("best_cfg")
        if bc:
            best_cfgs.append(bc)
    if best_cfgs:
        best_cfg = max(set(best_cfgs), key=best_cfgs.count)
        all_langs = set()
        for ed in all_experiment_data:
            runs = ed.get("sampling_strategy", {}).get("MultiJail", {}).get("runs", {})
            if best_cfg in runs:
                all_langs.update(runs[best_cfg].get("asr_any_by_language", {}).keys())
        langs = sorted(all_langs, key=lambda l: (RESOURCE_LEVEL.get(l, "z"), l))
        any_means, any_sems, mean_means, mean_sems = [], [], [], []
        for lg in langs:
            av, mv = [], []
            for ed in all_experiment_data:
                runs = (
                    ed.get("sampling_strategy", {}).get("MultiJail", {}).get("runs", {})
                )
                if best_cfg in runs:
                    a = runs[best_cfg].get("asr_any_by_language", {}).get(lg)
                    m = runs[best_cfg].get("asr_mean_by_language", {}).get(lg)
                    if a is not None:
                        av.append(a)
                    if m is not None:
                        mv.append(m)
            any_means.append(np.mean(av) if av else 0)
            any_sems.append(sem(av))
            mean_means.append(np.mean(mv) if mv else 0)
            mean_sems.append(sem(mv))
        x = np.arange(len(langs))
        w = 0.35
        plt.figure(figsize=(10, 4.5))
        plt.bar(
            x - w / 2,
            any_means,
            w,
            yerr=any_sems,
            capsize=3,
            label="any-of-N (mean ± SEM)",
            color="#4c72b0",
        )
        plt.bar(
            x + w / 2,
            mean_means,
            w,
            yerr=mean_sems,
            capsize=3,
            label="mean-of-N (mean ± SEM)",
            color="#dd8452",
        )
        plt.xticks(x, [f"{l}\n({RESOURCE_LEVEL.get(l,'?')})" for l in langs])
        plt.ylabel("ASR")
        plt.ylim(0, 1)
        plt.title(
            f"MultiJail Dataset: Aggregated Per-language ASR for best config ({best_cfg})\nLeft: any-of-N, Right: mean-of-N (N runs = {len(all_experiment_data)})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_agg_best_config_per_language_ASR.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating best config plot: {e}")
    plt.close()

# Print aggregated metrics
try:
    print(f"Aggregated across N = {len(all_experiment_data)} run(s)")
    print("Overall ASR per sampling config (mean ± SEM):")
    for c in cfg_names:
        am = np.nanmean(agg_overall_any[c]) if agg_overall_any[c] else float("nan")
        as_ = sem(agg_overall_any[c])
        mm = np.nanmean(agg_overall_mean[c]) if agg_overall_mean[c] else float("nan")
        ms = sem(agg_overall_mean[c])
        print(f"  {c}: any={am:.3f}±{as_:.3f}, mean={mm:.3f}±{ms:.3f}")
except Exception as e:
    print(f"Error printing metrics: {e}")
