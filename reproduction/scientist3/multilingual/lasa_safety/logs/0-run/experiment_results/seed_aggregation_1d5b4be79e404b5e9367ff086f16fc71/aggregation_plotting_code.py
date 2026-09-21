import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_23-35-02_lasa_safety_attempt_2/logs/0-run/experiment_results/experiment_d036ea40cc014b29befffae7c2d9860b_proc_1777750/experiment_data.npy",
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

# Collect MultiJail entries from all runs
mj_runs = []
for ed in all_experiment_data:
    mj = ed.get("max_new_tokens", {}).get("MultiJail", {})
    if mj:
        mj_runs.append(mj)


def sem(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) <= 1:
        return 0.0
    return np.std(arr, ddof=1) / np.sqrt(len(arr))


# Determine common hyperparameter values
hp_values = []
if mj_runs:
    hp_values = mj_runs[0].get("hyperparam_values", [])

# Plot 1: Overall ASR vs max_new_tokens (mean ± SEM across runs)
try:
    plt.figure(figsize=(6, 4))
    xs = list(hp_values)
    means, sems = [], []
    for x in xs:
        vals = []
        for mj in mj_runs:
            oa = mj.get("overall_asr_per_hp", {})
            v = oa.get(x, oa.get(str(x)))
            if v is not None:
                vals.append(v)
        if vals:
            means.append(np.mean(vals))
            sems.append(sem(vals))
        else:
            means.append(np.nan)
            sems.append(0.0)
    plt.errorbar(
        xs,
        means,
        yerr=sems,
        marker="o",
        capsize=4,
        label=f"Mean ± SEM (n={len(mj_runs)})",
    )
    plt.xlabel("max_new_tokens")
    plt.ylabel("Overall ASR")
    plt.title(
        "MultiJail: Overall ASR vs max_new_tokens\n(Aggregated Across Runs, Mean ± SEM)"
    )
    plt.xticks(xs)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "MultiJail_overall_ASR_vs_max_new_tokens_agg.png"),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: ASR by language across max_new_tokens (mean ± SEM), one line per language
try:
    all_langs = set()
    for mj in mj_runs:
        for k, run in mj.get("runs", {}).items():
            all_langs.update(run.get("asr_by_language", {}).keys())
    langs_sorted = sorted(all_langs, key=lambda l: (RESOURCE_LEVEL.get(l, "z"), l))

    plt.figure(figsize=(10, 5))
    for lang in langs_sorted:
        xs_l, means_l, sems_l = [], [], []
        for x in hp_values:
            vals = []
            for mj in mj_runs:
                run = mj.get("runs", {}).get(str(x), mj.get("runs", {}).get(x, {}))
                v = run.get("asr_by_language", {}).get(lang)
                if v is not None:
                    vals.append(v)
            if vals:
                xs_l.append(x)
                means_l.append(np.mean(vals))
                sems_l.append(sem(vals))
        if xs_l:
            plt.errorbar(xs_l, means_l, yerr=sems_l, marker="o", capsize=3, label=lang)
    plt.xlabel("max_new_tokens")
    plt.ylabel("Attack Success Rate")
    plt.title(
        "MultiJail: ASR by Language vs max_new_tokens\n(Aggregated Mean ± SEM per Language)"
    )
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "MultiJail_ASR_by_language_agg.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: ASR by resource level across max_new_tokens (mean ± SEM)
try:
    plt.figure(figsize=(7, 4))
    levels = ["high", "medium", "low"]
    for lvl in levels:
        xs_l, means_l, sems_l = [], [], []
        for x in hp_values:
            vals = []
            for mj in mj_runs:
                run = mj.get("runs", {}).get(str(x), mj.get("runs", {}).get(x, {}))
                v = run.get("asr_by_resource", {}).get(lvl)
                if v is not None:
                    vals.append(v)
            if vals:
                xs_l.append(x)
                means_l.append(np.mean(vals))
                sems_l.append(sem(vals))
        if xs_l:
            plt.errorbar(
                xs_l,
                means_l,
                yerr=sems_l,
                marker="o",
                capsize=4,
                label=f"{lvl} (mean ± SEM)",
            )
    plt.xlabel("max_new_tokens")
    plt.ylabel("Attack Success Rate")
    plt.title(
        "MultiJail: ASR by Resource Level vs max_new_tokens\n(Aggregated Across Runs)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "MultiJail_ASR_by_resource_agg.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Layer semantic alignment aggregated (mean ± SEM per layer)
try:
    lsa_list = [mj.get("layer_semantic_alignment", []) for mj in mj_runs]
    lsa_list = [l for l in lsa_list if len(l) > 0]
    if lsa_list:
        min_len = min(len(l) for l in lsa_list)
        arr = np.array([l[:min_len] for l in lsa_list], dtype=float)
        means = arr.mean(axis=0)
        sems = (
            arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
            if arr.shape[0] > 1
            else np.zeros(min_len)
        )
        layers = np.arange(min_len)
        plt.figure(figsize=(8, 4))
        plt.plot(
            layers, means, marker="o", markersize=3, label=f"Mean (n={arr.shape[0]})"
        )
        plt.fill_between(layers, means - sems, means + sems, alpha=0.3, label="± SEM")
        best_layers = [
            mj.get("best_bottleneck_layer")
            for mj in mj_runs
            if mj.get("best_bottleneck_layer") is not None
        ]
        if best_layers:
            mean_best = np.mean(best_layers)
            plt.axvline(
                mean_best,
                color="r",
                linestyle="--",
                label=f"Mean best layer={mean_best:.1f}",
            )
        plt.xlabel("Transformer Layer")
        plt.ylabel("Mean Cosine Similarity (non-en vs en)")
        plt.title(
            "MultiJail: Cross-lingual Semantic Alignment per Layer\n(Aggregated Mean ± SEM)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_layer_semantic_alignment_agg.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: ASR by language at best hyperparameter (bar chart with SEM)
try:
    best_hps = [mj.get("best_hp") for mj in mj_runs if mj.get("best_hp") is not None]
    if best_hps:
        # Use majority best_hp, or the first one's
        from collections import Counter

        best_hp = Counter([str(b) for b in best_hps]).most_common(1)[0][0]
        # Collect asr_by_language for best_hp per run
        lang_vals = {}
        for mj in mj_runs:
            run = mj.get("runs", {}).get(
                str(best_hp), mj.get("runs", {}).get(best_hp, {})
            )
            for l, v in run.get("asr_by_language", {}).items():
                lang_vals.setdefault(l, []).append(v)
        if lang_vals:
            langs_sorted = sorted(
                lang_vals.keys(), key=lambda l: (RESOURCE_LEVEL.get(l, "z"), l)
            )
            means = [np.mean(lang_vals[l]) for l in langs_sorted]
            sems = [sem(lang_vals[l]) for l in langs_sorted]
            colors_map = {"high": "#4c72b0", "medium": "#dd8452", "low": "#c44e52"}
            bar_colors = [
                colors_map.get(RESOURCE_LEVEL.get(l, ""), "#888") for l in langs_sorted
            ]
            plt.figure(figsize=(9, 4))
            plt.bar(langs_sorted, means, yerr=sems, capsize=4, color=bar_colors)
            plt.ylabel("Attack Success Rate")
            plt.ylim(0, 1)
            plt.title(
                f"MultiJail: ASR by Language at Best max_new_tokens={best_hp}\n(Aggregated Mean ± SEM, Colored by Resource Level)"
            )
            for lvl, c in colors_map.items():
                plt.bar([], [], color=c, label=lvl)
            plt.legend(title="resource")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, f"MultiJail_ASR_by_language_best_mnt{best_hp}_agg.png"
                ),
                dpi=120,
            )
            plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print aggregated evaluation metrics
print(f"Number of runs aggregated: {len(mj_runs)}")
print("Aggregated Overall ASR per max_new_tokens (mean ± SEM):")
for x in hp_values:
    vals = []
    for mj in mj_runs:
        oa = mj.get("overall_asr_per_hp", {})
        v = oa.get(x, oa.get(str(x)))
        if v is not None:
            vals.append(v)
    if vals:
        print(
            f"  max_new_tokens={x}: ASR={np.mean(vals):.4f} ± {sem(vals):.4f} (n={len(vals)})"
        )

best_hps = [mj.get("best_hp") for mj in mj_runs if mj.get("best_hp") is not None]
print(f"Best max_new_tokens per run: {best_hps}")
best_layers = [
    mj.get("best_bottleneck_layer")
    for mj in mj_runs
    if mj.get("best_bottleneck_layer") is not None
]
if best_layers:
    print(f"Best bottleneck layer per run: {best_layers}")
    print(f"Mean best bottleneck layer: {np.mean(best_layers):.2f}")
