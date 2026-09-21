import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_23-35-02_lasa_safety_attempt_2/logs/0-run/experiment_results/experiment_b9551452ac124595bb9171759c7ff3c3_proc_1674285/experiment_data.npy"
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
colors_map = {"high": "#4c72b0", "medium": "#dd8452", "low": "#c44e52"}

# Aggregate across runs
mj_runs = [ed.get("MultiJail", {}) for ed in all_experiment_data if ed.get("MultiJail")]
n_runs = len(mj_runs)
print(f"Number of runs: {n_runs}")


def agg_dict(runs, key):
    """Aggregate dict-valued metric across runs -> {k: (mean, sem)}"""
    combined = {}
    for r in runs:
        d = r.get(key, {})
        for k, v in d.items():
            combined.setdefault(k, []).append(v)
    return {
        k: (np.mean(vs), np.std(vs) / np.sqrt(len(vs)) if len(vs) > 1 else 0.0)
        for k, vs in combined.items()
    }


def agg_scalar(runs, key):
    vals = [r.get(key) for r in runs if r.get(key) is not None]
    if not vals:
        return None, None
    return np.mean(vals), (np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)


# Plot 1: ASR by language (aggregated)
try:
    asr_lang_agg = agg_dict(mj_runs, "asr_by_language")
    if asr_lang_agg:
        langs_sorted = sorted(
            asr_lang_agg.keys(), key=lambda l: (RESOURCE_LEVEL.get(l, "z"), l)
        )
        means = [asr_lang_agg[l][0] for l in langs_sorted]
        sems = [asr_lang_agg[l][1] for l in langs_sorted]
        bar_colors = [
            colors_map.get(RESOURCE_LEVEL.get(l, ""), "#888") for l in langs_sorted
        ]

        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.bar(
            langs_sorted,
            means,
            yerr=sems,
            color=bar_colors,
            capsize=4,
            error_kw={"ecolor": "black", "elinewidth": 1},
        )
        ax.set_ylabel("Attack Success Rate (mean ± SEM)")
        ax.set_xlabel("Language")
        ax.set_ylim(0, 1)
        ax.set_title(
            f"MultiJail Dataset: Aggregated ASR by Language\n"
            f"(Mean ± SEM across {n_runs} run(s), colored by resource level)"
        )
        for lvl, c in colors_map.items():
            ax.bar([], [], color=c, label=lvl)
        ax.bar([], [], color="white", edgecolor="black", label="SEM error bar")
        ax.legend(title="Resource")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_ASR_by_language_aggregated.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated ASR-by-language plot: {e}")
    plt.close()

# Plot 2: ASR by resource level (aggregated)
try:
    asr_res_agg = agg_dict(mj_runs, "asr_by_resource")
    if asr_res_agg:
        levels = [l for l in ["high", "medium", "low"] if l in asr_res_agg]
        means = [asr_res_agg[l][0] for l in levels]
        sems = [asr_res_agg[l][1] for l in levels]
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.bar(
            levels,
            means,
            yerr=sems,
            color=[colors_map[l] for l in levels],
            capsize=5,
            error_kw={"ecolor": "black"},
        )
        ax.set_ylabel("Attack Success Rate (mean ± SEM)")
        ax.set_xlabel("Resource Level")
        ax.set_ylim(0, 1)
        ax.set_title(
            f"MultiJail Dataset: Aggregated ASR by Resource Level\n"
            f"(Mean ± SEM across {n_runs} run(s))"
        )
        for i, (m, s) in enumerate(zip(means, sems)):
            ax.text(i, m + s + 0.02, f"{m:.2f}±{s:.2f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_ASR_by_resource_aggregated.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated ASR-by-resource plot: {e}")
    plt.close()

# Plot 3: Layer-wise semantic alignment (aggregated)
try:
    align_list = [
        np.array(r.get("layer_semantic_alignment", []))
        for r in mj_runs
        if len(r.get("layer_semantic_alignment", [])) > 0
    ]
    if align_list:
        min_len = min(len(a) for a in align_list)
        stacked = np.stack([a[:min_len] for a in align_list], axis=0)
        mean_curve = stacked.mean(axis=0)
        sem_curve = (
            stacked.std(axis=0) / np.sqrt(stacked.shape[0])
            if stacked.shape[0] > 1
            else np.zeros_like(mean_curve)
        )
        best_layer = int(np.argmax(mean_curve))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        x = np.arange(min_len)
        ax.plot(
            x,
            mean_curve,
            marker="o",
            markersize=3,
            color="#4c72b0",
            label="Mean alignment",
        )
        ax.fill_between(
            x,
            mean_curve - sem_curve,
            mean_curve + sem_curve,
            color="#4c72b0",
            alpha=0.25,
            label="± SEM",
        )
        ax.axvline(
            best_layer, color="r", linestyle="--", label=f"Best layer={best_layer}"
        )
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Mean cosine similarity (non-en vs en)")
        ax.set_title(
            f"MultiJail Dataset: Aggregated Cross-lingual Semantic Alignment\n"
            f"(Mean ± SEM across {n_runs} run(s), LLaMA-3.1-8B-Instruct)"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_layer_alignment_aggregated.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated alignment plot: {e}")
    plt.close()

# Plot 4: Overall vs per-resource ASR (aggregated)
try:
    overall_mean, overall_sem = agg_scalar(mj_runs, "overall_asr")
    asr_res_agg = agg_dict(mj_runs, "asr_by_resource")
    if overall_mean is not None and asr_res_agg:
        labels = ["overall"] + [
            l for l in ["high", "medium", "low"] if l in asr_res_agg
        ]
        means = [overall_mean] + [asr_res_agg[l][0] for l in labels[1:]]
        sems = [overall_sem] + [asr_res_agg[l][1] for l in labels[1:]]
        bar_colors = ["#555"] + [colors_map.get(l, "#888") for l in labels[1:]]

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.bar(
            labels,
            means,
            yerr=sems,
            color=bar_colors,
            capsize=5,
            error_kw={"ecolor": "black"},
        )
        ax.set_ylim(0, 1)
        ax.set_ylabel("Attack Success Rate (mean ± SEM)")
        ax.set_title(
            f"MultiJail Dataset: Overall vs Per-Resource ASR (Aggregated)\n"
            f"(Mean ± SEM across {n_runs} run(s))"
        )
        for i, (m, s) in enumerate(zip(means, sems)):
            ax.text(i, m + s + 0.02, f"{m:.2f}±{s:.2f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_overall_vs_resource_aggregated.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated summary plot: {e}")
    plt.close()

# Print aggregated metrics
print("\n=== Aggregated Evaluation Metrics ===")
overall_mean, overall_sem = agg_scalar(mj_runs, "overall_asr")
if overall_mean is not None:
    print(f"Overall ASR: {overall_mean:.4f} ± {overall_sem:.4f} (SEM, n={n_runs})")

asr_res_agg = agg_dict(mj_runs, "asr_by_resource")
print("\nASR by Resource Level (mean ± SEM):")
for lvl in ["high", "medium", "low"]:
    if lvl in asr_res_agg:
        m, s = asr_res_agg[lvl]
        print(f"  {lvl}: {m:.4f} ± {s:.4f}")

asr_lang_agg = agg_dict(mj_runs, "asr_by_language")
print("\nASR by Language (mean ± SEM):")
for lg in sorted(asr_lang_agg.keys(), key=lambda l: (RESOURCE_LEVEL.get(l, "z"), l)):
    m, s = asr_lang_agg[lg]
    print(f"  {lg} ({RESOURCE_LEVEL.get(lg, '?')}): {m:.4f} ± {s:.4f}")

print(f"\nPlots saved to: {working_dir}")
