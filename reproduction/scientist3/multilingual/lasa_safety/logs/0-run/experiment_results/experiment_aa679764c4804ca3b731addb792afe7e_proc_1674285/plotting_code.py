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

mj = experiment_data.get("MultiJail", {})

# Plot 1: ASR by language
try:
    asr_by_lang = mj.get("asr_by_language", {})
    if asr_by_lang:
        fig, ax = plt.subplots(figsize=(9, 4))
        langs_sorted = sorted(
            asr_by_lang.keys(), key=lambda l: (RESOURCE_LEVEL.get(l, "z"), l)
        )
        bar_colors = [
            colors_map.get(RESOURCE_LEVEL.get(l, ""), "#888") for l in langs_sorted
        ]
        ax.bar(langs_sorted, [asr_by_lang[l] for l in langs_sorted], color=bar_colors)
        ax.set_ylabel("Attack Success Rate")
        ax.set_ylim(0, 1)
        ax.set_title(
            "MultiJail Dataset: ASR by Language\n(LLaMA-3.1-8B-Instruct, colored by resource level)"
        )
        ax.set_xlabel("Language")
        for lvl, c in colors_map.items():
            ax.bar([], [], color=c, label=lvl)
        ax.legend(title="Resource")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_ASR_by_language_bar.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating ASR-by-language plot: {e}")
    plt.close()

# Plot 2: ASR by resource level
try:
    asr_by_res = mj.get("asr_by_resource", {})
    if asr_by_res:
        fig, ax = plt.subplots(figsize=(6, 4))
        levels = ["high", "medium", "low"]
        vals = [asr_by_res.get(l, 0) for l in levels]
        ax.bar(levels, vals, color=[colors_map[l] for l in levels])
        ax.set_ylabel("Attack Success Rate")
        ax.set_ylim(0, 1)
        ax.set_xlabel("Resource Level")
        ax.set_title(
            "MultiJail Dataset: ASR by Language Resource Level\n(LLaMA-3.1-8B-Instruct baseline)"
        )
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_ASR_by_resource_bar.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating ASR-by-resource plot: {e}")
    plt.close()

# Plot 3: Layer-wise semantic alignment
try:
    align = mj.get("layer_semantic_alignment", [])
    if align:
        align = np.array(align)
        best_layer = int(mj.get("best_bottleneck_layer", int(np.argmax(align))))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(range(len(align)), align, marker="o", markersize=3, color="#4c72b0")
        ax.axvline(
            best_layer, color="r", linestyle="--", label=f"Best layer={best_layer}"
        )
        ax.set_xlabel("Transformer Layer")
        ax.set_ylabel("Mean cosine similarity (non-en vs en)")
        ax.set_title(
            "MultiJail Dataset: Cross-lingual Semantic Alignment per Layer\n(LLaMA-3.1-8B-Instruct hidden states)"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_layer_semantic_alignment_curve.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating layer alignment plot: {e}")
    plt.close()

# Plot 4: Overall ASR summary comparison
try:
    overall = mj.get("overall_asr", None)
    asr_by_res = mj.get("asr_by_resource", {})
    if overall is not None and asr_by_res:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = ["overall"] + list(asr_by_res.keys())
        vals = [overall] + [asr_by_res[k] for k in asr_by_res.keys()]
        bar_colors = ["#555"] + [colors_map.get(k, "#888") for k in asr_by_res.keys()]
        ax.bar(labels, vals, color=bar_colors)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Attack Success Rate")
        ax.set_title(
            "MultiJail Dataset: Overall vs Per-Resource ASR\n(LLaMA-3.1-8B-Instruct)"
        )
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_overall_vs_resource_ASR.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating overall summary plot: {e}")
    plt.close()

# Plot 5: Refusal count by language (from predictions)
try:
    preds = mj.get("predictions", [])
    if preds:
        from collections import defaultdict

        succ = defaultdict(int)
        tot = defaultdict(int)
        for p in preds:
            lg = p["language"]
            tot[lg] += 1
            if p["attack_success"]:
                succ[lg] += 1
        langs_sorted = sorted(tot.keys(), key=lambda l: (RESOURCE_LEVEL.get(l, "z"), l))
        succ_vals = [succ[l] for l in langs_sorted]
        ref_vals = [tot[l] - succ[l] for l in langs_sorted]
        x = np.arange(len(langs_sorted))
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x, succ_vals, label="Attack Success", color="#c44e52")
        ax.bar(x, ref_vals, bottom=succ_vals, label="Refused", color="#4c72b0")
        ax.set_xticks(x)
        ax.set_xticklabels(langs_sorted)
        ax.set_ylabel("Number of Prompts")
        ax.set_xlabel("Language")
        ax.set_title(
            "MultiJail Dataset: Attack Success vs Refusal Counts by Language\n(Bottom: Success, Top: Refused)"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_success_vs_refusal_counts.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating success/refusal counts plot: {e}")
    plt.close()

print("Done. Plots saved to:", working_dir)
if mj:
    print(f"Overall ASR: {mj.get('overall_asr', 'N/A')}")
    print(f"ASR by resource: {mj.get('asr_by_resource', {})}")
