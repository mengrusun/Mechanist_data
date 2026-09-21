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

mj = experiment_data.get("max_new_tokens", {}).get("MultiJail", {})
hp_values = mj.get("hyperparam_values", [])
runs = mj.get("runs", {})
overall_asr_per_hp = mj.get("overall_asr_per_hp", {})
best_hp = mj.get("best_hp", None)

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

# Plot 1: Overall ASR vs max_new_tokens
try:
    plt.figure(figsize=(6, 4))
    xs = list(hp_values)
    ys = [overall_asr_per_hp.get(x, overall_asr_per_hp.get(str(x), 0)) for x in xs]
    plt.plot(xs, ys, marker="o")
    plt.xlabel("max_new_tokens")
    plt.ylabel("Overall ASR")
    plt.title("MultiJail: Overall ASR vs max_new_tokens\n(Hyperparameter Sweep)")
    plt.xticks(xs)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "MultiJail_overall_ASR_vs_max_new_tokens.png"),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Grouped bar chart - ASR by language across all max_new_tokens
try:
    plt.figure(figsize=(12, 5))
    all_langs = set()
    for k, run in runs.items():
        all_langs.update(run.get("asr_by_language", {}).keys())
    langs_sorted = sorted(all_langs, key=lambda l: (RESOURCE_LEVEL.get(l, "z"), l))
    n_hp = len(runs)
    width = 0.8 / max(n_hp, 1)
    x = np.arange(len(langs_sorted))
    for i, (k, run) in enumerate(sorted(runs.items(), key=lambda kv: int(kv[0]))):
        asr_lang = run.get("asr_by_language", {})
        vals = [asr_lang.get(l, 0) for l in langs_sorted]
        plt.bar(x + i * width - 0.4 + width / 2, vals, width, label=f"mnt={k}")
    plt.xticks(x, langs_sorted)
    plt.ylabel("Attack Success Rate")
    plt.xlabel("Language")
    plt.ylim(0, 1)
    plt.title(
        "MultiJail: ASR by Language across max_new_tokens\n(Grouped by Hyperparameter Value)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "MultiJail_ASR_by_language_grouped.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: ASR by resource level across max_new_tokens
try:
    plt.figure(figsize=(7, 4))
    levels = ["high", "medium", "low"]
    for lvl in levels:
        xs, ys = [], []
        for k, run in sorted(runs.items(), key=lambda kv: int(kv[0])):
            v = run.get("asr_by_resource", {}).get(lvl)
            if v is not None:
                xs.append(int(k))
                ys.append(v)
        if xs:
            plt.plot(xs, ys, marker="o", label=f"resource={lvl}")
    plt.xlabel("max_new_tokens")
    plt.ylabel("Attack Success Rate")
    plt.title(
        "MultiJail: ASR by Resource Level vs max_new_tokens\n(High/Medium/Low Resource Languages)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "MultiJail_ASR_by_resource_vs_mnt.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Layer semantic alignment
try:
    lsa = mj.get("layer_semantic_alignment", [])
    best_layer = mj.get("best_bottleneck_layer", None)
    if lsa:
        plt.figure(figsize=(8, 4))
        plt.plot(range(len(lsa)), lsa, marker="o", markersize=3)
        if best_layer is not None:
            plt.axvline(
                best_layer, color="r", linestyle="--", label=f"best layer={best_layer}"
            )
            plt.legend()
        plt.xlabel("Transformer Layer")
        plt.ylabel("Mean Cosine Similarity (non-en vs en)")
        plt.title(
            "MultiJail: Cross-lingual Semantic Alignment per Layer\n(LLaMA-3.1-8B-Instruct Hidden States)"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_layer_semantic_alignment.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: ASR by language for the best hyperparameter
try:
    if best_hp is not None:
        best_run = runs.get(str(best_hp), {})
        asr_lang = best_run.get("asr_by_language", {})
        if asr_lang:
            langs_sorted = sorted(
                asr_lang.keys(), key=lambda l: (RESOURCE_LEVEL.get(l, "z"), l)
            )
            colors_map = {"high": "#4c72b0", "medium": "#dd8452", "low": "#c44e52"}
            bar_colors = [
                colors_map.get(RESOURCE_LEVEL.get(l, ""), "#888") for l in langs_sorted
            ]
            plt.figure(figsize=(9, 4))
            plt.bar(langs_sorted, [asr_lang[l] for l in langs_sorted], color=bar_colors)
            plt.ylabel("Attack Success Rate")
            plt.ylim(0, 1)
            plt.title(
                f"MultiJail: ASR by Language (Best Hyperparameter: max_new_tokens={best_hp})\nBars Colored by Resource Level"
            )
            for lvl, c in colors_map.items():
                plt.bar([], [], color=c, label=lvl)
            plt.legend(title="resource")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, f"MultiJail_ASR_by_language_best_mnt{best_hp}.png"
                ),
                dpi=120,
            )
            plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print evaluation metrics
print("Overall ASR per max_new_tokens:")
for k, v in sorted(
    overall_asr_per_hp.items(),
    key=lambda kv: int(kv[0]) if str(kv[0]).isdigit() else kv[0],
):
    print(f"  max_new_tokens={k}: ASR={v:.4f}")
print(f"Best max_new_tokens: {best_hp}")
if mj.get("best_bottleneck_layer") is not None:
    print(f'Best semantic bottleneck layer: {mj["best_bottleneck_layer"]}')
