import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2026-07-13_23-35-02_lasa_safety_attempt_2/logs/0-run/experiment_results/experiment_24addaf7046642ef8ebd278706f34c35_proc_2175194/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full_path = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        if not os.path.exists(full_path):
            full_path = p
        ed = np.load(full_path, allow_pickle=True).item()
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

# Collect MultiJail dicts across all runs
mj_list = [
    ed.get("max_new_tokens", {}).get("MultiJail", {}) for ed in all_experiment_data
]
mj_list = [m for m in mj_list if m]


def sem(a):
    a = np.asarray(a, dtype=float)
    if len(a) <= 1:
        return 0.0
    return a.std(ddof=1) / np.sqrt(len(a))


# Union of hp values
hp_values = []
for m in mj_list:
    for hp in m.get("hyperparam_values", []):
        if hp not in hp_values:
            hp_values.append(hp)
hp_values = sorted(hp_values, key=lambda x: int(x) if str(x).isdigit() else x)

# ---- Plot 1: aggregated Overall ASR vs max_new_tokens ----
try:
    means, sems = [], []
    for hp in hp_values:
        vals = []
        for m in mj_list:
            d = m.get("overall_asr_per_hp", {})
            v = d.get(
                hp, d.get(str(hp), d.get(int(hp) if str(hp).isdigit() else hp, None))
            )
            if v is not None:
                vals.append(v)
        if vals:
            means.append(np.mean(vals))
            sems.append(sem(vals))
        else:
            means.append(np.nan)
            sems.append(0)
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        hp_values,
        means,
        yerr=sems,
        marker="o",
        capsize=4,
        label=f"Mean ± SE (n={len(mj_list)})",
    )
    plt.xlabel("max_new_tokens")
    plt.ylabel("Overall ASR")
    plt.title(
        "MultiJail: Aggregated Overall ASR vs max_new_tokens\n(Mean ± SE across runs)"
    )
    plt.xticks(hp_values)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "MultiJail_aggregated_overall_ASR_vs_mnt.png"),
        dpi=120,
    )
    plt.close()
    print("Aggregated Overall ASR per max_new_tokens:")
    for hp, mu, se in zip(hp_values, means, sems):
        print(f"  mnt={hp}: {mu:.4f} ± {se:.4f}")
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# ---- Plot 2: ASR by resource level vs max_new_tokens (aggregated) ----
try:
    plt.figure(figsize=(7, 4))
    levels = ["high", "medium", "low"]
    for lvl in levels:
        xs, mus, ses = [], [], []
        for hp in hp_values:
            vals = []
            for m in mj_list:
                runs = m.get("runs", {})
                run = runs.get(str(hp), runs.get(hp, {}))
                v = run.get("asr_by_resource", {}).get(lvl)
                if v is not None:
                    vals.append(v)
            if vals:
                xs.append(hp)
                mus.append(np.mean(vals))
                ses.append(sem(vals))
        if xs:
            plt.errorbar(
                xs, mus, yerr=ses, marker="o", capsize=3, label=f"{lvl} (Mean±SE)"
            )
    plt.xlabel("max_new_tokens")
    plt.ylabel("Attack Success Rate")
    plt.title(
        "MultiJail: ASR by Resource Level vs max_new_tokens\n(Aggregated Mean ± SE across runs)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "MultiJail_aggregated_ASR_by_resource_vs_mnt.png"),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# ---- Plot 3: ASR by language at best hp (aggregated) ----
try:
    # determine most common best_hp
    best_hps = [m.get("best_hp") for m in mj_list if m.get("best_hp") is not None]
    if best_hps:
        best_hp = max(set(best_hps), key=best_hps.count)
    else:
        best_hp = hp_values[-1] if hp_values else None

    if best_hp is not None:
        # collect language ASR values across runs at best_hp
        lang_vals = {}
        for m in mj_list:
            runs = m.get("runs", {})
            run = runs.get(str(best_hp), runs.get(best_hp, {}))
            for lang, v in run.get("asr_by_language", {}).items():
                lang_vals.setdefault(lang, []).append(v)
        if lang_vals:
            langs_sorted = sorted(
                lang_vals.keys(), key=lambda l: (RESOURCE_LEVEL.get(l, "z"), l)
            )
            mus = [np.mean(lang_vals[l]) for l in langs_sorted]
            ses = [sem(lang_vals[l]) for l in langs_sorted]
            colors_map = {"high": "#4c72b0", "medium": "#dd8452", "low": "#c44e52"}
            bar_colors = [
                colors_map.get(RESOURCE_LEVEL.get(l, ""), "#888") for l in langs_sorted
            ]
            plt.figure(figsize=(9, 4))
            plt.bar(
                langs_sorted,
                mus,
                yerr=ses,
                color=bar_colors,
                capsize=4,
                error_kw={"ecolor": "black"},
            )
            plt.ylabel("Attack Success Rate")
            plt.ylim(0, 1)
            plt.title(
                f"MultiJail: Aggregated ASR by Language (best mnt={best_hp})\nMean ± SE across runs, colored by resource level"
            )
            for lvl, c in colors_map.items():
                plt.bar([], [], color=c, label=lvl)
            plt.legend(title="resource")
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir,
                    f"MultiJail_aggregated_ASR_by_language_best_mnt{best_hp}.png",
                ),
                dpi=120,
            )
            plt.close()
            print(f"Best max_new_tokens (mode across runs): {best_hp}")
            for l, mu, se in zip(langs_sorted, mus, ses):
                print(f"  {l}: {mu:.4f} ± {se:.4f}")
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# ---- Plot 4: Layer semantic alignment aggregated ----
try:
    lsa_list = [
        np.asarray(m.get("layer_semantic_alignment", []), dtype=float)
        for m in mj_list
        if len(m.get("layer_semantic_alignment", [])) > 0
    ]
    if lsa_list:
        min_len = min(len(a) for a in lsa_list)
        stacked = np.stack([a[:min_len] for a in lsa_list], axis=0)
        mus = stacked.mean(axis=0)
        ses = (
            stacked.std(axis=0, ddof=1) / np.sqrt(stacked.shape[0])
            if stacked.shape[0] > 1
            else np.zeros(min_len)
        )
        xs = np.arange(min_len)
        plt.figure(figsize=(8, 4))
        plt.plot(
            xs, mus, marker="o", markersize=3, label=f"Mean (n={stacked.shape[0]})"
        )
        plt.fill_between(xs, mus - ses, mus + ses, alpha=0.3, label="± SE")
        best_layers = [
            m.get("best_bottleneck_layer")
            for m in mj_list
            if m.get("best_bottleneck_layer") is not None
        ]
        if best_layers:
            bl = int(np.round(np.mean(best_layers)))
            plt.axvline(bl, color="r", linestyle="--", label=f"mean best layer={bl}")
        plt.xlabel("Transformer Layer")
        plt.ylabel("Mean Cosine Similarity (non-en vs en)")
        plt.title(
            "MultiJail: Aggregated Cross-lingual Semantic Alignment per Layer\n(Mean ± SE across runs)"
        )
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "MultiJail_aggregated_layer_semantic_alignment.png"
            ),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

print(f"Number of runs aggregated: {len(mj_list)}")
