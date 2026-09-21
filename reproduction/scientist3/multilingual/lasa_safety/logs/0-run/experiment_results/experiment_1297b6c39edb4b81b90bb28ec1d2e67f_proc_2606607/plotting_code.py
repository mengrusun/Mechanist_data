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

mj = experiment_data.get("sampling_strategy", {}).get("MultiJail", {})
runs = mj.get("runs", {})
cfg_names = list(runs.keys())
mnt = mj.get("max_new_tokens", "NA")

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

# Plot 1: Overall ASR across sampling configs (any vs mean)
try:
    overall = mj.get("overall_asr_per_cfg", {})
    if overall:
        names = list(overall.keys())
        any_vals = [overall[c]["any"] for c in names]
        mean_vals = [overall[c]["mean"] for c in names]
        x = np.arange(len(names))
        w = 0.35
        plt.figure(figsize=(8, 4.5))
        plt.bar(x - w / 2, any_vals, w, label="any-of-N", color="#4c72b0")
        plt.bar(x + w / 2, mean_vals, w, label="mean-of-N", color="#dd8452")
        plt.xticks(x, names, rotation=20, ha="right")
        plt.ylabel("Overall ASR")
        plt.ylim(0, 1)
        plt.title(
            f"MultiJail Dataset: Overall ASR across sampling strategies\n(max_new_tokens={mnt})"
        )
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_overall_ASR_sampling_comparison.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating overall ASR plot: {e}")
    plt.close()

# Plot 2: Per-language ASR heatmap across sampling configs
try:
    all_langs = set()
    for cname in cfg_names:
        all_langs.update(runs[cname].get("asr_any_by_language", {}).keys())
    langs_sorted = sorted(all_langs, key=lambda l: (RESOURCE_LEVEL.get(l, "z"), l))
    if langs_sorted and cfg_names:
        mat = np.zeros((len(cfg_names), len(langs_sorted)))
        for i, cname in enumerate(cfg_names):
            d = runs[cname].get("asr_any_by_language", {})
            for j, lg in enumerate(langs_sorted):
                mat[i, j] = d.get(lg, 0.0)
        fig, ax = plt.subplots(figsize=(10, 4))
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(langs_sorted)))
        ax.set_xticklabels(
            [f"{l}\n({RESOURCE_LEVEL.get(l,'?')})" for l in langs_sorted]
        )
        ax.set_yticks(np.arange(len(cfg_names)))
        ax.set_yticklabels(cfg_names)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{mat[i,j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color="black",
                )
        plt.colorbar(im, ax=ax, label="ASR (any-of-N)")
        ax.set_title(
            f"MultiJail Dataset: Per-language ASR heatmap\n(rows=sampling config, cols=language)"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_per_language_ASR_heatmap.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating heatmap: {e}")
    plt.close()

# Plot 3: ASR by resource level across sampling configs
try:
    levels = ["high", "medium", "low"]
    if cfg_names:
        x = np.arange(len(levels))
        n_cfg = len(cfg_names)
        w = 0.8 / max(n_cfg, 1)
        plt.figure(figsize=(9, 4.5))
        for i, cname in enumerate(cfg_names):
            vals = [
                runs[cname].get("asr_any_by_resource", {}).get(lvl, 0.0)
                for lvl in levels
            ]
            plt.bar(x + (i - n_cfg / 2 + 0.5) * w, vals, w, label=cname)
        plt.xticks(x, levels)
        plt.ylabel("ASR (any-of-N)")
        plt.ylim(0, 1)
        plt.title("MultiJail Dataset: ASR by resource level across sampling strategies")
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_ASR_by_resource_level.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating resource-level plot: {e}")
    plt.close()

# Plot 4: Best config per-language ASR (any vs mean)
try:
    best_cfg = mj.get("best_cfg", None)
    if best_cfg and best_cfg in runs:
        br = runs[best_cfg]
        any_d = br.get("asr_any_by_language", {})
        mean_d = br.get("asr_mean_by_language", {})
        langs = sorted(any_d.keys(), key=lambda l: (RESOURCE_LEVEL.get(l, "z"), l))
        x = np.arange(len(langs))
        w = 0.35
        plt.figure(figsize=(10, 4.5))
        plt.bar(
            x - w / 2, [any_d[l] for l in langs], w, label="any-of-N", color="#4c72b0"
        )
        plt.bar(
            x + w / 2,
            [mean_d.get(l, 0) for l in langs],
            w,
            label="mean-of-N",
            color="#dd8452",
        )
        plt.xticks(x, [f"{l}\n({RESOURCE_LEVEL.get(l,'?')})" for l in langs])
        plt.ylabel("ASR")
        plt.ylim(0, 1)
        plt.title(
            f"MultiJail Dataset: Per-language ASR for best config ({best_cfg})\nLeft bar: any-of-N, Right bar: mean-of-N"
        )
        plt.legend()
        plt.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "MultiJail_best_config_per_language_ASR.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating best config plot: {e}")
    plt.close()

# Print metrics summary
try:
    print("Overall ASR per sampling config:")
    for c, v in mj.get("overall_asr_per_cfg", {}).items():
        print(f"  {c}: any={v['any']:.3f}, mean={v['mean']:.3f}")
    print(f"Best config: {mj.get('best_cfg', 'NA')}")
except Exception as e:
    print(f"Error printing metrics: {e}")
