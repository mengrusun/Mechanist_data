It seems the script is being saved with the triple backtick fence included. I'll deliver a shorter, self-contained script that fits cleanly in one message.

```python
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 15,
    "legend.fontsize": 11, "xtick.labelsize": 12, "ytick.labelsize": 12,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300,
})

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

RESOURCE_LEVEL = {"en":"high","zh":"high","it":"high","vi":"medium","ar":"medium",
                  "ko":"medium","th":"low","bn":"low","sw":"low","jv":"low"}
RES_COLORS = {"high":"#4c72b0","medium":"#dd8452","low":"#c44e52"}
LANG_FULLNAME = {"en":"English","zh":"Chinese","it":"Italian","vi":"Vietnamese",
                 "ar":"Arabic","ko":"Korean","th":"Thai","bn":"Bengali",
                 "sw":"Swahili","jv":"Javanese"}

NPY_PATH = "experiment_results/experiment_8cd8a2bcd5aa4ec788cbae8487a9e3d0_proc_1777750/experiment_data.npy"

experiment_data = {}
try:
    experiment_data = np.load(NPY_PATH, allow_pickle=True).item()
    print(f"Loaded data from {NPY_PATH}")
except Exception as e:
    print(f"Could not load {NPY_PATH}: {e}")

mj = experiment_data.get("max_new_tokens", {}).get("MultiJail", {})
hp_values = mj.get("hyperparam_values", [])
runs = mj.get("runs", {})
overall_asr_per_hp = mj.get("overall_asr_per_hp", {})
best_hp = mj.get("best_hp", None)
lsa = mj.get("layer_semantic_alignment", [])
best_layer = mj.get("best_bottleneck_layer", None)


def _get_hp_val(k):
    try: return int(k)
    except Exception: return k


def _sorted_runs():
    return sorted(runs.items(), key=lambda kv: _get_hp_val(kv[0]))


# FIGURE 1: main results overview
try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    ax = axes[0]
    if hp_values and overall_asr_per_hp:
        xs = list(hp_values)
        ys = [overall_asr_per_hp.get(x, overall_asr_per_hp.get(str(x), 0)) for x in xs]
        ax.plot(xs, ys, marker="o", linewidth=2, color="#333", label="Overall ASR")
        ax.set_xticks(xs)
        ax.set_xlabel("Max new tokens"); ax.set_ylabel("Overall ASR")
        ax.set_title("(a) Overall Attack Success Rate vs Generation Length")
        ax.grid(True, alpha=0.3)
        if best_hp is not None:
            ax.axvline(int(best_hp), color="red", linestyle="--", alpha=0.6,
                       label=f"Best setting: {best_hp}")
        ax.legend()
    ax = axes[1]
    for lvl in ["high","medium","low"]:
        xs, ys = [], []
        for k, run in _sorted_runs():
            v = run.get("asr_by_resource", {}).get(lvl)
            if v is not None:
                xs.append(_get_hp_val(k)); ys.append(v)
        if xs:
            ax.plot(xs, ys, marker="o", linewidth=2, color=RES_COLORS[lvl],
                    label=f"{lvl}-resource languages")
    ax.set_xlabel("Max new tokens"); ax.set_ylabel("Attack Success Rate")
    ax.set_title("(b) ASR by Language Resource Tier")
    ax.grid(True, alpha=0.3); ax.legend(title="Resource tier")
    ax = axes[2]
    if len(lsa) > 0:
        ax.plot(range(len(lsa)), lsa, marker="o", markersize=3, linewidth=1.6,
                color="#2a9d8f", label="Mean cosine similarity")
        if best_layer is not None:
            ax.axvline(best_layer, color="red", linestyle="--",
                       label=f"Semantic bottleneck: layer {best_layer}")
        ax.set_xlabel("Transformer layer index")
        ax.set_ylabel("Mean cosine similarity (non-en vs en)")
        ax.set_title("(c) Cross-Lingual Semantic Alignment per Layer")
        ax.grid(True, alpha=0.3); ax.legend()
    fig.suptitle("LASA Baseline on MultiJail: Safety and the Semantic Bottleneck in LLaMA-3.1-8B-Instruct",
                 fontsize=16, y=1.04)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1-main-results-overview.png"), bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print(f"Error Figure 1: {e}"); plt.close("all")


# FIGURE 2: ASR by language at best hyperparameter
try:
    if best_hp is not None:
        best_run = runs.get(str(best_hp), runs.get(best_hp, {}))
        asr_lang = best_run.get("asr_by_language", {})
        if asr_lang:
            langs_sorted = sorted(asr_lang.keys(), key=lambda l:(RESOURCE_LEVEL.get(l,"z"), l))
            labels = [LANG_FULLNAME.get(l, l) for l in langs_sorted]
            vals = [asr_lang[l] for l in langs_sorted]
            colors = [RES_COLORS.get(RESOURCE_LEVEL.get(l,""), "#888") for l in langs_sorted]
            fig, ax = plt.subplots(figsize=(11, 4.8))
            bars = ax.bar(labels, vals, color=colors, edgecolor="black", linewidth=0.6)
            for b, v in zip(bars, vals):
                ax.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=10)
            top = max(0.15, min(1.0, (max(vals) if vals else 0)*1.3 + 0.05))
            ax.set_ylim(0, top)
            ax.set_ylabel("Attack Success Rate"); ax.set_xlabel("Language")
            ax.set_title(f"Cross-Lingual Safety Gap at Best Setting (max new tokens = {best_hp})")
            plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
            for lvl, c in RES_COLORS.items():
                ax.bar([], [], color=c, label=f"{lvl}-resource")
            ax.legend(title="Resource tier", loc="upper left")
            fig.tight_layout()
            fig.savefig(os.path.join(FIG_DIR, "fig2-asr-by-language-best-hp.png"), bbox_inches="tight")
            plt.close(fig)
except Exception as e:
    print(f"Error Figure 2: {e}"); plt.close("all")


# FIGURE 3: language-wise sweep, grouped bars + heatmap
try:
    all_langs = set()
    for _, run in runs.items():
        all_langs.update(run.get("asr_by_language", {}).keys())
    langs_sorted = sorted(all_langs, key=lambda l:(RESOURCE_LEVEL.get(l,"z"), l))
    if langs_sorted and runs:
        sorted_run_items = _sorted_runs()
        hp_keys = [k for k, _ in sorted_run_items]
        fig, axes = plt.subplots(1, 2, figsize=(17, 5.2))
        ax = axes[0]
        n_hp = len(sorted_run_items); width = 0.8/max(n_hp,1)
        x = np.arange(len(langs_sorted))
        cmap_colors = plt.cm.viridis(np.linspace(0.15, 0.85, max(n_hp,1)))
        for i,(k,run) in enumerate(sorted_run_items):
            vals = [run.get("asr_by_language",{}).get(l,0) for l in langs_sorted]
            ax.bar(x + i*width - 0.4 + width/2, vals, width,
                   label=f"{k} tokens", color=cmap_colors[i], edgecolor="black", linewidth=0.4)
        ax.set_xticks(x); ax.set_xticklabels(langs_sorted)
        ax.set_ylabel("Attack Success Rate"); ax.set_xlabel("Language")
        ax.set_ylim(0,1); ax.set_title("(a) ASR by Language across Generation Lengths")
        ax.legend(title="Max new tokens", ncol=2, fontsize=10)
        ax = axes[1]
        matrix = np.zeros((len(hp_keys), len(langs_sorted)))
        for i,(k,run) in enumerate(sorted_run_items):
            for j,l in enumerate(langs_sorted):
                matrix[i,j] = run.get("asr_by_language",{}).get(l,0)
        vmax = max(0.5, float(matrix.max()) if matrix.size else 0.5)
        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_xticks(range(len(langs_sorted))); ax.set_xticklabels(langs_sorted)
        ax.set_yticks(range(len(hp_keys))); ax.set_yticklabels([str(k) for k in hp_keys])
        ax.set_xlabel("Language"); ax.set_ylabel("Max new tokens")
        ax.set_title("(b) ASR Heatmap across Settings and Languages")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                        color="black" if matrix[i,j] < 0.55*vmax else "white", fontsize=9)
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Attack Success Rate")
        fig.suptitle("Language-wise Attack Success Rate under Generation-Length Sweep",
                     fontsize=16, y=1.04)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig3-asr-per-language-sweep.png"), bbox_inches="tight")
        plt.close(fig)
except Exception as e:
    print(f"Error Figure 3: {e}"); plt.close("all")


# FIGURE 4: Detailed semantic bottleneck (line, gain, top-k bars)
try:
    if len(lsa) > 0:
        arr = np.asarray(lsa, dtype=float)
        n_layers = len(arr)
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.6))

        ax = axes[0]
        ax.plot(range(n_layers), arr, marker="o", markersize=3, linewidth=1.6, color="#264653")
        if best_layer is not None: