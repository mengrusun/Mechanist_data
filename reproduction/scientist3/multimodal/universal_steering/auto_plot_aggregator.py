I notice the response is being cut off mid-string when my message exceeds a length limit. Let me produce a shorter, complete, and syntactically-correct aggregator script.

```python
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300,
})

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

BASELINE_NPY = "experiment_results/experiment_cd4d4c99f638482084314c283335f277_proc_1647063/experiment_data.npy"
RESEARCH_NPY = "experiment_results/experiment_cfe801e35b3c47389fe39e3e284c079f_proc_2170206/experiment_data.npy"
ABLATION_NPY = "experiment_results/experiment_9bceab78e52143ac8d811aaa7246b296_proc_4033862/experiment_data.npy"


def safe_load(p):
    try:
        return np.load(p, allow_pickle=True).item()
    except Exception as e:
        print("[WARN] load failed:", p, e)
        return {}


def pretty(s):
    return str(s).replace("_", " ")


baseline_data = safe_load(BASELINE_NPY)
research_data = safe_load(RESEARCH_NPY)
ablation_data = safe_load(ABLATION_NPY)


# ---------- FIGURE 1: Baseline alpha tuning ----------
try:
    cb = baseline_data.get("alpha_tuning", {}).get("concept_benchmark", {})
    per_alpha = cb.get("per_alpha", {})
    baseline_rate = cb.get("overall_baseline_success_rate", None)
    if per_alpha:
        keys = list(per_alpha.keys())
        alphas_raw = [per_alpha[k]["alpha"] for k in keys]
        order = np.argsort(alphas_raw)
        alphas = [alphas_raw[i] for i in order]
        keys = [keys[i] for i in order]
        succ = [per_alpha[k]["overall_steering_success_rate"] for k in keys]
        vloss = [per_alpha[k]["val_loss"] for k in keys]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))
        ax = axes[0]
        ax.plot(alphas, succ, marker="o", lw=2, color="steelblue", label="RFM Steering")
        if baseline_rate is not None:
            ax.axhline(baseline_rate, color="gray", ls="--", label="Unsteered Baseline")
        ax.set_xlabel("Steering Strength (alpha)")
        ax.set_ylabel("Overall Steering Success Rate")
        ax.set_title("(a) Success Rate vs Steering Strength")
        ax.set_ylim(0, 1.05); ax.legend()

        ax = axes[1]
        ax.plot(alphas, vloss, marker="s", lw=2, color="crimson",
                label="Validation Loss (1 minus success)")
        ax.set_xlabel("Steering Strength (alpha)")
        ax.set_ylabel("Validation Loss")
        ax.set_title("(b) Validation Loss vs Steering Strength")
        ax.legend()
        fig.suptitle("Figure 1: Alpha Tuning on GPT-4o Concept Benchmark (Llama-3.1-8B)", y=1.03)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig1_alpha_tuning.png"), bbox_inches="tight")
        plt.close(fig)
except Exception as e:
    print("[ERR] fig1:", e); plt.close("all")


# ---------- FIGURE 2: Baseline per-class success by alpha ----------
try:
    cb = baseline_data.get("alpha_tuning", {}).get("concept_benchmark", {})
    per_alpha = cb.get("per_alpha", {})
    if per_alpha:
        alphas = sorted({per_alpha[k]["alpha"] for k in per_alpha})
        okeys = ["alpha_" + str(a) for a in alphas if ("alpha_" + str(a)) in per_alpha]
        classes = sorted({r["class"] for r in per_alpha[okeys[0]]["per_concept"]})
        pcls = [pretty(c).capitalize() for c in classes]

        base_by = {c: [] for c in classes}
        for r in per_alpha[okeys[0]]["per_concept"]:
            base_by[r["class"]].append(r["baseline_success"] / max(r["total"], 1))
        base_means = [np.mean(base_by[c]) for c in classes]

        fig, ax = plt.subplots(figsize=(10, 4.8))
        xs = np.arange(len(classes))
        w = 0.8 / (len(okeys) + 1)
        ax.bar(xs - 0.4 + w / 2, base_means, width=w, label="Unsteered",
               color="lightgray", edgecolor="k")
        cmap = plt.cm.viridis(np.linspace(0.25, 0.85, len(okeys)))
        for i, k in enumerate(okeys):
            cs = {c: [] for c in classes}
            for r in per_alpha[k]["per_concept"]:
                cs[r["class"]].append(r["steered_success"] / max(r["total"], 1))
            means = [np.mean(cs[c]) for c in classes]
            lbl = "RFM Steered (alpha=" + str(per_alpha[k]["alpha"]) + ")"
            ax.bar(xs - 0.4 + w * (i + 1.5), means, width=w, label=lbl, color=cmap[i])
        ax.set_xticks(xs); ax.set_xticklabels(pcls, rotation=15)
        ax.set_ylabel("Success Rate"); ax.set_ylim(0, 1.05)
        ax.set_title("Figure 2: Per-Class Steering Success on Concept Benchmark")
        ax.legend(ncol=min(4, len(okeys) + 1), fontsize=10, loc="upper left")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig2_per_class_success.png"), bbox_inches="tight")
        plt.close(fig)
except Exception as e:
    print("[ERR] fig2:", e); plt.close("all")


# ---------- FIGURE 3: Layer x Alpha heatmap ----------
try:
    cb = research_data.get("layer_alpha_sweep", {}).get("concept_benchmark", {})
    per_config = cb.get("per_config", {})
    fkeys = [k for k in per_config if per_config[k].get("mode") == "fixed"]
    layers = sorted({per_config[k]["layer"] for k in fkeys})
    falphas = sorted({per_config[k]["alpha"] for k in fkeys})

    if fkeys and layers and falphas:
        gs = np.full((len(layers), len(falphas)), np.nan)
        go = np.full((len(layers), len(falphas)), np.nan)
        for i, L in enumerate(layers):
            for j, a in enumerate(falphas):
                k = "L" + str(L) + "_a" + str(a) + "_fixed"
                if k in per_config:
                    gs[i, j] = per_config[k]["success_rate"]
                    go[i, j] = per_config[k]["off_target_rate"]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
        for ax, grid, title, cmap in [
            (axes[0], gs, "(a) On-Target Success Rate", "viridis"),
            (axes[1], go, "(b) Off-Target (Distractor) Rate", "magma"),
        ]:
            im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            ax.set_xticks(range(len(falphas))); ax.set_xticklabels(falphas)
            ax.set_yticks(range(len(layers))); ax.set_yticklabels(layers)
            ax.set_xlabel("Steering Strength (alpha)")
            ax.set_ylabel("Injection Layer Index")
            ax.set_title(title)
            for i in range(grid.shape[0]):
                for j in range(grid.shape[1]):
                    v = grid[i, j]
                    if not np.isnan(v):
                        ax.text(j, i, "{:.2f}".format(v), ha="center", va="center",
                                color="white" if v < 0.6 else "black", fontsize=9)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle("Figure 3: Joint Layer x Alpha Sweep on Concept Benchmark", y=1.03)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig3_layer_alpha_heatmap.png"), bbox_inches="tight")
        plt.close(fig)
except Exception as e:
    print("[ERR] fig3:", e); plt.close("all")


# ---------- FIGURE 4: Methods comparison ----------
try:
    cb = research_data.get("layer_alpha_sweep", {}).get("concept_benchmark", {})
    per_config = cb.get("per_config", {})
    baseline = cb.get("baseline", {})
    prompt_baseline = cb.get("prompt_baseline", {})

    fkeys = [k for k in per_config if per_config[k].get("mode") == "fixed"]
    akeys = [k for k in per_config if per_config[k].get("mode") == "adaptive"]
    bfix = max(fkeys, key=lambda k: per_config[k]["success_rate"]) if fkeys else None
    bada = max(akeys, key=lambda k: per_config[k]["success_rate"]) if akeys else None

    names, succ, off = [], [], []
    if baseline:
        names.append("Unsteered")
        succ.append(baseline.get("success_rate", 0.0))
        off.append(baseline.get("off_target_rate", 0.0))
    if prompt_baseline:
        names.append("Prompt Baseline")
        succ.append(prompt_baseline.get("success_rate", 0.0))
        off.append(prompt_baseline.get("off_target_rate", 0.0))
    if bfix:
        lb = "RFM Fixed (L={}, a={})".format(per_config[bfix]["layer"], per_config[bfix]["alpha"])
        names.append(lb)
        succ.append(per_config[bfix]["success_rate"])
        off.append(per_config[bfix]["off_target_rate"])
    if bada:
        lb = "RFM Adaptive (L={}, a={})".format(per_config[bada]["layer"], per_config[bada]["alpha"])
        names.append(lb)
        succ.append(per_config[bada]["success_rate"])
        off.append(per_config[bada]["off_target_rate"])

    if names:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        x = np.arange(len(names))
        w = 0.38
        ax.bar(x - w / 2, succ, w, color="steelblue", label="On-target Success")
        ax.bar(x + w / 2, off, w, color="salmon", label="Off-target (Distractor)")
        ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10)
        ax.set_yl