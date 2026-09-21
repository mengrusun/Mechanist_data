```python
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.dpi": 120, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

BASELINE_NPY = "experiment_results/experiment_7e415c6ca1794be6a2b188e975cd098f_proc_3987024/experiment_data.npy"
RESEARCH_NPY = "experiment_results/experiment_0ef9754eefd74f9cbc5627961de30956_proc_4119455/experiment_data.npy"
ABLATION_NPY = "experiment_results/experiment_2b284dbb699c4fd7aee5e4e373bdb1f8_proc_621193/experiment_data.npy"


def safe_load(path):
    try:
        return np.load(path, allow_pickle=True).item()
    except Exception as e:
        print("Load warning:", path, e)
        return {}


baseline_data = safe_load(BASELINE_NPY)
research_data = safe_load(RESEARCH_NPY)
ablation_data = safe_load(ABLATION_NPY)

B = baseline_data.get("gemma2_2b_gemmascope_res_16k", {})
R = research_data.get("gemma2_2b_gemmascope_res_16k_multi", {})
A = ablation_data.get("no_negative_contrastive_probes", {}).get("gemma2_2b_gemmascope_res_16k_multi", {})


# Figure 1: Method comparison by layer (Generative and Predictive)
try:
    summary = R.get("summary", {})
    layer_keys = [k for k in summary if k.startswith("L")]
    if "OVERALL" in summary:
        layer_keys = layer_keys + ["OVERALL"]
    if layer_keys:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        methods = [("np", "Neuronpedia", "#888888"),
                   ("ss", "Single-Shot GPT-5", "#f39c12"),
                   ("sage", "SAGE (Ours)", "#2980b9")]
        x = np.arange(len(layer_keys))
        w = 0.25
        for i, (m, label, c) in enumerate(methods):
            vals = [summary[k].get(m + "_gen", {}).get("mean", 0) for k in layer_keys]
            errs = [summary[k].get(m + "_gen", {}).get("ci", 0) for k in layer_keys]
            axes[0].bar(x + i * w - w, vals, w, yerr=errs, label=label, color=c, capsize=3)
        axes[0].set_xticks(x); axes[0].set_xticklabels(layer_keys)
        axes[0].set_ylabel("Generative Accuracy"); axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Generative Accuracy by Layer"); axes[0].legend()
        for i, (m, label, c) in enumerate(methods):
            vals = [summary[k].get(m + "_pred", {}).get("mean", 0) for k in layer_keys]
            errs = [summary[k].get(m + "_pred", {}).get("ci", 0) for k in layer_keys]
            axes[1].bar(x + i * w - w, vals, w, yerr=errs, label=label, color=c, capsize=3)
        axes[1].set_xticks(x); axes[1].set_xticklabels(layer_keys)
        axes[1].set_ylabel("Predictive Accuracy (Pearson r)")
        axes[1].set_title("Predictive Accuracy by Layer"); axes[1].legend()
        axes[1].axhline(0, color="k", lw=0.5)
        fig.suptitle("SAGE vs Baselines on Gemma-2-2B GemmaScope-Res-16k")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "fig1-method-comparison-by-layer.png"), bbox_inches="tight")
        plt.close()
except Exception as e:
    print("fig1 error:", e); plt.close()


# Figure 2: SAGE iteration refinement curve
try:
    per_feature = R.get("per_feature", {})
    summary = R.get("summary", {})
    max_iters = max((len(pf.get("iterations", [])) for pf in per_feature.values()), default=0)
    if max_iters > 0:
        iter_vals = [[] for _ in range(max_iters)]
        for pf in per_feature.values():
            its = pf.get("iterations", [])
            best_g = 0
            for i, it in enumerate(its):
                g = max((c.get("gen_acc", 0) for c in it.get("candidates", [])), default=0)
                best_g = max(best_g, g)
                iter_vals[i].append(best_g)
            for i in range(len(its), max_iters):
                iter_vals[i].append(best_g)
        means = [np.mean(v) if v else 0 for v in iter_vals]
        sems = [np.std(v) / np.sqrt(max(1, len(v))) if v else 0 for v in iter_vals]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(range(1, max_iters + 1), means, yerr=sems, marker="o",
                    color="#2980b9", lw=2, capsize=4, label="SAGE Best-so-far")
        overall = summary.get("OVERALL", {})
        np_mean = overall.get("np_gen", {}).get("mean")
        ss_mean = overall.get("ss_gen", {}).get("mean")
        if np_mean is not None:
            ax.axhline(np_mean, color="gray", ls="--", label="Neuronpedia (" + f"{np_mean:.2f}" + ")")
        if ss_mean is not None:
            ax.axhline(ss_mean, color="#f39c12", ls="--", label="Single-Shot GPT-5 (" + f"{ss_mean:.2f}" + ")")
        ax.set_xlabel("SAGE Iteration"); ax.set_ylabel("Generative Accuracy")
        ax.set_ylim(0, 1.05); ax.legend()
        ax.set_title("SAGE Iterative Refinement (mean and SEM across features)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "fig2-iteration-curve.png"), bbox_inches="tight")
        plt.close()
except Exception as e:
    print("fig2 error:", e); plt.close()


# Figure 3: Per-feature delta vs frequency and paired scatter
try:
    all_rows = R.get("all_rows", [])
    if all_rows:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        color_map = {5: "#1f77b4", 12: "#ff7f0e", 20: "#2ca02c"}
        for r in all_rows:
            L = r.get("L")
            freq = max(r.get("freq", 1e-6), 1e-6)
            axes[0].scatter(freq, r.get("sage_gen", 0) - r.get("np_gen", 0),
                            c=color_map.get(L, "gray"), alpha=0.75, s=45)
        axes[0].axhline(0, color="k", lw=0.5); axes[0].set_xscale("log")
        axes[0].set_xlabel("Feature Activation Frequency (log scale)")
        axes[0].set_ylabel("Delta Generative Accuracy (SAGE minus Neuronpedia)")
        axes[0].set_title("Per-Feature Improvement vs Activation Frequency")
        for L, c in color_map.items():
            if any(r.get("L") == L for r in all_rows):
                axes[0].scatter([], [], c=c, label="Layer " + str(L), s=45)
        axes[0].legend()

        np_g = [r.get("np_gen", 0) for r in all_rows]
        sage_g = [r.get("sage_gen", 0) for r in all_rows]
        Ls = [r.get("L") for r in all_rows]
        for L, c in color_map.items():
            xs = [n for n, l in zip(np_g, Ls) if l == L]
            ys = [s for s, l in zip(sage_g, Ls) if l == L]
            if xs:
                axes[1].scatter(xs, ys, c=c, alpha=0.75, s=45, label="Layer " + str(L))
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="y equals x")
        axes[1].set_xlabel("Neuronpedia Generative Accuracy")
        axes[1].set_ylabel("SAGE Generative Accuracy")
        axes[1].set_title("Per-Feature Paired Comparison")
        axes[1].set_xlim(-0.02, 1.05); axes[1].set_ylim(-0.02, 1.05); axes[1].legend()
        fig.suptitle("Where Does SAGE Help? Gemma-2-2B Multi-layer Analysis")
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "fig3-per-feature-analysis.png"), bbox_inches="tight")
        plt.close()
except Exception as e:
    print("fig3 error:", e); plt.close()


# Figure 4: Ablation - No Negative Contrastive Probes vs Full SAGE
try:
    a_summary = A.get("summary", {})
    r_summary = R.get("summary", {})
    layer_keys = [k for k in r_summary if k.startswith("L")]
    if a_summary and r_summary and layer_keys:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(layer_keys)); w = 0.35
        full_gen = [r_summary[k].get("sage_gen", {}).get("mean", 0) for k in layer_keys]
        abl_gen = [a_summary.get(k, {}).get("sage_gen", {}).get("mean", 0) for k in layer_keys]
        full_err = [r_summary[k].get("sage_gen", {}).get("ci", 0) for k in layer_keys]
        abl_err = [a_summary.get(k, {}).get("sage_gen", {}).get("ci", 0) for k in layer_keys]
        axes[0].bar(x - w/2, full_gen, w, yerr=full_err, capsize=3, label="Full SAGE", color="#2980b9")
        axes[0].bar(x + w/2, abl_gen, w, yerr=abl_err, capsize=3, label="SAGE without Negative Probes", color="#c0392b")
        axes[0].set_xticks(x); axes[0].set_xticklabels(layer_keys)
        axes[0].set_ylabel("Generative Accuracy"); axes[0].set_ylim(0, 1.05)
        axes[0].set_title("Generative Accuracy: Ablation vs Full SAGE"); axes[0].legend()

        full_pr = [r_summary[k].get("sage_pred", {}).get("mean", 0) for k in layer_keys]
        abl_pr = [a_summary.get(k, {}).get("sage_pred", {}).get("mean", 0) for k in layer_keys]
        full_pe = [r_summary[k].get("sage_pred", {}).get("ci", 0) for k in layer_keys]
        abl_pe = [a_summary.get(k, {}).get("sage_pred", {}).get("