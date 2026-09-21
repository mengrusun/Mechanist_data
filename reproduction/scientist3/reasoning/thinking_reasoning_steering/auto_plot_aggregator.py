```python
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

BASELINE_NPY = "experiment_results/experiment_01e195f636544d63bd8e4d422ce30ecd_proc_1318014/experiment_data.npy"
RESEARCH_NPY = "experiment_results/experiment_5769111f25784257afadbd3e0b5838ee_proc_2022124/experiment_data.npy"
ABLATION_NPY = "experiment_results/experiment_b431bc9ffd0b48dcb4e1645920269811_proc_2753333/experiment_data.npy"


def pretty(s):
    return str(s).replace("_", " ")


def safe_load(path):
    try:
        return np.load(path, allow_pickle=True).item()
    except Exception as e:
        print("Could not load", path, ":", e)
        return {}


baseline_data = safe_load(BASELINE_NPY)
research_data = safe_load(RESEARCH_NPY)
ablation_data = safe_load(ABLATION_NPY)


# ======================================================================
# FIG 1 - BASELINE: max-new-tokens tuning
# ======================================================================
try:
    root = baseline_data.get("max_new_tokens", {}).get("r1_distill_llama_8b_steering", {})
    runs = root.get("runs", {})
    mnts = sorted([int(k) for k in runs.keys()])
    if mnts:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        rates = [runs[str(m)].get("behaviour_steering_success_rate", np.nan) for m in mnts]
        accs = [runs[str(m)].get("baseline_accuracy", np.nan) for m in mnts]
        axes[0].plot(mnts, rates, "o-", color="tab:blue", lw=2, label="Steering success rate")
        axes[0].plot(mnts, accs, "s-", color="tab:orange", lw=2, label="Baseline accuracy")
        axes[0].set_xlabel("max new tokens")
        axes[0].set_ylabel("Value")
        axes[0].set_title("(a) Success rate & accuracy vs generation length")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        behaviours = None
        for m in mnts:
            bc = runs[str(m)].get("behaviour_counts", {}).get("baseline_total", {})
            if bc and behaviours is None:
                behaviours = list(bc.keys())
        if behaviours:
            x = np.arange(len(behaviours))
            w = 0.8 / max(1, len(mnts))
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(mnts)))
            for i, m in enumerate(mnts):
                bc = runs[str(m)].get("behaviour_counts", {}).get("baseline_total", {})
                vals = [bc.get(b, 0) for b in behaviours]
                axes[1].bar(x + (i - (len(mnts) - 1) / 2) * w, vals, w,
                            label="MNT=" + str(m), color=colors[i])
            axes[1].set_xticks(x)
            axes[1].set_xticklabels([pretty(b) for b in behaviours], rotation=20)
            axes[1].set_ylabel("Total occurrences (baseline)")
            axes[1].set_title("(b) Baseline behaviour totals across generation lengths")
            axes[1].legend()

        best_val = root.get("best_value", mnts[-1])
        sr = runs.get(str(best_val), {}).get("steering_results", {})
        behs = list(sr.keys())
        if behs:
            COEFFS = [-1.0, 0.0, 1.0]
            xb = np.arange(len(behs))
            wb = 0.25
            palette = ["#d62728", "#7f7f7f", "#2ca02c"]
            for i, coeff in enumerate(COEFFS):
                vals = [sr[b].get(str(coeff), {}).get("mean", 0.0) for b in behs]
                axes[2].bar(xb + (i - 1) * wb, vals, wb,
                            label="coeff=" + str(coeff), color=palette[i])
            axes[2].set_xticks(xb)
            axes[2].set_xticklabels([pretty(b) for b in behs], rotation=20)
            axes[2].set_ylabel("Mean behaviour count / task")
            axes[2].set_title("(c) Behaviour counts under steering (MNT=" + str(best_val) + ")")
            axes[2].legend()

        fig.suptitle("Baseline pipeline: max-new-tokens tuning on R1-Distill-Llama-8B",
                     fontsize=14, y=1.03)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig1_baseline_mnt_tuning.png"), bbox_inches="tight")
        plt.close(fig)
except Exception as e:
    print("Fig1 error:", e)
    plt.close("all")


# ======================================================================
# FIG 2 - RESEARCH: dose response, accuracy vs coefficient, BSE
# ======================================================================
try:
    root_r = research_data.get("steering_analysis", {}).get("r1_distill_llama_8b", {})
    dose_response = root_r.get("dose_response", {})
    bse_scores = root_r.get("bse_scores", {})
    baseline_r = root_r.get("baseline", {})
    baseline_acc_r = baseline_r.get("accuracy", None)

    if dose_response:
        fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
        for b, r in dose_response.items():
            cs = sorted([float(k) for k in r.keys()])
            ys = [r[str(c)]["target_mean"] for c in cs]
            axes[0].plot(cs, ys, "o-", label=pretty(b), lw=2, markersize=6)
        axes[0].set_xlabel("Steering coefficient (x BASE SCALE)")
        axes[0].set_ylabel("Mean target behaviour count / CoT")
        axes[0].set_title("(a) Dose-response curves (difference-of-means)")
        axes[0].axvline(0, color="k", lw=0.5, ls=":")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        for b, r in dose_response.items():
            cs = sorted([float(k) for k in r.keys()])
            ys = [r[str(c)]["acc"] for c in cs]
            axes[1].plot(cs, ys, "s-", label=pretty(b), lw=2, markersize=6)
        if baseline_acc_r is not None:
            axes[1].axhline(baseline_acc_r, color="k", ls="--",
                            label="baseline = {:.2f}".format(baseline_acc_r))
        axes[1].set_xlabel("Steering coefficient")
        axes[1].set_ylabel("Task accuracy")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_title("(b) Downstream accuracy vs steering coefficient")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        if bse_scores:
            bs = list(bse_scores.keys())
            ys_a1 = [bse_scores[b].get("bse_a1", 0.0) for b in bs]
            ys_a2 = [bse_scores[b].get("bse_a2") if bse_scores[b].get("bse_a2") is not None else 0.0
                     for b in bs]
            x = np.arange(len(bs))
            w = 0.35
            axes[2].bar(x - w / 2, ys_a1, w, label="BSE (alpha=1)", color="steelblue")
            axes[2].bar(x + w / 2, ys_a2, w, label="BSE (alpha=2)", color="orange")
            axes[2].set_xticks(x)
            axes[2].set_xticklabels([pretty(b) for b in bs], rotation=15)
            axes[2].set_ylabel("BSE score")
            axes[2].axhline(0, color="k", lw=0.5)
            axes[2].set_title("(c) Behaviour Steering Efficacy")
            axes[2].legend()

        fig.suptitle("Main result: difference-of-means steering on R1-Distill-Llama-8B (math)",
                     fontsize=14, y=1.03)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "fig2_research_main_results.png"), bbox_inches="tight")
        plt.close(fig)
except Exception as e:
    print("Fig2 error:", e)
    plt.close("all")


# ======================================================================
# FIG 3 - RESEARCH: layer ablation, cross-interference, specificity
# ======================================================================
try:
    root_r = research_data.get("steering_analysis", {}).get("r1_distill_llama_8b", {})
    layer_ablation = root_r.get("layer_ablation", {})
    interference = root_r.get("cross_interference", {})
    dose_response = root_r.get("dose_response", {})

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    hedge_abl = layer_ablation.get("hedging", {})
    if hedge_abl:
        lis = sorted([int(k) for k in hedge_abl.keys()])
        deltas = [hedge_abl[str(li)]["delta"] for li in lis]
        bses = [hedge_abl[str(li)]["bse"] for li in lis]
        accs_pos = [hedge_abl[str(li)]["acc_pos"] for li in lis]
        accs_neg = [hedge_abl[str(li)]["acc_neg"] for li in lis]
        ax1 = axes[0]
        l1, = ax1.plot(lis, deltas, "o-", color="steelblue", label="Delta freq (+ vs -)")
        l2, = ax1.plot(lis, bses, "s-", color="green", label="BSE")
        ax1.set_xlabel("Transformer layer index")
        ax1.set_ylabel("Delta frequency / BSE")
        ax1.set_title("(a) Per-layer ablation on hedging behaviour")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.spines["top"].set_visible(False)
        l3, = ax2.plot(lis, accs_pos, "^--", color="red", alpha=0.7, label="accuracy (+coeff)")
        l4, = ax2.plot(lis, accs_neg, "v--", color="purple", alpha=0.7, label="accuracy (-coeff)")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.05)
        ax1.legend(handles=[l1, l2, l3, l4], loc="best", fontsize=9)
    else:
        axes[0].text(0.5, 0.5, "no layer ablation data", ha="center", va="center")
        axes[0].set_title("(a) Per-layer ablation on hedging behaviour")

    if interference:
        src = list(interference.keys())
        tgt = sorted({t for r in interference.values() for t in r.keys()})