```python
import os
import numpy as np
import matplotlib.pyplot as plt

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

BASELINE_NPY = "experiment_results/experiment_4313083cf25745ffa90f0d42acd916ad_proc_395380/experiment_data.npy"
RESEARCH_NPY = "experiment_results/experiment_ff42395603ec42719294c154fcb73750_proc_2293813/experiment_data.npy"
ABLATION_NPY = "experiment_results/experiment_1e120dbac14044faa77c0d00881fd716_proc_2544293/experiment_data.npy"


def safe_load(path):
    try:
        return np.load(path, allow_pickle=True).item()
    except Exception as e:
        print(f"[warn] could not load {path}: {e}")
        return {}


def save_fig(fig, name):
    path = os.path.join(FIG_DIR, name)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def pretty(s):
    return str(s).replace("_", " ")


baseline_data = safe_load(BASELINE_NPY)
research_data = safe_load(RESEARCH_NPY)
ablation_data = safe_load(ABLATION_NPY)

BASE_HB = baseline_data.get("harmbench", {})
RES_HB = research_data.get("harmbench", {})
ABL = ablation_data.get("single_vs_multi_layer_RR", {})


# =============================================================================
# FIGURE 1: RR training dynamics — loss components + weight schedule
# =============================================================================
try:
    loss_log = RES_HB.get("loss_log", []) or BASE_HB.get("loss_log", [])
    if loss_log:
        steps = [l["step"] for l in loss_log]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))

        ax = axes[0]
        if any("harm_loss" in l for l in loss_log):
            ax.plot(steps, [l.get("harm_loss", np.nan) for l in loss_log],
                    label="Harm (RR) loss", color="tab:red", lw=2)
        if any("retain_ce" in l for l in loss_log):
            ax.plot(steps, [l.get("retain_ce", np.nan) for l in loss_log],
                    label="Retain CE loss", color="tab:blue", lw=2)
        if any("rep_diff" in l for l in loss_log):
            ax.plot(steps, [l.get("rep_diff", np.nan) for l in loss_log],
                    label="Representation difference", color="tab:green", lw=2)
        if any("loss" in l for l in loss_log):
            ax.plot(steps, [l.get("loss", np.nan) for l in loss_log],
                    label="Total loss", color="black", ls="--", lw=1.5, alpha=0.8)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Loss value")
        ax.set_title("(a) RR training loss components")
        ax.legend(frameon=False)

        ax = axes[1]
        if any("w_harm" in l for l in loss_log):
            ax.plot(steps, [l.get("w_harm", np.nan) for l in loss_log],
                    label="Harm weight", color="tab:red", lw=2)
            ax.plot(steps, [l.get("w_retain", np.nan) for l in loss_log],
                    label="Retain weight", color="tab:blue", lw=2)
            ax.set_xlabel("Training step")
            ax.set_ylabel("Loss weight")
            ax.set_title("(b) Cosine-scheduled loss weights")
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "Weight schedule not logged",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

        fig.suptitle("RR training dynamics on Llama-3-8B-Instruct (Circuit Breakers dataset)",
                     fontsize=14, y=1.03)
        save_fig(fig, "fig1_training_dynamics.png")
except Exception as e:
    print(f"[fig1] error: {e}")
    plt.close("all")


# =============================================================================
# FIGURE 2: Headline HarmBench ASR (overall + per attack style)
# =============================================================================
try:
    base_asr = RES_HB.get("base_asr")
    rr_asr = RES_HB.get("rr_asr")
    base_ci = RES_HB.get("base_ci")
    rr_ci = RES_HB.get("rr_ci")
    base_by = RES_HB.get("base_asr_by_attack", {}) or {}
    rr_by = RES_HB.get("rr_asr_by_attack", {}) or {}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    ax = axes[0]
    if base_asr is not None and rr_asr is not None:
        vals = [base_asr, rr_asr]
        errs = None
        if base_ci and rr_ci:
            errs = [[base_asr - base_ci[0], rr_asr - rr_ci[0]],
                    [base_ci[1] - base_asr, rr_ci[1] - rr_asr]]
        bars = ax.bar(["Base Llama-3-8B", "RR (Ours)"], vals,
                      yerr=errs, capsize=6,
                      color=["tab:red", "tab:green"], edgecolor="black")
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                    f"{v:.2f}", ha="center", fontsize=12, fontweight="bold")
        ax.set_ylim(0, max(1.0, max(vals) + 0.15))
        ax.set_ylabel("Attack success rate")
        ax.set_title("(a) Overall HarmBench ASR with 95% CI")
    else:
        ax.text(0.5, 0.5, "No overall ASR data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()

    ax = axes[1]
    if base_by and rr_by:
        styles = list(base_by.keys())
        labels = [pretty(s) for s in styles]
        x = np.arange(len(styles))
        w = 0.38
        ax.bar(x - w / 2, [base_by[s] for s in styles], w,
               label="Base", color="tab:red", edgecolor="black")
        ax.bar(x + w / 2, [rr_by.get(s, 0) for s in styles], w,
               label="RR (Ours)", color="tab:green", edgecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylabel("Attack success rate")
        ax.set_ylim(0, 1.05)
        ax.set_title("(b) ASR by HarmBench attack style")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No per-style ASR data",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle("Representation Rerouting substantially reduces HarmBench attack success rate",
                 fontsize=14, y=1.03)
    save_fig(fig, "fig2_headline_asr.png")
except Exception as e:
    print(f"[fig2] error: {e}")
    plt.close("all")


# =============================================================================
# FIGURE 3: ASR change by style + per-sample breakdown
# =============================================================================
try:
    base_by = RES_HB.get("base_asr_by_attack", {}) or {}
    rr_by = RES_HB.get("rr_asr_by_attack", {}) or {}
    samples_base = RES_HB.get("samples_base", []) or []
    samples_rr = RES_HB.get("samples_rr", []) or []

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    if base_by and rr_by:
        styles = list(base_by.keys())
        labels = [pretty(s) for s in styles]
        deltas = [rr_by.get(s, 0) - base_by[s] for s in styles]
        colors = ["tab:green" if d < 0 else "tab:red" for d in deltas]
        bars = ax.bar(labels, deltas, color=colors, edgecolor="black")
        for b, v in zip(bars, deltas):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (0.015 if v >= 0 else -0.04),
                    f"{v:+.2f}", ha="center", fontsize=10)
        ax.axhline(0, color="k", lw=0.7)
        ax.set_ylabel("ASR change (RR minus Base)")
        ax.set_title("(a) Per-style ASR change; negative means safer")
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(25)
            lbl.set_ha("right")
    else:
        ax.text(0.5, 0.5, "No delta data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()

    ax = axes[1]
    if samples_base and samples_rr:
        n = min(len(samples_base), len(samples_rr))
        idx = np.arange(n)
        base_scores = [s.get("asr", 0) for s in samples_base[:n]]
        rr_scores = [s.get("asr", 0) for s in samples_rr[:n]]
        w = 0.4
        ax.bar(idx - w / 2, base_scores, w, label="Base", color="tab:red")
        ax.bar(idx + w / 2, rr_scores, w, label="RR (Ours)", color="tab:green")
        ax.set_xlabel("HarmBench sample index")
        ax.set_ylabel("ASR (1 equals jailbroken)")
        ax.set_ylim(0, 1.15)
        ax.set_title("(b) Per-sample attack success")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No per-sample data", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle("Attack-style and per-sample analysis of RR defence",
                 fontsize=14, y=1.03)
    save_fig(fig, "fig3_per_style_and_sample.png")
except Exception as e:
    print(f"[fig3] error: {e}")
    plt.close("all")


# =============================================================================
# FIGURE 4: Held-out generalization probe + benign response preservation
# =============================================================================
try:
    probe = RES_HB.get("generalization_probe", {}) or {}
    pre_cos = probe.get("pre_train_cos")
    post_cos = probe.get("post_train_cos")
    before = RES_HB.get("benign_before", "") or BASE_HB.get("benign_before", "")
    after = RES_HB.get("benign_after", "") or BASE_HB.get("benign_after", "")

    fig, axes = plt.subplots(1, 2, figs