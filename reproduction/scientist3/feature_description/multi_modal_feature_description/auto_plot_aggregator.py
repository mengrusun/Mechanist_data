```python
"""Aggregator for final SemanticLens paper figures."""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 120, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

NPROBE_PATH = ("experiment_results/experiment_7454962ce9c7442d9e0d2876a83263be_"
               "proc_1122355/experiment_data.npy")
TOPK_PATH = ("experiment_results/experiment_15c348a200fc4eaa9da81f4a7396513c_"
             "proc_1176445/experiment_data.npy")


def safe_load(p):
    try:
        return np.load(p, allow_pickle=True).item()
    except Exception as e:
        print("[WARN] load failed", p, e)
        return {}


def as_arr(x):
    return np.asarray(x)


nprobe = safe_load(NPROBE_PATH)
topk = safe_load(TOPK_PATH)

npr = nprobe.get("N_PROBE", {}).get("imagenet_val_resnet50", {})
np_values = list(npr.get("n_probe_values", []))
np_per = npr.get("per_n_probe", {})
np_best = npr.get("best_n_probe", None)
np_val_metrics = npr.get("metrics", {}).get("val", [])

tkr = topk.get("TOP_K", {}).get("imagenet_val_resnet50", {})
tk_values = list(tkr.get("top_k_values", []))
tk_per = tkr.get("per_top_k", {})
tk_best = tkr.get("best_top_k", None)
tk_inspected = list(tkr.get("inspected_channels", []))


# ============= Fig 1: sweep curves ============= #
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    if np_val_metrics:
        xs = [m["N_PROBE"] for m in np_val_metrics]
        yc = [m["concept_consistency_score_mean"] for m in np_val_metrics]
        yr = [m["random_baseline_mean"] for m in np_val_metrics]
        ax.plot(xs, yc, "o-", color="tab:blue", lw=2, ms=8,
                label="Top-k activating images")
        ax.plot(xs, yr, "s--", color="tab:orange", lw=2, ms=8,
                label="Random baseline")
        ax.set_xscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("Probe pool size (N PROBE)")
    ax.set_ylabel("Mean pairwise CLIP cosine similarity")
    ax.set_title("(a) Concept consistency vs probe pool size")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    ax = axes[1]
    if tk_values:
        xs = tk_values
        yc = [tk_per[k]["mean_concept"] for k in xs]
        yr = [tk_per[k]["mean_random"] for k in xs]
        yd = [tk_per[k]["delta"] for k in xs]
        ax.plot(xs, yc, "o-", color="tab:blue", lw=2, ms=8,
                label="Top-k activating images")
        ax.plot(xs, yr, "s--", color="tab:orange", lw=2, ms=8,
                label="Random baseline")
        ax.plot(xs, yd, "^:", color="tab:green", lw=2, ms=8,
                label="Delta (concept minus random)")
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel("Number of top-activating images (TOP K)")
    ax.set_ylabel("Mean pairwise CLIP cosine similarity")
    ax.set_title("(b) Concept consistency vs number of reference inputs")
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)

    fig.suptitle("ResNet-50 layer4 on ImageNet-val: sweep summary",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_sweep_curves.png"),
                bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("[fig1] failed:", e); plt.close("all")


# ============= Fig 2: delta bars ============= #
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    if np_val_metrics:
        xs = [m["N_PROBE"] for m in np_val_metrics]
        ds = [m["delta"] for m in np_val_metrics]
        bars = ax.bar([str(x) for x in xs], ds, color="steelblue",
                      edgecolor="black")
        for b, v in zip(bars, ds):
            ax.text(b.get_x() + b.get_width() / 2, v, "%.3f" % v,
                    ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Probe pool size (N PROBE)")
    ax.set_ylabel("Delta (concept minus random)")
    ax.set_title("(a) Improvement over random vs probe pool size")
    ax.grid(alpha=0.3, axis="y")

    ax = axes[1]
    if tk_values:
        xs = tk_values
        ds = [tk_per[k]["delta"] for k in xs]
        bars = ax.bar([str(x) for x in xs], ds, color="darkorange",
                      edgecolor="black")
        for b, v in zip(bars, ds):
            ax.text(b.get_x() + b.get_width() / 2, v, "%.3f" % v,
                    ha="center", va="bottom", fontsize=10)
    ax.set_xlabel("Number of top-activating images (TOP K)")
    ax.set_ylabel("Delta (concept minus random)")
    ax.set_title("(b) Improvement over random vs number of reference inputs")
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Concept-vs-random gap across ablation dimensions",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_delta_bars.png"),
                bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("[fig2] failed:", e); plt.close("all")


# ============= Fig 3: histograms at best configs ============= #
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    labels = [("N PROBE", np_per, np_best), ("TOP K", tk_per, tk_best)]
    for i, (name, per, best) in enumerate(labels):
        ax = axes[i]
        d = per.get(best, {})
        cs = as_arr(d.get("concept_scores", []))
        rs = as_arr(d.get("random_scores", []))
        if cs.size and rs.size:
            lo = float(min(cs.min(), rs.min()))
            hi = float(max(cs.max(), rs.max()))
            bins = np.linspace(lo, hi, 24)
            ax.hist(rs, bins=bins, alpha=0.6, color="tab:orange",
                    label="Random baseline")
            ax.hist(cs, bins=bins, alpha=0.6, color="tab:blue",
                    label="Top-k activating images")
            ax.axvline(cs.mean(), color="tab:blue", ls="--", lw=1.5,
                       label="concept mean = %.3f" % cs.mean())
            ax.axvline(rs.mean(), color="tab:orange", ls="--", lw=1.5,
                       label="random mean = %.3f" % rs.mean())
        ax.set_xlabel("Mean pairwise CLIP cosine similarity")
        ax.set_ylabel("Number of ResNet-50 channels")
        letter = "ab"[i]
        ax.set_title("(%s) Per-channel scores at best %s = %s" %
                     (letter, name, str(best)))
        ax.legend(frameon=False, fontsize=9)
        ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Per-channel concept-consistency distributions",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_score_histograms.png"),
                bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("[fig3] failed:", e); plt.close("all")


# ============= Fig 4: scatter concept vs random ============= #
try:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.3))
    labels = [("N PROBE", np_per, np_best), ("TOP K", tk_per, tk_best)]
    for i, (name, per, best) in enumerate(labels):
        ax = axes[i]
        d = per.get(best, {})
        cs = as_arr(d.get("concept_scores", []))
        rs = as_arr(d.get("random_scores", []))
        if cs.size and rs.size:
            ax.scatter(rs, cs, s=32, alpha=0.75, color="tab:purple",
                       edgecolor="k", linewidths=0.4,
                       label="ResNet-50 channel")
            lo = float(min(cs.min(), rs.min()))
            hi = float(max(cs.max(), rs.max()))
            ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, label="y = x (no gain)")
            frac = float(np.mean(cs > rs)) * 100.0
            ax.text(0.04, 0.96, "%.1f%% of channels above diagonal" % frac,
                    transform=ax.transAxes, va="top", fontsize=11,
                    bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        ax.set_xlabel("Random baseline score")
        ax.set_ylabel("Concept (top-k) score")
        letter = "ab"[i]
        ax.set_title("(%s) Per-channel concept vs random at best %s = %s" %
                     (letter, name, str(best)))
        ax.legend(frameon=False, loc="lower right")
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    fig.suptitle("Concept scores dominate the random baseline channel-by-channel",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_scatter_concept_vs_random.png"),
                bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("[fig4] failed:", e); plt.close("all")


# ============= Fig 5: boxplots across sweep values ============= #
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5