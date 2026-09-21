I notice my previous scripts were being truncated. Let me produce a shorter, complete, self-contained script.

```python
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 120, "savefig.dpi": 300,
})

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

NPY_PATH = "experiment_results/experiment_2d0947357395450d8631ff1e46644350_proc_329916/experiment_data.npy"

try:
    experiment_data = np.load(NPY_PATH, allow_pickle=True).item()
except Exception as e:
    print("Error loading experiment data:", e)
    experiment_data = {}

data = experiment_data.get("triviaqa_gemma3_27b_pt", {})

layer_indices = np.array(data.get("layer_indices", []))
conf_acc = np.array(data.get("per_layer_confidence_acc", []), dtype=float)
corr_acc = np.array(data.get("per_layer_correctness_acc", []), dtype=float)
ctrl_acc = np.array(data.get("per_layer_control_conf_acc", []), dtype=float)
maj_conf = data.get("majority_baseline_conf", None)
maj_corr = data.get("majority_baseline_corr", None)
raw_conf = np.array(data.get("raw_confidences", []), dtype=float)
raw_corr = np.array(data.get("raw_correctness", []), dtype=float)
best_layer_conf = data.get("best_layer_conf", None)
best_layer_corr = data.get("best_layer_corr", None)


# ---------- Figure 1: per-layer probe accuracies ----------
try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    ax = axes[0]
    if len(layer_indices) and len(conf_acc):
        ax.plot(layer_indices, conf_acc, "o-", color="tab:blue",
                label="Post-answer hidden state")
    if len(layer_indices) and len(ctrl_acc):
        ax.plot(layer_indices, ctrl_acc, "s--", color="tab:orange",
                label="Pre-answer hidden state (control)")
    if maj_conf is not None:
        ax.axhline(maj_conf, color="gray", linestyle=":",
                   label="Majority baseline ({:.3f})".format(maj_conf))
    if best_layer_conf is not None:
        ax.axvline(best_layer_conf, color="tab:blue", linestyle=":", alpha=0.5,
                   label="Best layer = {}".format(best_layer_conf))
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Verbal-confidence probe accuracy")
    ax.set_title("(a) Confidence probe vs. layer")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    ax = axes[1]
    if len(layer_indices) and len(corr_acc):
        ax.plot(layer_indices, corr_acc, "^-", color="tab:green",
                label="Post-answer hidden state")
    if maj_corr is not None:
        ax.axhline(maj_corr, color="black", linestyle=":",
                   label="Majority baseline ({:.3f})".format(maj_corr))
    if best_layer_corr is not None:
        ax.axvline(best_layer_corr, color="tab:green", linestyle=":", alpha=0.5,
                   label="Best layer = {}".format(best_layer_corr))
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Answer-correctness probe accuracy")
    ax.set_title("(b) Correctness probe vs. layer")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    ax = axes[2]
    if len(layer_indices) and len(conf_acc) and len(ctrl_acc):
        gap = conf_acc - ctrl_acc
        ax.plot(layer_indices, gap, "d-", color="tab:red",
                label="Post minus Pre (confidence probe)")
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.fill_between(layer_indices, 0, gap, where=(gap > 0),
                        color="tab:red", alpha=0.2,
                        label="Caching advantage (post > pre)")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Accuracy gap (post - pre)")
    ax.set_title("(c) Evidence of post-answer caching")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    fig.suptitle("Verbal-confidence caching in Gemma-3-27B on TriviaQA",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1-probe-accuracy-vs-layer.png"),
                bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("[Fig1] Error:", e)
    plt.close("all")


# ---------- Figure 2: best-layer summary bar chart ----------
try:
    best_conf_val = float(np.nanmax(conf_acc)) if len(conf_acc) else 0.0
    best_corr_val = float(np.nanmax(corr_acc)) if len(corr_acc) else 0.0
    best_ctrl_val = float(np.nanmax(ctrl_acc)) if len(ctrl_acc) else 0.0

    labels = ["Post-answer\nConfidence probe",
              "Pre-answer\nConfidence probe",
              "Confidence\nMajority baseline",
              "Post-answer\nCorrectness probe",
              "Correctness\nMajority baseline"]
    values = [best_conf_val, best_ctrl_val, maj_conf or 0.0,
              best_corr_val, maj_corr or 0.0]
    colors = ["steelblue", "orange", "lightsteelblue",
              "seagreen", "lightgreen"]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars = ax.bar(labels, values, color=colors, edgecolor="black")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.015,
                "{:.3f}".format(v), ha="center", fontsize=11)
    ax.set_ylim(0, max(max(values) * 1.25, 0.4))
    ax.set_ylabel("Accuracy")
    ax.set_title("Best-layer probe accuracy vs. baselines (TriviaQA, Gemma-3-27B)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2-probe-vs-baseline-summary.png"),
                bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("[Fig2] Error:", e)
    plt.close("all")


# ---------- Figure 3: verbal-confidence behavior ----------
try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

    ax = axes[0]
    if len(raw_conf):
        ax.hist(raw_conf, bins=20, color="steelblue",
                edgecolor="black", alpha=0.85)
        ax.axvline(raw_conf.mean(), color="red", linestyle="--",
                   label="mean = {:.2f}".format(raw_conf.mean()))
        ax.axvline(np.median(raw_conf), color="black", linestyle=":",
                   label="median = {:.2f}".format(np.median(raw_conf)))
        ax.legend()
    ax.set_xlabel("Verbal confidence (expected first digit x 10)")
    ax.set_ylabel("Count")
    ax.set_title("(a) Verbal-confidence distribution (n={})".format(len(raw_conf)))
    ax.grid(alpha=0.3)

    ax = axes[1]
    if len(raw_conf) and len(raw_corr) and len(raw_conf) == len(raw_corr):
        c0 = raw_conf[raw_corr == 0]
        c1 = raw_conf[raw_corr == 1]
        bp = ax.boxplot([c0, c1], labels=["Incorrect", "Correct"],
                        patch_artist=True, widths=0.5)
        for patch, color in zip(bp["boxes"], ["#ff9999", "#99cc99"]):
            patch.set_facecolor(color)
        rng = np.random.default_rng(0)
        ax.scatter(np.ones(len(c0)) + rng.uniform(-0.08, 0.08, len(c0)),
                   c0, alpha=0.5, color="red", s=15,
                   label="Incorrect (n={})".format(len(c0)))
        ax.scatter(2 * np.ones(len(c1)) + rng.uniform(-0.08, 0.08, len(c1)),
                   c1, alpha=0.5, color="green", s=15,
                   label="Correct (n={})".format(len(c1)))
        ax.legend(loc="lower right")
    ax.set_ylabel("Verbal confidence")
    ax.set_title("(b) Confidence conditioned on correctness")
    ax.grid(alpha=0.3)

    ax = axes[2]
    if len(raw_conf) and len(raw_corr):
        lo, hi = raw_conf.min(), raw_conf.max() + 1e-6
        bins = np.linspace(lo, hi, 6)
        bin_ids = np.clip(np.digitize(raw_conf, bins) - 1, 0, len(bins) - 2)
        centers, accs, counts = [], [], []
        for b in range(len(bins) - 1):
            mask = bin_ids == b
            if mask.sum() > 0:
                centers.append((bins[b] + bins[b + 1]) / 2)
                accs.append(raw_corr[mask].mean())
                counts.append(int(mask.sum()))
        centers = np.array(centers); accs = np.array(accs)
        ax.plot(centers, accs, "o-", color="purple", linewidth=2,
                label="Empirical accuracy per bin")
        ax.plot([lo, hi], [lo / 10.0, hi / 10.0], "k--", alpha=0.4,
                label="Ideal calibration (conf / 10)")
        for x, y, c in zip(centers, accs, counts):
            ax.annotate("n={}".format(c), (x, y),
                        textcoords="offset points", xytext=(0, 8),
                        ha="center", fontsize=9)
        ax.axhline(raw_corr.mean(), color="gray", linestyle=":",
                   label="Overall accuracy = {:.3f}".format(raw_corr.mean()))
        ax.legend(loc="best")
    ax.set_xlabel("Verbal confidence (bin center)")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title("(c) Calibration curve")
    ax.grid(alpha=0.3)

    fig.suptitle("Verbal-confidence behavior on TriviaQA (Gemma-3-27B)",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3-confidence