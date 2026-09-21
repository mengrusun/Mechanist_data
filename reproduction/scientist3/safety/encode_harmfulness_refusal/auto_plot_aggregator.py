It appears my previous responses were being truncated causing the triple backticks to be included in the saved file. I will now provide a shorter, complete script that fits within the response.

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

os.makedirs("figures", exist_ok=True)

BASELINE_NPY = "experiment_results/experiment_1123b89974864fd4b6e87b401e98f68d_proc_647101/experiment_data.npy"
RESEARCH_NPY = "experiment_results/experiment_81674ebba51942a08800d4270c901383_proc_722033/experiment_data.npy"

baseline_ed = None
try:
    baseline_ed = np.load(BASELINE_NPY, allow_pickle=True).item()["N_PER_CLASS_tuning"]["llama3_advbench_alpaca"]
    print("Loaded baseline.")
except Exception as e:
    print(f"[warn] baseline load: {e}")

research_ed = None
try:
    research_ed = np.load(RESEARCH_NPY, allow_pickle=True).item()["harmful_vs_refusal_dissociation"]["llama3_advbench_alpaca"]
    print("Loaded research.")
except Exception as e:
    print(f"[warn] research load: {e}")

def by_pos(records, pos):
    return sorted([r for r in records if r.get("position") == pos], key=lambda r: r["layer"])

COND_LABEL = {"baseline": "baseline", "posH": "+H", "negH": "-H", "posR": "+R", "negR": "-R"}

# =====================================================================
# FIGURE 1: Baseline probe layer sweep (3 subplots)
# =====================================================================
try:
    if baseline_ed is None: raise RuntimeError("no baseline")
    N_cands = baseline_ed["N_candidates"]; per_conf = baseline_ed["per_config"]
    best_N = baseline_ed["best_N"]; best_L = baseline_ed["best_layer"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))
    for N in N_cands:
        L = sorted(per_conf[N]["per_layer"].keys())
        axes[0].plot(L, [per_conf[N]["per_layer"][l]["val_acc"] for l in L], "o-", label=f"N={N}")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Validation Accuracy")
    axes[0].set_title("(a) Probe Accuracy vs Layer")
    axes[0].legend(loc="lower right", title="Samples/class"); axes[0].grid(True, alpha=0.3)

    L = sorted(per_conf[best_N]["per_layer"].keys())
    tr = [per_conf[best_N]["per_layer"][l]["train_acc"] for l in L]
    va = [per_conf[best_N]["per_layer"][l]["val_acc"] for l in L]
    axes[1].plot(L, tr, "o-", label="Train"); axes[1].plot(L, va, "s-", label="Validation")
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title(f"(b) Train vs Validation (best N={best_N}, layer={best_L})")
    axes[1].legend(loc="lower right"); axes[1].grid(True, alpha=0.3)

    ra = [per_conf[best_N]["per_layer"][l]["random_acc"] for l in L]
    sa = [per_conf[best_N]["per_layer"][l]["shuffled_acc"] for l in L]
    axes[2].plot(L, va, "o-", label="Mean-difference probe")
    axes[2].plot(L, ra, "s--", label="Random direction")
    axes[2].plot(L, sa, "^--", label="Shuffled labels")
    axes[2].set_xlabel("Layer"); axes[2].set_ylabel("Validation Accuracy")
    axes[2].set_title(f"(c) Probe vs Control Baselines (N={best_N})")
    axes[2].legend(loc="lower right"); axes[2].grid(True, alpha=0.3)

    fig.suptitle("Harmfulness Probe Layer Sweep on Llama-3-8B-Instruct (AdvBench vs Alpaca)", fontsize=15, y=1.03)
    fig.tight_layout()
    fig.savefig("figures/fig1_probe_layer_sweep.png", bbox_inches="tight")
    plt.close(fig); print("Saved fig1")
except Exception as e:
    print(f"[fig1] {e}"); plt.close("all")

# =====================================================================
# FIGURE 2: Best-N accuracy vs training-set size + confusion matrix
# =====================================================================
try:
    if baseline_ed is None: raise RuntimeError("no baseline")
    N_cands = baseline_ed["N_candidates"]; per_conf = baseline_ed["per_config"]
    best_N = baseline_ed["best_N"]; best_L = baseline_ed["best_layer"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    bv = [per_conf[N]["best_val_acc"] for N in N_cands]
    bl = [per_conf[N]["best_layer"] for N in N_cands]
    axes[0].plot(N_cands, bv, "o-", color="tab:blue", linewidth=2)
    for N, v, L in zip(N_cands, bv, bl):
        axes[0].annotate(f"layer {L}", (N, v), textcoords="offset points", xytext=(6, 5), fontsize=10)
    axes[0].set_xlabel("Samples per class"); axes[0].set_ylabel("Best Validation Accuracy")
    axes[0].set_title("(a) Best Val Accuracy vs Training-Set Size"); axes[0].grid(True, alpha=0.3)

    preds = np.array(baseline_ed["predictions"]); gt = np.array(baseline_ed["ground_truth"])
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(gt, preds): cm[int(t), int(p)] += 1
    im = axes[1].imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, str(cm[i, j]), ha="center", va="center", color="black", fontsize=14, fontweight="bold")
    axes[1].set_xticks([0, 1]); axes[1].set_xticklabels(["Benign", "Harmful"])
    axes[1].set_yticks([0, 1]); axes[1].set_yticklabels(["Benign", "Harmful"])
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Ground Truth")
    axes[1].set_title(f"(b) Confusion Matrix (N={best_N}, layer={best_L})")
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle("Harmfulness Probe Best Configuration Summary", fontsize=15, y=1.03)
    fig.tight_layout()
    fig.savefig("figures/fig2_probe_best_config.png", bbox_inches="tight")
    plt.close(fig); print("Saved fig2")
except Exception as e:
    print(f"[fig2] {e}"); plt.close("all")

# =====================================================================
# FIGURE 3: Position-specific probe accuracy & direction norms
# =====================================================================
try:
    if research_ed is None: raise RuntimeError("no research")
    val_records = research_ed["metrics"]["val"]
    a = by_pos(val_records, "A"); b = by_pos(val_records, "B")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    if a:
        axes[0].plot([r["layer"] for r in a], [r["acc"] for r in a], "o-", color="tab:blue",
                     label="Position A (final instruction)")
        axes[0].plot([r["layer"] for r in a], [r["shuffled"] for r in a], "o--", color="tab:blue",
                     alpha=0.5, label="Position A shuffled")
    if b:
        axes[0].plot([r["layer"] for r in b], [r["acc"] for r in b], "s-", color="tab:red",
                     label="Position B (post-instruction)")
        axes[0].plot([r["layer"] for r in b], [r["shuffled"] for r in b], "s--", color="tab:red",
                     alpha=0.5, label="Position B shuffled")
    axes[0].axhline(0.5, color="gray", ls=":", alpha=0.5, label="Chance")
    axes[0].set_xlabel("Layer"); axes[0].set_ylabel("Validation Accuracy")
    axes[0].set_title(f"(a) Probe Accuracy by Layer and Token Position\n"
                      f"Best H layer={research_ed['best_H_layer']}, Best R layer={research_ed['best_R_layer']}")
    axes[0].legend(loc="lower right", fontsize=9); axes[0].grid(True, alpha=0.3)

    if a:
        axes[1].plot([r["layer"] for r in a], [r["direction_norm"] for r in a], "o-",
                     color="tab:blue", label="Position A (harmfulness)")
    if b:
        axes[1].plot([r["layer"] for r in b], [r["direction_norm"] for r in b], "s-",
                     color="tab:red", label="Position B (refusal)")
    axes[1].set_xlabel("Layer"); axes[1].set_ylabel(r"$\|\mu_{harmful} - \mu_{benign}\|$")
    axes[1].set_title("(b) Mean-Difference Direction Norm by Layer")
    axes[1].legend(loc="best"); axes[1].grid(True, alpha=0.3)

    fig.suptitle("Position-Specific Probing of Harmfulness and Refusal Directions", fontsize=15, y=1.03)
    fig.tight_layout()
    fig.savefig("figures/fig3_position_probes.png", bbox_inches="tight")
    plt.close(fig); print("Saved fig3")
except Exception as e:
    print(f"[fig3] {e}"); plt.close("all")

# =====================================================================
# FIGURE 4: Causal steering effects on harmful and benign prompts
# =====================================================================
try:
    if research_ed is None: raise RuntimeError("no research")
    sh = research_ed["steering_results"]["harmful_eval"]
    sb = research_ed["steering_results"]["benign_eval"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    hconds = [c for c in ["baseline", "posH", "negH", "posR", "negR"] if c in sh]
    x = np.arange(len(hconds)); w = 0.38
    hp_h = [sh[c]["harm_prob"] for c in hconds]
    rr_h = [sh[c]["refusal_rate"] for c in hconds]
    axes[0].bar(x - w/2, hp_h, w, color="tab:blue", label="Internal harm probability")
    axes[0].bar(x + w/2, rr_h, w, color="tab:red", label="Behavioral refusal rate")