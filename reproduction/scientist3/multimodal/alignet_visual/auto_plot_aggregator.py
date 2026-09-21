```python
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 10,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

os.makedirs("figures", exist_ok=True)

BASELINE_DIR = "experiment_results/experiment_486a902f4a9e4caca14be7278ce9a010_proc_2155478"
ABLATION_DIR = "experiment_results/experiment_1a4f52ffa20d4ffd88afa35babda594d_proc_4031535"

BASELINE_EXP  = os.path.join(BASELINE_DIR, "experiment_data.npy")
BASELINE_DINO = os.path.join(BASELINE_DIR, "dino_feats.npy")
BASELINE_SIG  = os.path.join(BASELINE_DIR, "siglip_feats.npy")
ABLATION_EXP  = os.path.join(ABLATION_DIR, "experiment_data.npy")


def safe_load_dict(path):
    try:
        return np.load(path, allow_pickle=True).item()
    except Exception as e:
        print("warn load", path, e)
        return None


def safe_load_arr(path):
    try:
        return np.load(path, allow_pickle=True)
    except Exception as e:
        print("warn load", path, e)
        return None


baseline_exp = safe_load_dict(BASELINE_EXP)
ablation_exp = safe_load_dict(ABLATION_EXP)

runs, run_keys, baseline_ooo, teacher_ooo = {}, [], None, None
if baseline_exp is not None and "N_EPOCHS" in baseline_exp:
    runs = baseline_exp["N_EPOCHS"]
    run_keys = sorted(runs.keys(), key=lambda k: int(k.split("_")[1]))
    if run_keys:
        first = runs[run_keys[0]]["THINGS"]
        baseline_ooo = first.get("baseline_ooo")
        teacher_ooo  = first.get("teacher_ooo")

ab_runs, ab_baseline_ooo, ab_teacher_ooo = {}, None, None
if ablation_exp is not None:
    ab_runs = (ablation_exp.get("direct_feature_distillation", {})
                           .get("THINGS", {})
                           .get("N_EPOCHS", {}))
    if ab_runs:
        first = next(iter(ab_runs.values()))
        ab_baseline_ooo = first.get("baseline_ooo")
        ab_teacher_ooo  = first.get("teacher_ooo")


# -------------------------------------------------------------------
# FIGURE 1 : Training dynamics on THINGS (train loss, val loss, OOO)
# -------------------------------------------------------------------
try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for rk in run_keys:
        ed = runs[rk]["THINGS"]
        axes[0].plot(ed["losses"]["train"], label="Epochs=" + str(ed["n_epochs"]), lw=2)
        axes[1].plot(ed["losses"]["val"],   label="Epochs=" + str(ed["n_epochs"]), lw=2)
        axes[2].plot(ed["metrics"]["val"],  label="Epochs=" + str(ed["n_epochs"]), lw=2)

    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Train Loss (MSE)")
    axes[0].set_title("(a) Training Loss"); axes[0].legend()
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Validation Loss (MSE)")
    axes[1].set_title("(b) Validation Loss"); axes[1].legend()

    if baseline_ooo is not None:
        axes[2].axhline(baseline_ooo, color="k", ls="--",
                        label="DINOv2 Baseline (" + f"{baseline_ooo:.3f}" + ")")
    if teacher_ooo is not None:
        axes[2].axhline(teacher_ooo, color="r", ls="--",
                        label="SigLIP Teacher (" + f"{teacher_ooo:.3f}" + ")")
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Aligned OOO Accuracy")
    axes[2].set_title("(c) Odd-One-Out Accuracy on THINGS")
    axes[2].legend(loc="lower right")

    fig.suptitle("Alignment Training Dynamics on THINGS (Pairwise-Similarity Distillation)",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig1-training-dynamics.png", bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("Figure 1 failed:", e); plt.close()


# -------------------------------------------------------------------
# FIGURE 2 : Final OOO accuracy vs baseline / teacher (bar chart)
# -------------------------------------------------------------------
try:
    fig, ax = plt.subplots(figsize=(9, 5.4))
    names = ["Epochs=" + str(runs[rk]["THINGS"]["n_epochs"]) for rk in run_keys]
    vals  = [runs[rk]["THINGS"]["final_aligned_ooo"] for rk in run_keys]
    x = np.arange(len(names))
    ax.bar(x, vals, color="steelblue", label="Aligned Student (DINOv2 ViT-B)")
    if baseline_ooo is not None:
        ax.axhline(baseline_ooo, color="k", ls="--",
                   label="Unaligned DINOv2 (" + f"{baseline_ooo:.3f}" + ")")
    if teacher_ooo is not None:
        ax.axhline(teacher_ooo, color="r", ls="--",
                   label="SigLIP Teacher (" + f"{teacher_ooo:.3f}" + ")")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.003, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Final Aligned OOO Accuracy")
    ax.set_title("Final Human-Alignment Accuracy vs. Training Length on THINGS")
    ax.legend(loc="lower right")
    lo = min(vals + ([baseline_ooo] if baseline_ooo is not None else [0.0]))
    hi = max(vals + ([teacher_ooo]  if teacher_ooo  is not None else [1.0]))
    ax.set_ylim([lo * 0.95, hi * 1.05])
    plt.tight_layout()
    plt.savefig("figures/fig2-final-ooo-accuracy.png", bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("Figure 2 failed:", e); plt.close()


# -------------------------------------------------------------------
# FIGURE 3 : Prediction analysis (GT hist, Pred hist, confusion matrix)
# -------------------------------------------------------------------
try:
    best_rk = max(run_keys, key=lambda k: runs[k]["THINGS"]["final_aligned_ooo"])
    ed = runs[best_rk]["THINGS"]
    preds = np.asarray(ed["predictions"]).astype(int)
    gts   = np.asarray(ed["ground_truth"]).astype(int)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].hist(gts, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8, color="green")
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_title("(a) Ground-Truth Odd-One-Out Position")
    axes[0].set_xlabel("Position (0, 1, 2)"); axes[0].set_ylabel("Count")

    axes[1].hist(preds, bins=[-0.5, 0.5, 1.5, 2.5], rwidth=0.8, color="orange")
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_title("(b) Predicted Odd-One-Out Position")
    axes[1].set_xlabel("Position (0, 1, 2)")

    cm = np.zeros((3, 3), dtype=int)
    for g, p in zip(gts, preds):
        if 0 <= g < 3 and 0 <= p < 3:
            cm[g, p] += 1
    im = axes[2].imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            axes[2].text(j, i, str(cm[i, j]), ha="center", va="center",
                         color="white" if cm[i, j] > cm.max() / 2 else "black",
                         fontsize=12)
    axes[2].set_xticks([0, 1, 2]); axes[2].set_yticks([0, 1, 2])
    axes[2].set_xticklabels(["Pred 0", "Pred 1", "Pred 2"])
    axes[2].set_yticklabels(["GT 0", "GT 1", "GT 2"])
    axes[2].set_title("(c) Confusion Matrix")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle("Prediction Analysis on THINGS (Best Run: Epochs=" + str(ed["n_epochs"]) + ")",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig3-prediction-analysis.png", bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("Figure 3 failed:", e); plt.close()


# -------------------------------------------------------------------
# FIGURE 4 : Representational similarity (Student vs Teacher)
#            (a) student pairwise-cosine histogram
#            (b) teacher pairwise-cosine histogram
#            (c) joint scatter with RSA correlation
# -------------------------------------------------------------------
try:
    dino_feats = safe_load_arr(BASELINE_DINO)
    sig_feats  = safe_load_arr(BASELINE_SIG)
    if dino_feats is not None and sig_feats is not None:
        d = np.asarray(dino_feats, dtype=np.float32)
        s = np.asarray(sig_feats,  dtype=np.float32)
        d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
        s = s / (np.linalg.norm(s, axis=1, keepdims=True) + 1e-8)

        sim_d = d @ d.T
        sim_s = s @ s.T
        n = sim_d.shape[0]
        iu = np.triu_indices(n, k=1)
        pd_all = sim_d[iu].astype(np.float32)
        ps_all = sim_s[iu].astype(np.float32)

        pdc = pd_all - pd_all.mean()
        psc = ps_all - ps_all.mean()
        denom = np.sqrt((pdc * pdc).sum() * (psc * psc).sum()) + 1e-12
        rsa = float((pdc * psc).sum() / denom)

        rng = np.random.RandomState(