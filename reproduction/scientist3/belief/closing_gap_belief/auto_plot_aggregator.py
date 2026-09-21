```python
"""Aggregator for final paper figures. Writes to ./figures/."""
import os
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 300, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
os.makedirs("figures", exist_ok=True)

DATA_PATH = ("experiment_results/experiment_5496a7db2b6f451ea24f92a25c12ad8d_proc_1167167/"
             "experiment_data.npy")
try:
    experiment_data = np.load(DATA_PATH, allow_pickle=True).item()
except Exception as e:
    print("load error:", e); experiment_data = {}

d = experiment_data.get("N_tuning", {}).get("trivia_qa", {})
per_N = d.get("per_N", {})
Ns = sorted(per_N.keys())
layers = sorted({int(L) for N in Ns for L in per_N[N]["per_layer"].keys()}) if Ns else []
val_metrics = d.get("metrics", {}).get("val", [])
preds = d.get("predictions", [])
conf_all = np.asarray(d.get("verbalized_confidence", []), dtype=float)
corr_all = np.asarray(d.get("ground_truth", []), dtype=int)
print(f"Ns={Ns} layers={layers} n={len(conf_all)}")


def reliability(p, y, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])
    accs, confs, counts = [], [], []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        if m.sum() > 0:
            accs.append(y[m].mean()); confs.append(p[m].mean()); counts.append(int(m.sum()))
        else:
            accs.append(np.nan); confs.append(np.nan); counts.append(0)
    return centers, np.array(accs), np.array(confs), np.array(counts)


def ece(p, y, nb=10):
    if len(p) == 0: return np.nan
    _, a, c, cnt = reliability(p, y, nb); n = cnt.sum(); e = 0.0
    for A, C, K in zip(a, c, cnt):
        if K > 0 and not np.isnan(A): e += (K / n) * abs(A - C)
    return e


def brier(p, y): return float(np.mean((p - y) ** 2)) if len(p) else np.nan


def clean_conf(v):
    v = np.array(v, dtype=float) / 100.0
    if np.all(np.isnan(v)): return np.full_like(v, 0.9)
    m = np.nanmedian(v); v = np.where(np.isnan(v), m, v)
    return np.clip(v, 0, 1)


N_big = max(Ns) if Ns else None
entry_big = next((p for p in preds if p["N"] == N_big), None) if N_big is not None else None
mid_layer = layers[len(layers) // 2] if layers else None
cmap = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(layers), 1)))


# FIG 1: headline
try:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.0))
    ax = axes[0]
    for i, L in enumerate(layers):
        ys = [per_N[N]["per_layer"][int(L)]["ortho"] for N in Ns]
        yc = [per_N[N]["per_layer"][int(L)].get("rand_ctrl_ortho", np.nan) for N in Ns]
        ax.plot(Ns, ys, marker="o", color=cmap[i], lw=2, label=f"Layer {L} (probes)")
        ax.plot(Ns, yc, marker="x", ls="--", color=cmap[i], lw=1.2, alpha=0.6,
                label=f"Layer {L} (random control)")
    ax.axhline(1.0, color="k", ls=":", alpha=0.5)
    ax.set_xlabel("Number of TriviaQA samples (N)")
    ax.set_ylabel(r"$1 - |\cos(w_{corr}, w_{conf})|$")
    ax.set_title("(a) Subspace orthogonality of correctness\nand verbalized-confidence directions")
    ax.set_ylim(0, 1.08); ax.legend(fontsize=8, ncol=2, loc="lower center")

    if entry_big is not None and mid_layer is not None:
        pl = entry_big["per_layer"][int(mid_layer)]
        y_true = np.asarray(pl["y_true"])
        probe_p = np.asarray(pl["probe_prob"])
        test_idx = np.asarray(pl["test_idx"])
        conf_te = clean_conf(conf_all[test_idx])

        centers, accs, _, counts = reliability(conf_te, y_true)
        e_v = ece(conf_te, y_true); b_v = brier(conf_te, y_true)
        ax = axes[1]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfectly calibrated")
        ax.bar(centers, counts / max(counts.max(), 1) * 0.20, width=0.08,
               color="C3", alpha=0.25, label="bin density")
        ax.plot(centers, accs, marker="o", color="C3", lw=2, label="verbalized confidence")
        ax.set_xlabel("Verbalized confidence"); ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"(b) Reliability of verbalized confidence\nECE={e_v:.3f}, Brier={b_v:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=9)

        centers, accs, _, counts = reliability(probe_p, y_true)
        e_p = ece(probe_p, y_true); b_p = brier(probe_p, y_true)
        ax = axes[2]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfectly calibrated")
        ax.bar(centers, counts / max(counts.max(), 1) * 0.20, width=0.08,
               color="C0", alpha=0.25, label="bin density")
        ax.plot(centers, accs, marker="s", color="C0", lw=2, label=f"probe (Layer {mid_layer})")
        ax.set_xlabel("Probe-predicted probability"); ax.set_ylabel("Empirical accuracy")
        ax.set_title(f"(c) Reliability of linear correctness probe\nECE={e_p:.3f}, Brier={b_p:.3f}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.legend(fontsize=9)

    fig.suptitle("Verbalized vs. internal correctness signals in Llama-3.1-8B-Instruct on TriviaQA",
                 fontsize=15, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig1 headline orthogonality and reliability.png", bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("Fig1:", e); plt.close("all")


# FIG 2: probe scaling
try:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    for i, L in enumerate(layers):
        ys = [per_N[N]["per_layer"][int(L)]["probe_acc"] for N in Ns]
        ax.plot(Ns, ys, marker="s", color=cmap[i], lw=2, label=f"Layer {L}")
    if len(corr_all):
        maj = max(corr_all.mean(), 1 - corr_all.mean())
        ax.axhline(maj, color="k", ls=":", label=f"majority baseline ({maj:.2f})")
    ax.set_xlabel("Number of TriviaQA samples (N)"); ax.set_ylabel("Held-out accuracy")
    ax.set_title("(a) Correctness probe: test accuracy"); ax.legend()

    ax = axes[1]
    for i, L in enumerate(layers):
        ys = [per_N[N]["per_layer"][int(L)]["probe_auc"] for N in Ns]
        ax.plot(Ns, ys, marker="^", color=cmap[i], lw=2, label=f"Layer {L}")
    ax.axhline(0.5, color="k", ls=":", label="chance")
    ax.set_xlabel("Number of TriviaQA samples (N)"); ax.set_ylabel("ROC-AUC")
    ax.set_title("(b) Correctness probe: ROC-AUC"); ax.legend()

    fig.suptitle("Correctness is linearly decodable from hidden states across training scales",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig("figures/fig2 correctness probe scaling.png", bbox_inches="tight")
    plt.close(fig)
except Exception as e:
    print("Fig2:", e); plt.close("all")


# FIG 3: calibration gap
try:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    if val_metrics:
        Ns2 = [v["N"] for v in val_metrics]
        ece_v = [np.mean([lm["ece_verbalized"] for lm in v["layers"]]) for v in val_metrics]
        ece_p = [np.mean([lm["ece_probe"] for lm in v["layers"]]) for v in val_metrics]
        ax.plot(Ns2, ece_v, marker="o", lw=2, color="C3", label="verbalized confidence")
        ax.plot(Ns2, ece_p, marker="s", lw=2, color="C0", label="probe (layer-averaged)")
        ax.fill_between(Ns2, ece_p, ece_v, color="gray", alpha=0.18, label="calibration gap")
    ax.set_xlabel("Number of TriviaQA samples (N)")
    ax.set_ylabel("Expected Calibration Error")
    ax.set_title("(a) Calibration gap: verbalized vs. probe ECE across N")
    ax.legend()

    ax = axes[1]
    if val_metrics:
        v_last = val_metrics[-1]
        Ls = [lm["layer"] for lm in v_last["layers"]]
        eV = [lm["ece_verbalized"] for lm in v_last["layers"]]
        eP = [lm["ece_probe"] for lm in v_last["layers"]]
        x = np.arange(len(Ls)); w = 0.38
        ax.bar(x - w / 2, eV, w, color="C3", label="verbalized confidence")
        ax.bar(x + w / 2, eP, w, color="C0", label="probe")
        ax.set_xticks(x); ax.set_xticklabels([f"L{L}" for L in Ls])
    ax.set_xlabel("Transformer layer"); ax.set_ylabel("Expected Calibration Error")
    ax.set_title(