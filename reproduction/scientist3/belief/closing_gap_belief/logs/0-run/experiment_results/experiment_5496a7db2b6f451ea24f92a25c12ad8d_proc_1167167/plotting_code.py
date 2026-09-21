import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

d = experiment_data.get("N_tuning", {}).get("trivia_qa", {})
per_N = d.get("per_N", {})
Ns = sorted(per_N.keys())
layers = (
    sorted({int(L) for N in Ns for L in per_N[N]["per_layer"].keys()}) if Ns else []
)

# Plot 1: Orthogonality vs N
try:
    plt.figure(figsize=(7, 4))
    for L in layers:
        ys = [per_N[N]["per_layer"][int(L)]["ortho"] for N in Ns]
        plt.plot(Ns, ys, marker="o", label=f"Layer {L}")
        yc = [per_N[N]["per_layer"][int(L)].get("rand_ctrl_ortho", np.nan) for N in Ns]
        plt.plot(Ns, yc, marker="x", linestyle="--", alpha=0.5, label=f"Rand ctrl L{L}")
    plt.xlabel("N (TriviaQA samples)")
    plt.ylabel("1 - |cos(w_corr, w_conf)|")
    plt.title(
        "TriviaQA: Subspace Orthogonality vs N\n(Solid: Probe pairs, Dashed: Random control)"
    )
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "triviaqa_orthogonality_vs_N.png"), dpi=140)
    plt.close()
except Exception as e:
    print(f"Error creating orthogonality plot: {e}")
    plt.close()

# Plot 2: Probe accuracy vs N
try:
    plt.figure(figsize=(7, 4))
    for L in layers:
        ys = [per_N[N]["per_layer"][int(L)]["probe_acc"] for N in Ns]
        plt.plot(Ns, ys, marker="s", label=f"Layer {L}")
    plt.xlabel("N")
    plt.ylabel("Probe accuracy (test)")
    plt.title("TriviaQA: Correctness Probe Accuracy vs N")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "triviaqa_probe_acc_vs_N.png"), dpi=140)
    plt.close()
except Exception as e:
    print(f"Error creating probe acc plot: {e}")
    plt.close()

# Plot 3: Probe AUC vs N
try:
    plt.figure(figsize=(7, 4))
    for L in layers:
        ys = [per_N[N]["per_layer"][int(L)]["probe_auc"] for N in Ns]
        plt.plot(Ns, ys, marker="^", label=f"Layer {L}")
    plt.xlabel("N")
    plt.ylabel("Probe ROC-AUC (test)")
    plt.title("TriviaQA: Correctness Probe AUC vs N")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "triviaqa_probe_auc_vs_N.png"), dpi=140)
    plt.close()
except Exception as e:
    print(f"Error creating AUC plot: {e}")
    plt.close()

# Plot 4: ECE (verbalized vs probe) vs N averaged over layers
try:
    val = d.get("metrics", {}).get("val", [])
    Ns2 = [v["N"] for v in val]
    ece_v = [np.mean([lm["ece_verbalized"] for lm in v["layers"]]) for v in val]
    ece_p = [np.mean([lm["ece_probe"] for lm in v["layers"]]) for v in val]
    plt.figure(figsize=(7, 4))
    plt.plot(Ns2, ece_v, marker="o", label="Verbalized ECE")
    plt.plot(Ns2, ece_p, marker="s", label="Probe ECE (layer-avg)")
    plt.xlabel("N")
    plt.ylabel("Expected Calibration Error")
    plt.title(
        "TriviaQA: Calibration Error vs N\n(Left curve: verbalized confidence, Right/other: probe)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "triviaqa_ece_vs_N.png"), dpi=140)
    plt.close()
except Exception as e:
    print(f"Error creating ECE plot: {e}")
    plt.close()

# Plot 5: Verbalized confidence histogram
try:
    conf = np.array(d.get("verbalized_confidence", []), dtype=float)
    corr = np.array(d.get("ground_truth", []), dtype=int)
    valid = ~np.isnan(conf)
    plt.figure(figsize=(6, 4))
    plt.hist(conf[valid], bins=20, color="salmon", edgecolor="k")
    acc = corr.mean() if len(corr) else float("nan")
    plt.xlabel("Verbalized confidence (%)")
    plt.ylabel("Count")
    plt.title(
        f"TriviaQA: Verbalized Confidence Distribution\n(overall accuracy={acc:.2f})"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_verbalized_confidence_hist.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating hist plot: {e}")
    plt.close()

# Plot 6: Reliability diagram at largest N (verbalized vs probe mid-layer)
try:
    if Ns:
        N_big = max(Ns)
        preds = d.get("predictions", [])
        entry = next((p for p in preds if p["N"] == N_big), None)
        val_entry = next(
            (v for v in d.get("metrics", {}).get("val", []) if v["N"] == N_big), None
        )
        mid_layer = layers[len(layers) // 2]
        pl = entry["per_layer"][int(mid_layer)]
        y_true = np.array(pl["y_true"])
        probe_prob = np.array(pl["probe_prob"])
        test_idx = np.array(pl["test_idx"])
        conf_te = np.clip(np.array(d["verbalized_confidence"])[test_idx] / 100.0, 0, 1)
        conf_te = np.where(np.isnan(conf_te), np.nanmedian(conf_te), conf_te)

        bins = np.linspace(0, 1, 11)
        centers = 0.5 * (bins[:-1] + bins[1:])

        def reliab(p, y):
            out = []
            for i in range(len(bins) - 1):
                m = (p >= bins[i]) & (
                    p < bins[i + 1] if i < len(bins) - 2 else p <= bins[i + 1]
                )
                out.append(y[m].mean() if m.sum() > 0 else np.nan)
            return np.array(out)

        r_v = reliab(conf_te, y_true)
        r_p = reliab(probe_prob, y_true)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[0].plot(centers, r_v, marker="o", color="C1")
        axes[0].set_title("Left: Verbalized Confidence")
        axes[0].set_xlabel("Predicted conf")
        axes[0].set_ylabel("Empirical accuracy")
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
        axes[1].plot(centers, r_p, marker="s", color="C2")
        axes[1].set_title(f"Right: Probe (Layer {mid_layer})")
        axes[1].set_xlabel("Predicted prob")
        axes[1].set_ylabel("Empirical accuracy")
        fig.suptitle(f"TriviaQA: Reliability Diagrams (N={N_big})")
        fig.tight_layout()
        fig.savefig(
            os.path.join(working_dir, "triviaqa_reliability_diagram.png"), dpi=140
        )
        plt.close(fig)
except Exception as e:
    print(f"Error creating reliability plot: {e}")
    plt.close()

# Print headline metrics
try:
    print("Headline orthogonality by N:", d.get("headline_by_N", {}))
except Exception as e:
    print(f"Error printing metrics: {e}")
