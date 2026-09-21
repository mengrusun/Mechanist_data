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

ds_key = "trivia_qa"
data = experiment_data.get(ds_key, {})
val_metrics = data.get("metrics", {}).get("val", [])
losses_val = data.get("losses", {}).get("val", [])
correct = np.array(data.get("ground_truth", []))
conf_arr = np.array(data.get("verbalized_confidence", []))
valid_conf = ~np.isnan(conf_arr)

layers = [m["layer"] for m in val_metrics]

# Plot 1: Orthogonality by layer vs random control
try:
    ortho = [m["ortho"] for m in val_metrics]
    ctrl = [m["rand_ctrl_ortho"] for m in val_metrics]
    x = np.arange(len(layers))
    w = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - w / 2, ortho, w, label="Probe pair (corr vs conf)", color="steelblue")
    plt.bar(x + w / 2, ctrl, w, label="Random control", color="gray")
    plt.xticks(x, [str(L) for L in layers])
    plt.xlabel("Layer")
    plt.ylabel("1 - |cos|")
    plt.title(
        "TriviaQA: Subspace Orthogonality by Layer\nProbe pair vs Random-direction control"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_orthogonality_vs_control.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating orthogonality plot: {e}")
    plt.close()

# Plot 2: Probe accuracy and AUC by layer
try:
    accs = [m["probe_acc"] for m in val_metrics]
    aucs = [m["probe_auc"] for m in val_metrics]
    x = np.arange(len(layers))
    w = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - w / 2, accs, w, label="Accuracy", color="seagreen")
    plt.bar(x + w / 2, aucs, w, label="AUC", color="orange")
    plt.xticks(x, [str(L) for L in layers])
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.title("TriviaQA: Correctness Probe Performance by Layer")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_probe_acc_auc_by_layer.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating probe metrics plot: {e}")
    plt.close()

# Plot 3: ECE verbalized vs probe by layer
try:
    ece_v = [m["ece_verbalized"] for m in val_metrics]
    ece_p = [m["ece_probe"] for m in val_metrics]
    x = np.arange(len(layers))
    w = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(x - w / 2, ece_v, w, label="Verbalized confidence", color="salmon")
    plt.bar(x + w / 2, ece_p, w, label="Learned probe", color="steelblue")
    plt.xticks(x, [str(L) for L in layers])
    plt.xlabel("Layer")
    plt.ylabel("ECE (lower is better)")
    plt.title(
        "TriviaQA: Calibration Error by Layer\nLeft bars: Verbalized, Right bars: Probe"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "triviaqa_ece_verb_vs_probe.png"), dpi=140)
    plt.close()
except Exception as e:
    print(f"Error creating ECE plot: {e}")
    plt.close()

# Plot 4: Verbalized confidence histogram split by correctness
try:
    plt.figure(figsize=(6, 4))
    if valid_conf.any():
        c_correct = conf_arr[valid_conf & (correct == 1)]
        c_wrong = conf_arr[valid_conf & (correct == 0)]
        bins = np.linspace(0, 100, 21)
        plt.hist(
            c_correct,
            bins=bins,
            alpha=0.6,
            label=f"Correct (n={len(c_correct)})",
            color="seagreen",
        )
        plt.hist(
            c_wrong,
            bins=bins,
            alpha=0.6,
            label=f"Incorrect (n={len(c_wrong)})",
            color="salmon",
        )
    plt.xlabel("Verbalized confidence (%)")
    plt.ylabel("Count")
    plt.title(
        f"TriviaQA: Verbalized Confidence Distribution\nOverall accuracy={correct.mean():.2f}"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_confidence_hist_by_correctness.png"),
        dpi=140,
    )
    plt.close()
except Exception as e:
    print(f"Error creating conf hist plot: {e}")
    plt.close()

# Plot 5: Reliability diagram for verbalized confidence
try:
    plt.figure(figsize=(5, 5))
    if valid_conf.any():
        probs = conf_arr[valid_conf] / 100.0
        labs = correct[valid_conf]
        bins = np.linspace(0, 1, 11)
        centers, accs_b, counts = [], [], []
        for i in range(len(bins) - 1):
            m = (probs >= bins[i]) & (
                probs <= bins[i + 1] if i == len(bins) - 2 else probs < bins[i + 1]
            )
            if m.sum() > 0:
                centers.append(probs[m].mean())
                accs_b.append(labs[m].mean())
                counts.append(m.sum())
        plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        plt.scatter(
            centers,
            accs_b,
            s=[max(20, c * 5) for c in counts],
            color="salmon",
            label="Verbalized (size ~ count)",
        )
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(
        "TriviaQA: Reliability Diagram\nVerbalized confidence vs empirical accuracy"
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "triviaqa_reliability_diagram.png"), dpi=140)
    plt.close()
except Exception as e:
    print(f"Error creating reliability plot: {e}")
    plt.close()

# Print headline metric
try:
    print(
        f"Headline subspace orthogonality (layer {data.get('headline_layer')}): "
        f"{data.get('subspace_orthogonality_score'):.4f}"
    )
    print(f"Overall accuracy: {correct.mean():.4f}" if len(correct) else "No labels.")
except Exception as e:
    print(f"Error printing metrics: {e}")
