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

try:
    d = experiment_data["token_position_pooling"]["trivia_qa"]
    variants = d["pool_variants"]
    layers = d["target_layers"]
    per_variant = d["per_variant"]
    gt = np.array(d["ground_truth"])
    verb = np.array(d["verbalized_confidence"])
except Exception as e:
    print(f"Error extracting: {e}")
    d = None

# Plot 1: Orthogonality by variant/layer
try:
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(layers))
    w = 0.2
    for i, v in enumerate(variants):
        ys = [per_variant[v]["per_layer"][int(L)]["ortho"] for L in layers]
        ax.bar(xs + (i - (len(variants) - 1) / 2) * w, ys, width=w, label=v)
        rc = [per_variant[v]["per_layer"][int(L)]["rand_ctrl_ortho"] for L in layers]
    # random control line (avg across variants, same value likely)
    ctrl = np.mean(
        [
            [per_variant[v]["per_layer"][int(L)]["rand_ctrl_ortho"] for L in layers]
            for v in variants
        ],
        axis=0,
    )
    ax.plot(xs, ctrl, "k--", label="random control")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylabel("1 - |cos(w_corr, w_conf)|")
    ax.set_title(
        "TriviaQA: Subspace Orthogonality by Pooling Variant\n(Bars: variants across layers; Dashed: random control)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(working_dir, "triviaqa_orthogonality_by_pool_variant.png"), dpi=140
    )
    plt.close(fig)
except Exception as e:
    print(f"Error plot1: {e}")
    plt.close()

# Plot 2: Probe accuracy and AUC side-by-side
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    xs = np.arange(len(layers))
    w = 0.2
    for i, v in enumerate(variants):
        accs = [per_variant[v]["per_layer"][int(L)]["probe_acc"] for L in layers]
        aucs = [per_variant[v]["per_layer"][int(L)]["probe_auc"] for L in layers]
        axes[0].bar(xs + (i - (len(variants) - 1) / 2) * w, accs, width=w, label=v)
        axes[1].bar(xs + (i - (len(variants) - 1) / 2) * w, aucs, width=w, label=v)
    for ax, ttl, ylab in zip(
        axes, ["Left: Probe Accuracy", "Right: Probe AUC"], ["Accuracy", "AUC"]
    ):
        ax.set_xticks(xs)
        ax.set_xticklabels([f"L{L}" for L in layers])
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.legend(fontsize=8)
    fig.suptitle("TriviaQA: Correctness Probe Performance by Pooling Variant")
    fig.tight_layout()
    fig.savefig(
        os.path.join(working_dir, "triviaqa_probe_acc_auc_by_pool_variant.png"), dpi=140
    )
    plt.close(fig)
except Exception as e:
    print(f"Error plot2: {e}")
    plt.close()

# Plot 3: Confidence MSE
try:
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(layers))
    w = 0.2
    for i, v in enumerate(variants):
        ys = [per_variant[v]["per_layer"][int(L)]["conf_mse"] for L in layers]
        ax.bar(xs + (i - (len(variants) - 1) / 2) * w, ys, width=w, label=v)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylabel("MSE")
    ax.set_title(
        "TriviaQA: Verbalized-Confidence Ridge Regression MSE by Pooling Variant"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(working_dir, "triviaqa_conf_mse_by_pool_variant.png"), dpi=140
    )
    plt.close(fig)
except Exception as e:
    print(f"Error plot3: {e}")
    plt.close()

# Plot 4: ECE comparison (verbalized vs probe)
try:
    fig, ax = plt.subplots(figsize=(8, 4))
    xs = np.arange(len(layers))
    w = 0.1
    n = len(variants)
    for i, v in enumerate(variants):
        ev = [per_variant[v]["per_layer"][int(L)]["ece_verbalized"] for L in layers]
        ep = [per_variant[v]["per_layer"][int(L)]["ece_probe"] for L in layers]
        ax.bar(xs + (2 * i - (2 * n - 1) / 2) * w, ev, width=w, label=f"{v} (verb)")
        ax.bar(
            xs + (2 * i + 1 - (2 * n - 1) / 2) * w, ep, width=w, label=f"{v} (probe)"
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylabel("ECE")
    ax.set_title(
        "TriviaQA: Expected Calibration Error\n(Left group per layer: Verbalized vs Probe by variant)"
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(working_dir, "triviaqa_ece_by_pool_variant.png"), dpi=140)
    plt.close(fig)
except Exception as e:
    print(f"Error plot4: {e}")
    plt.close()

# Plot 5: Verbalized confidence hist split by correctness
try:
    valid = ~np.isnan(verb)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        verb[valid & (gt == 1)],
        bins=20,
        alpha=0.6,
        label="correct",
        color="green",
        edgecolor="k",
    )
    ax.hist(
        verb[valid & (gt == 0)],
        bins=20,
        alpha=0.6,
        label="incorrect",
        color="red",
        edgecolor="k",
    )
    ax.set_xlabel("Verbalized confidence (%)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"TriviaQA: Verbalized Confidence Distribution\n(Overall accuracy={gt.mean():.2f})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(working_dir, "triviaqa_verbalized_confidence_hist.png"), dpi=140
    )
    plt.close(fig)
except Exception as e:
    print(f"Error plot5: {e}")
    plt.close()

# Print headline metrics
try:
    print("Headline subspace orthogonality by variant:")
    for v, s in d["headline_by_variant"].items():
        print(f"  {v}: {s:.4f}")
except Exception as e:
    print(f"Error headline: {e}")
