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

ds_key = "gemma2_2b_gemmascope_res_16k"
data = experiment_data.get(ds_key, {})
layer = data.get("layer", "NA")
features = data.get("features", [])
per_feature = data.get("per_feature", {})
summary = data.get("summary", {})

# Plot 1: Mean gen accuracy - SAGE iters vs Neuronpedia
try:
    np_mean = summary.get("neuronpedia_mean_gen_acc", 0.0)
    sage_means = summary.get("sage_mean_gen_acc_per_iter", [])
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Neuronpedia"] + [f"SAGE it{i}" for i in range(len(sage_means))]
    vals = [np_mean] + list(sage_means)
    colors = ["gray"] + ["C0"] * len(sage_means)
    ax.bar(labels, vals, color=colors)
    ax.set_ylabel("Generative Accuracy")
    ax.set_ylim(0, 1)
    ax.set_title(
        f"Mean Generative Accuracy: SAGE vs Neuronpedia\nDataset: Gemma-2-2B GemmaScope-Res-16k (Layer {layer})"
    )
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "gemma2_2b_mean_gen_accuracy_bar.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Per-feature accuracy across iterations (line plot)
try:
    fig, ax = plt.subplots(figsize=(9, 5))
    for fid in features:
        pf = per_feature.get(fid, {})
        iters = pf.get("iterations", [])
        accs = [it["gen_acc"] for it in iters]
        ax.plot(range(len(accs)), accs, marker="o", label=f"feat {fid} (SAGE)")
        np_acc = pf.get("neuronpedia", {}).get("gen_acc", None)
        if np_acc is not None:
            ax.axhline(
                np_acc,
                linestyle="--",
                alpha=0.5,
                label=f"feat {fid} Neuronpedia={np_acc:.2f}",
            )
    ax.set_xlabel("SAGE Iteration")
    ax.set_ylabel("Generative Accuracy")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title(
        f"Per-Feature Generative Accuracy across SAGE Iterations\nDataset: Gemma-2-2B GemmaScope-Res-16k (Layer {layer})"
    )
    ax.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "gemma2_2b_per_feature_accuracy_curves.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Per-probe activation heatmap for each feature (final SAGE iter)
try:
    n_feats = len(features)
    if n_feats > 0:
        fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 4))
        if n_feats == 1:
            axes = [axes]
        for ax, fid in zip(axes, features):
            pf = per_feature.get(fid, {})
            iters = pf.get("iterations", [])
            np_pp = pf.get("neuronpedia", {}).get("per_probe_acts", [])
            rows = [np_pp] + [it["per_probe_acts"] for it in iters]
            row_labels = ["Neuronpedia"] + [f"SAGE it{i}" for i in range(len(iters))]
            arr = np.array(rows, dtype=float)
            im = ax.imshow(arr, aspect="auto", cmap="viridis")
            thr = data.get("thresholds", {}).get(fid, 0)
            ax.set_title(f"Feature {fid}\nThr={thr:.3f}")
            ax.set_yticks(range(len(row_labels)))
            ax.set_yticklabels(row_labels, fontsize=8)
            ax.set_xlabel("Probe idx")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            f"Per-Probe Max Activations (Rows: Explanation Source, Cols: Probes)\nDataset: Gemma-2-2B GemmaScope-Res-16k (Layer {layer})"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemma2_2b_per_probe_activation_heatmap.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Validation loss (1 - final SAGE acc) per feature
try:
    val_losses = data.get("losses", {}).get("val", [])
    if val_losses:
        fig, ax = plt.subplots(figsize=(7, 4))
        xs = [f"feat {f}" for f in features[: len(val_losses)]]
        ax.plot(xs, val_losses, marker="s", color="C3")
        ax.set_ylabel("Validation Loss (1 - final gen_acc)")
        ax.set_title(
            f"Validation Loss per Feature (final SAGE iter)\nDataset: Gemma-2-2B GemmaScope-Res-16k (Layer {layer})"
        )
        ax.set_ylim(-0.05, 1.05)
        for i, v in enumerate(val_losses):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemma2_2b_val_loss_per_feature.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Print summary metric
try:
    print(
        f"generative_accuracy (SAGE final) = {summary.get('sage_final_gen_acc', 'NA')}"
    )
    print(f"neuronpedia_mean_gen_acc = {summary.get('neuronpedia_mean_gen_acc', 'NA')}")
    print(f"delta = {summary.get('delta', 'NA')}")
except Exception as e:
    print(f"Error printing summary: {e}")
