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

DATASET = "NARCBench-Core-Synthetic"
try:
    ds = experiment_data["layer_depth_probing"][DATASET]
except Exception as e:
    print(f"Cannot access dataset entry: {e}")
    ds = None

if ds is not None:
    layer_sweep = ds["hyperparams"]["layer_sweep"]
    layer_names = [k for k, _ in layer_sweep]
    layer_idxs = [v for _, v in layer_sweep]
    n_transformer = ds["hyperparams"]["n_transformer_layers"]
    runs = ds["runs"]
    all_results = ds["summary"]["all_results"]
    summary = ds["summary"]
    best_layer = summary["best_layer"]
    variants = [
        "agent1",
        "agent2",
        "agent3",
        "mean_pool",
        "max_pool",
        "concat",
        "tfidf_text",
        "random_features",
    ]
    group_aggs = ["mean_pool", "max_pool", "concat"]

    # Plot 1: AUROC vs layer index per variant
    try:
        fig, ax = plt.subplots(figsize=(10, 5))
        for name in variants:
            ys = [all_results[l][name] for l in layer_names]
            ax.plot(layer_idxs, ys, marker="o", label=name)
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.4, label="chance")
        ax.set_xlabel(f"Hidden-state layer index (0=embedding, {n_transformer}=final)")
        ax.set_ylabel("AUROC (5-fold CV)")
        ax.set_title(
            f"NARCBench-Core-Synthetic: AUROC vs Layer Depth\n(Probing across variants)"
        )
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "NARCBench_Core_auroc_vs_layer_depth.png"),
            dpi=120,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot1: {e}")
        plt.close()

    # Plot 2: bar chart at best layer
    try:
        best_results = all_results[best_layer]
        names_sorted = sorted(best_results.keys(), key=lambda k: -best_results[k])
        vals = [best_results[k] for k in names_sorted]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(names_sorted, vals)
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.5, label="chance")
        ax.set_ylabel("AUROC (5-fold CV)")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"NARCBench-Core-Synthetic: AUROC by Probe Variant\n(Best layer={best_layer}, idx={summary['best_layer_idx']})"
        )
        plt.xticks(rotation=35, ha="right")
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "NARCBench_Core_auroc_by_variant_best_layer.png"),
            dpi=120,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot2: {e}")
        plt.close()

    # Plot 3: per-fold AUROC boxplot for group aggregations at best layer
    try:
        folds_data = runs[best_layer]["auroc_folds"]
        data_to_plot = [folds_data[g] for g in group_aggs]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot(data_to_plot, labels=group_aggs)
        for i, d in enumerate(data_to_plot):
            ax.scatter([i + 1] * len(d), d, color="red", alpha=0.6)
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.4)
        ax.set_ylabel("AUROC (per fold)")
        ax.set_title(
            f"NARCBench-Core-Synthetic: Per-Fold AUROC for Group Aggregations\n(Best layer={best_layer})"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "NARCBench_Core_perfold_auroc_group_aggs.png"),
            dpi=120,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot3: {e}")
        plt.close()

    # Plot 4: heatmap AUROC across layers x variants
    try:
        M = np.array([[all_results[l][v] for l in layer_names] for v in variants])
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0.4, vmax=1.0)
        ax.set_xticks(range(len(layer_names)))
        ax.set_xticklabels(
            [f"{n}\n(idx={i})" for n, i in zip(layer_names, layer_idxs)],
            rotation=30,
            ha="right",
            fontsize=8,
        )
        ax.set_yticks(range(len(variants)))
        ax.set_yticklabels(variants)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{M[i,j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if M[i, j] < 0.7 else "black",
                    fontsize=7,
                )
        plt.colorbar(im, ax=ax, label="AUROC")
        ax.set_title(
            "NARCBench-Core-Synthetic: AUROC Heatmap\n(Rows: probe variants, Cols: hidden-state layers)"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "NARCBench_Core_auroc_heatmap.png"), dpi=120
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot4: {e}")
        plt.close()

    # Plot 5: score distribution by class at best layer for best group probe
    try:
        best_group = summary["best_group"]
        scores = np.array(runs[best_layer]["predictions"][best_group])
        labels = np.array(ds["ground_truth"])
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.hist(
            scores[labels == 0],
            bins=20,
            alpha=0.6,
            label="Independent (label=0)",
            color="#94a3b8",
        )
        ax.hist(
            scores[labels == 1],
            bins=20,
            alpha=0.6,
            label="Collusion (label=1)",
            color="#3b82f6",
        )
        ax.set_xlabel("Predicted probability (collusion)")
        ax.set_ylabel("Count")
        ax.set_title(
            f"NARCBench-Core-Synthetic: Prediction Score Distribution\n(Best layer={best_layer}, probe={best_group}, Left/Right: overlaid histograms by class)"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "NARCBench_Core_pred_score_distribution.png"),
            dpi=120,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot5: {e}")
        plt.close()

    # Print evaluation metric
    try:
        print(f"collusion_detection_auroc = {summary['collusion_detection_auroc']:.4f}")
        print(
            f"best_layer = {best_layer} (idx={summary['best_layer_idx']}), best_group = {summary['best_group']}"
        )
    except Exception as e:
        print(f"Error printing summary: {e}")
