import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2026-07-13_11-03-14_multi_agent_attempt_0/logs/0-run/experiment_results/experiment_01f68a07cd13452dba7d3c36016a04e5_proc_1477688/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        ed = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")

DATASET = "NARCBench-Core-Synthetic"

# Gather per-run structures
runs_list = []
for ed in all_experiment_data:
    try:
        ds = ed["layer_depth_probing"][DATASET]
        runs_list.append(ds)
    except Exception as e:
        print(f"Skipping ed: {e}")

if not runs_list:
    print("No runs available.")
else:
    # Use first run structure for metadata
    ref = runs_list[0]
    layer_sweep = ref["hyperparams"]["layer_sweep"]
    layer_names = [k for k, _ in layer_sweep]
    layer_idxs = [v for _, v in layer_sweep]
    n_transformer = ref["hyperparams"]["n_transformer_layers"]
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
    best_layer = ref["summary"]["best_layer"]

    # Aggregate per-fold AUROC across all runs => for each (layer, variant), collect fold values
    agg = {l: {v: [] for v in variants} for l in layer_names}
    for ds in runs_list:
        for l in layer_names:
            folds = ds["runs"][l].get("auroc_folds", {})
            for v in variants:
                if v in folds:
                    agg[l][v].extend(list(folds[v]))
                elif v in ds["summary"]["all_results"].get(l, {}):
                    agg[l][v].append(ds["summary"]["all_results"][l][v])

    def mean_sem(arr):
        arr = np.asarray(arr, dtype=float)
        if arr.size == 0:
            return np.nan, np.nan
        m = np.nanmean(arr)
        s = np.nanstd(arr, ddof=1) / np.sqrt(arr.size) if arr.size > 1 else 0.0
        return m, s

    # Plot 1: AUROC vs layer with SEM error bars
    try:
        fig, ax = plt.subplots(figsize=(11, 6))
        for v in variants:
            means, sems = [], []
            for l in layer_names:
                m, s = mean_sem(agg[l][v])
                means.append(m)
                sems.append(s)
            ax.errorbar(
                layer_idxs,
                means,
                yerr=sems,
                marker="o",
                capsize=3,
                label=f"{v} (mean±SEM)",
            )
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.4, label="chance")
        ax.set_xlabel(f"Hidden-state layer index (0=embedding, {n_transformer}=final)")
        ax.set_ylabel("AUROC (mean over folds, ±SEM)")
        ax.set_title(
            "NARCBench-Core-Synthetic: Aggregated AUROC vs Layer Depth\n(Mean ± SEM across CV folds)"
        )
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "NARCBench_Core_agg_auroc_vs_layer_depth.png"),
            dpi=120,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot1: {e}")
        plt.close()

    # Plot 2: Bar chart at best layer with SEM
    try:
        means = []
        sems = []
        for v in variants:
            m, s = mean_sem(agg[best_layer][v])
            means.append(m)
            sems.append(s)
        order = np.argsort(-np.array(means))
        names_sorted = [variants[i] for i in order]
        m_sorted = [means[i] for i in order]
        s_sorted = [sems[i] for i in order]
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            names_sorted, m_sorted, yerr=s_sorted, capsize=4, label="Mean AUROC ± SEM"
        )
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.5, label="chance")
        ax.set_ylabel("AUROC (mean over folds)")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"NARCBench-Core-Synthetic: AUROC by Probe Variant (Best layer={best_layer})\nBars: mean, Error bars: SEM across CV folds"
        )
        plt.xticks(rotation=35, ha="right")
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "NARCBench_Core_agg_auroc_by_variant_best_layer.png"
            ),
            dpi=120,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot2: {e}")
        plt.close()

    # Plot 3: Aggregated heatmap of mean AUROC
    try:
        M = np.array([[mean_sem(agg[l][v])[0] for l in layer_names] for v in variants])
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
        plt.colorbar(im, ax=ax, label="Mean AUROC")
        ax.set_title(
            "NARCBench-Core-Synthetic: Mean AUROC Heatmap\n(Rows: probe variants, Cols: hidden-state layers)"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "NARCBench_Core_agg_auroc_heatmap.png"), dpi=120
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot3: {e}")
        plt.close()

    # Plot 4: Boxplot per-fold at best layer for group aggregations
    try:
        data_to_plot = [agg[best_layer][g] for g in group_aggs]
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.boxplot(data_to_plot, labels=group_aggs)
        for i, d in enumerate(data_to_plot):
            ax.scatter(
                [i + 1] * len(d),
                d,
                color="red",
                alpha=0.6,
                label="fold values" if i == 0 else None,
            )
        # add mean markers
        means = [np.nanmean(d) if len(d) > 0 else np.nan for d in data_to_plot]
        ax.scatter(
            range(1, len(group_aggs) + 1),
            means,
            color="blue",
            marker="D",
            s=60,
            label="mean",
        )
        ax.axhline(0.5, color="k", linestyle="--", alpha=0.4)
        ax.set_ylabel("AUROC (per fold, aggregated across runs)")
        ax.set_title(
            f"NARCBench-Core-Synthetic: Aggregated Per-Fold AUROC for Group Aggregations\n(Best layer={best_layer})"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "NARCBench_Core_agg_perfold_auroc_group_aggs.png"
            ),
            dpi=120,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot4: {e}")
        plt.close()

    # Plot 5: Prediction score distribution at best layer (from first run - class-conditional overlay)
    try:
        best_group = ref["summary"]["best_group"]
        scores_all = []
        labels_all = []
        for ds in runs_list:
            preds = ds["runs"][best_layer].get("predictions", {})
            if best_group in preds:
                scores_all.append(np.array(preds[best_group]))
                labels_all.append(np.array(ds["ground_truth"]))
        if scores_all:
            scores = np.concatenate(scores_all)
            labels = np.concatenate(labels_all)
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
            ax.set_ylabel("Count (aggregated across runs)")
            ax.set_title(
                f"NARCBench-Core-Synthetic: Aggregated Prediction Score Distribution\n(Best layer={best_layer}, probe={best_group})"
            )
            ax.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, "NARCBench_Core_agg_pred_score_distribution.png"
                ),
                dpi=120,
            )
            plt.close()
    except Exception as e:
        print(f"Error creating plot5: {e}")
        plt.close()

    # Print aggregated evaluation metric
    try:
        aurocs = [
            ds["summary"]["collusion_detection_auroc"]
            for ds in runs_list
            if "collusion_detection_auroc" in ds["summary"]
        ]
        if aurocs:
            m = np.mean(aurocs)
            s = (
                np.std(aurocs, ddof=1) / np.sqrt(len(aurocs))
                if len(aurocs) > 1
                else 0.0
            )
            print(
                f"collusion_detection_auroc (mean ± SEM across {len(aurocs)} runs) = {m:.4f} ± {s:.4f}"
            )
        best_group = ref["summary"]["best_group"]
        print(f"best_layer = {best_layer}, best_group = {best_group}")
        # Also print aggregated best-layer mean per variant
        print("Aggregated mean±SEM at best layer:")
        for v in variants:
            m, s = mean_sem(agg[best_layer][v])
            print(f"  {v}: {m:.4f} ± {s:.4f}")
    except Exception as e:
        print(f"Error printing summary: {e}")
