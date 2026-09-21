import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    root = experiment_data["multi_dataset_circuit_generalization"]
    summary = root.get("_summary", {})
    ds_names = summary.get("dataset_names", [k for k in root.keys() if k != "_summary"])
    Ks = summary.get("Ks", [5, 10, 20, 40, 80, 160])

    # Plot 1: Head effect heatmaps per dataset
    try:
        fig, axes = plt.subplots(1, len(ds_names), figsize=(5 * len(ds_names), 5))
        if len(ds_names) == 1:
            axes = [axes]
        for idx, ds in enumerate(ds_names):
            he = np.array(root[ds]["head_effects"])
            vmax = max(abs(he).max(), 1e-6)
            im = axes[idx].imshow(
                he, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax
            )
            axes[idx].set_title(f"{ds}")
            axes[idx].set_xlabel("Head")
            axes[idx].set_ylabel("Layer")
            plt.colorbar(im, ax=axes[idx])
        fig.suptitle(
            "Attention Head Patching Effects (Recovery) - Logic Reasoning Datasets"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "logic_reasoning_head_effects_heatmap.png"),
            dpi=100,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating head effects plot: {e}")
        plt.close()

    # Plot 2: MLP effects per dataset (bar plots)
    try:
        fig, axes = plt.subplots(1, len(ds_names), figsize=(5 * len(ds_names), 4))
        if len(ds_names) == 1:
            axes = [axes]
        for idx, ds in enumerate(ds_names):
            me = np.array(root[ds]["mlp_effects"])
            axes[idx].bar(range(len(me)), me)
            axes[idx].set_title(f"{ds}")
            axes[idx].set_xlabel("Layer")
            axes[idx].set_ylabel("Recovery")
        fig.suptitle("MLP Patching Effects Per Layer - Logic Reasoning Datasets")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "logic_reasoning_mlp_effects_bars.png"), dpi=100
        )
        plt.close()
    except Exception as e:
        print(f"Error creating MLP effects plot: {e}")
        plt.close()

    # Plot 3: Faithfulness vs K per dataset
    try:
        plt.figure(figsize=(8, 5))
        for ds in ds_names:
            faith = root[ds]["faithfulness"]
            ks = sorted(faith.keys())
            match = [faith[k]["match"] for k in ks]
            faith_vals = [faith[k]["faithfulness"] for k in ks]
            plt.plot(ks, match, marker="o", label=f"{ds} match")
            plt.plot(ks, faith_vals, marker="x", linestyle="--", label=f"{ds} faith")
        plt.xlabel("K (num components)")
        plt.ylabel("Score")
        plt.title(
            "Circuit Faithfulness vs K - Logic Reasoning Datasets\n(solid: match rate, dashed: faithfulness = match * sparsity)"
        )
        plt.legend(fontsize=7, ncol=2)
        plt.xscale("log")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "logic_reasoning_faithfulness_vs_K.png"), dpi=100
        )
        plt.close()
    except Exception as e:
        print(f"Error creating faithfulness plot: {e}")
        plt.close()

    # Plot 4: Jaccard overlap matrix
    try:
        jm = summary.get("jaccard_matrix")
        if jm is not None:
            jm = np.array(jm)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(jm, cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(len(ds_names)))
            ax.set_xticklabels(ds_names, rotation=45, ha="right")
            ax.set_yticks(range(len(ds_names)))
            ax.set_yticklabels(ds_names)
            for i in range(len(ds_names)):
                for j in range(len(ds_names)):
                    ax.text(
                        j, i, f"{jm[i,j]:.2f}", ha="center", va="center", color="white"
                    )
            ax.set_title(
                "Jaccard Overlap of Top-K Circuit Components\nLogic Reasoning Datasets"
            )
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "logic_reasoning_jaccard_matrix.png"), dpi=100
            )
            plt.close()
    except Exception as e:
        print(f"Error creating Jaccard plot: {e}")
        plt.close()

    # Plot 5: Cross-dataset faithfulness matrix
    try:
        cfm = summary.get("cross_faithfulness_matrix")
        K_cross = summary.get("cross_faithfulness_K", 40)
        if cfm is not None:
            mat = np.zeros((len(ds_names), len(ds_names)))
            for i, c in enumerate(ds_names):
                for j, t in enumerate(ds_names):
                    mat[i, j] = cfm[c][t]["faithfulness"]
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(mat, cmap="viridis")
            ax.set_xticks(range(len(ds_names)))
            ax.set_xticklabels(ds_names, rotation=45, ha="right")
            ax.set_yticks(range(len(ds_names)))
            ax.set_yticklabels(ds_names)
            ax.set_xlabel("Test dataset")
            ax.set_ylabel("Circuit source dataset")
            for i in range(len(ds_names)):
                for j in range(len(ds_names)):
                    ax.text(
                        j, i, f"{mat[i,j]:.2f}", ha="center", va="center", color="white"
                    )
            ax.set_title(
                f"Cross-Dataset Circuit Faithfulness @ K={K_cross}\nLogic Reasoning Datasets"
            )
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            plt.savefig(
                os.path.join(
                    working_dir, "logic_reasoning_cross_dataset_faithfulness.png"
                ),
                dpi=100,
            )
            plt.close()
    except Exception as e:
        print(f"Error creating cross-dataset plot: {e}")
        plt.close()

    # Plot 6: Baseline accuracy and logit diffs bar chart
    try:
        clean_accs = []
        corr_accs = []
        clean_lds = []
        corr_lds = []
        for ds in ds_names:
            v = root[ds]["metrics"]["val"][0]
            clean_accs.append(v["clean_acc"])
            corr_accs.append(v["corr_acc"])
            clean_lds.append(v["mean_clean_logit_diff"])
            corr_lds.append(v["mean_corr_logit_diff"])
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        x = np.arange(len(ds_names))
        w = 0.35
        axes[0].bar(x - w / 2, clean_accs, w, label="clean")
        axes[0].bar(x + w / 2, corr_accs, w, label="corrupted")
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(ds_names, rotation=45, ha="right")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Left: Baseline Accuracy (clean vs corrupted)")
        axes[0].legend()
        axes[1].bar(x - w / 2, clean_lds, w, label="clean")
        axes[1].bar(x + w / 2, corr_lds, w, label="corrupted")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(ds_names, rotation=45, ha="right")
        axes[1].set_ylabel("Mean logit diff (True - False)")
        axes[1].set_title("Right: Mean Logit Difference")
        axes[1].legend()
        fig.suptitle("Baseline Model Performance - Logic Reasoning Datasets")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "logic_reasoning_baseline_performance.png"),
            dpi=100,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating baseline plot: {e}")
        plt.close()

    # Plot 7: Role counts of top-20 components stacked bar
    try:
        roles = ["early_fact_reading", "mid_rule_composition", "late_answer_projection"]
        data = np.array(
            [[root[ds]["role_counts_top20"][r] for r in roles] for ds in ds_names]
        )
        fig, ax = plt.subplots(figsize=(7, 4))
        bottom = np.zeros(len(ds_names))
        colors = ["tab:blue", "tab:orange", "tab:green"]
        for i, r in enumerate(roles):
            ax.bar(ds_names, data[:, i], bottom=bottom, label=r, color=colors[i])
            bottom += data[:, i]
        ax.set_ylabel("Count in Top-20")
        ax.set_title(
            "Layer Role Distribution of Top-20 Circuit Components\nLogic Reasoning Datasets"
        )
        ax.legend(fontsize=8)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "logic_reasoning_role_distribution.png"), dpi=100
        )
        plt.close()
    except Exception as e:
        print(f"Error creating role distribution plot: {e}")
        plt.close()

    # Print summary metrics
    try:
        print("\n=== Summary Metrics ===")
        for ds in ds_names:
            v = root[ds]["metrics"]["val"][0]
            print(f'{ds}: clean_acc={v["clean_acc"]:.3f} corr_acc={v["corr_acc"]:.3f}')
        print("\nOwn faithfulness @ K=40:")
        for ds in ds_names:
            f = root[ds]["faithfulness"].get(40, {})
            if f:
                print(f'  {ds}: match={f["match"]:.3f} faith={f["faithfulness"]:.3f}')
    except Exception as e:
        print(f"Error printing summary: {e}")

print("Plotting done.")
