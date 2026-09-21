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
    experiment_data = None

if experiment_data is not None:
    root = experiment_data["TOP_K"]["imagenet_val_resnet50"]
    per_top_k = root["per_top_k"]
    top_k_values = root["top_k_values"]
    best_k = root["best_top_k"]
    inspected = root["inspected_channels"]
    ds_name = root["config"]["dataset"]
    model_name = root["config"]["model"]

    # 1) Summary curves across TOP_K
    try:
        plt.figure(figsize=(6, 4))
        xs = list(top_k_values)
        mc = [per_top_k[k]["mean_concept"] for k in xs]
        mr = [per_top_k[k]["mean_random"] for k in xs]
        md = [per_top_k[k]["delta"] for k in xs]
        plt.plot(xs, mc, "o-", label="concept (mean)")
        plt.plot(xs, mr, "s--", label="random baseline (mean)")
        plt.plot(xs, md, "^:", label="delta (concept - random)")
        plt.xscale("log", base=2)
        plt.xticks(xs, [str(x) for x in xs])
        plt.xlabel("TOP_K")
        plt.ylabel("mean pairwise CLIP cos sim")
        plt.title(f"TOP_K sweep summary\nDataset: {ds_name} | Model: {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_{model_name}_topk_summary.png"),
            dpi=140,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating summary plot: {e}")
        plt.close()

    # 2) Histogram of per-channel concept scores at best K
    try:
        cs = per_top_k[best_k]["concept_scores"]
        rs = per_top_k[best_k]["random_scores"]
        plt.figure(figsize=(6, 4))
        bins = np.linspace(min(rs.min(), cs.min()), max(rs.max(), cs.max()), 25)
        plt.hist(cs, bins=bins, alpha=0.6, label="concept", color="C0")
        plt.hist(rs, bins=bins, alpha=0.6, label="random", color="C1")
        plt.xlabel("mean pairwise CLIP cos sim")
        plt.ylabel("# channels")
        plt.title(
            f"Per-channel score distribution at TOP_K={best_k}\n"
            f"Left: concept, Right: random baseline | Dataset: {ds_name}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_{model_name}_hist_bestK{best_k}.png"),
            dpi=140,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating histogram plot: {e}")
        plt.close()

    # 3) Scatter: concept vs random per channel at best K
    try:
        cs = per_top_k[best_k]["concept_scores"]
        rs = per_top_k[best_k]["random_scores"]
        plt.figure(figsize=(5, 5))
        plt.scatter(rs, cs, s=20, alpha=0.7)
        lo = min(rs.min(), cs.min())
        hi = max(rs.max(), cs.max())
        plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="y=x")
        plt.xlabel("random baseline score")
        plt.ylabel("concept score")
        plt.title(
            f"Concept vs Random per channel (TOP_K={best_k})\n"
            f"Dataset: {ds_name} | Model: {model_name}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, f"{ds_name}_{model_name}_scatter_bestK{best_k}.png"
            ),
            dpi=140,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating scatter plot: {e}")
        plt.close()

    # 4) Top and bottom channels bar chart
    try:
        cs = per_top_k[best_k]["concept_scores"]
        order = np.argsort(-cs)
        top_n = 10
        top_idx = order[:top_n]
        bot_idx = order[-top_n:]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(range(top_n), cs[top_idx], color="C2")
        axes[0].set_xticks(range(top_n))
        axes[0].set_xticklabels([str(inspected[i]) for i in top_idx], rotation=45)
        axes[0].set_title("Left: Top-10 channels")
        axes[0].set_ylabel("concept score")
        axes[1].bar(range(top_n), cs[bot_idx], color="C3")
        axes[1].set_xticks(range(top_n))
        axes[1].set_xticklabels([str(inspected[i]) for i in bot_idx], rotation=45)
        axes[1].set_title("Right: Bottom-10 channels")
        fig.suptitle(
            f"Channel ranking by concept consistency (TOP_K={best_k})\n"
            f"Dataset: {ds_name} | Model: {model_name}"
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(
                working_dir, f"{ds_name}_{model_name}_top_bottom_channels.png"
            ),
            dpi=140,
        )
        plt.close(fig)
    except Exception as e:
        print(f"Error creating top/bottom plot: {e}")
        plt.close()

    # 5) Delta bar chart across TOP_K
    try:
        plt.figure(figsize=(6, 4))
        xs = list(top_k_values)
        md = [per_top_k[k]["delta"] for k in xs]
        plt.bar([str(x) for x in xs], md, color="C4")
        plt.xlabel("TOP_K")
        plt.ylabel("delta (concept - random)")
        plt.title(
            f"Concept-vs-random delta across TOP_K\n"
            f"Dataset: {ds_name} | Model: {model_name}"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"{ds_name}_{model_name}_delta_bar.png"), dpi=140
        )
        plt.close()
    except Exception as e:
        print(f"Error creating delta bar plot: {e}")
        plt.close()

    # Print evaluation metrics
    print("=== Evaluation Metrics ===")
    for k in top_k_values:
        d = per_top_k[k]
        print(
            f"TOP_K={k}: concept_mean={d['mean_concept']:.4f}, "
            f"concept_median={d['median_concept']:.4f}, "
            f"random_mean={d['mean_random']:.4f}, delta={d['delta']:.4f}"
        )
    print(f"Best TOP_K = {best_k}")
