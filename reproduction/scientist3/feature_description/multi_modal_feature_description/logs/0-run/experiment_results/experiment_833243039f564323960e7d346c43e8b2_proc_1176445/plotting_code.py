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
    experiment_data = {}

root = experiment_data.get("N_PROBE", {}).get("imagenet_val_resnet50", {})
n_probe_values = root.get("n_probe_values", [])
per_n = root.get("per_n_probe", {})
best_n = root.get("best_n_probe", None)
val_metrics = root.get("metrics", {}).get("val", [])

# Plot 1: N_PROBE sweep - concept vs random baseline
try:
    xs = [m["N_PROBE"] for m in val_metrics]
    ys_c = [m["concept_consistency_score_mean"] for m in val_metrics]
    ys_r = [m["random_baseline_mean"] for m in val_metrics]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys_c, "o-", label="Concept (mean)")
    plt.plot(xs, ys_r, "s--", label="Random baseline (mean)")
    plt.xscale("log")
    plt.xlabel("N_PROBE")
    plt.ylabel("Mean pairwise CLIP cosine similarity")
    plt.title(
        "ImageNet-val ResNet-50: Concept Consistency vs N_PROBE\nLeft-to-right: increasing probe pool size"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "imagenet_val_resnet50_nprobe_sweep.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Delta vs N_PROBE
try:
    xs = [m["N_PROBE"] for m in val_metrics]
    deltas = [m["delta"] for m in val_metrics]
    plt.figure(figsize=(6, 4))
    plt.bar([str(x) for x in xs], deltas, color="steelblue")
    plt.xlabel("N_PROBE")
    plt.ylabel("Delta (concept - random)")
    plt.title(
        "ImageNet-val ResNet-50: Concept Score Improvement Over Random\nBars: mean(concept) - mean(random) per N_PROBE"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "imagenet_val_resnet50_delta_bar.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Histogram of per-channel concept scores at best N_PROBE
try:
    d = per_n[best_n]
    cs = np.asarray(d["concept_scores"])
    rs = np.asarray(d["random_scores"])
    plt.figure(figsize=(6, 4))
    plt.hist(cs, bins=20, alpha=0.6, label="Concept", color="tab:blue")
    plt.hist(rs, bins=20, alpha=0.6, label="Random", color="tab:orange")
    plt.xlabel("Mean pairwise CLIP cosine similarity")
    plt.ylabel("Number of channels")
    plt.title(
        f"ImageNet-val ResNet-50: Per-Channel Score Distribution (N_PROBE={best_n})\nLeft: Random baseline, Right: Concept (top-k) scores"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "imagenet_val_resnet50_score_hist_best.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Scatter concept vs random per channel at best N_PROBE
try:
    d = per_n[best_n]
    cs = np.asarray(d["concept_scores"])
    rs = np.asarray(d["random_scores"])
    plt.figure(figsize=(5, 5))
    plt.scatter(rs, cs, alpha=0.7)
    lo = float(min(cs.min(), rs.min()))
    hi = float(max(cs.max(), rs.max()))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    plt.xlabel("Random baseline score")
    plt.ylabel("Concept (top-k) score")
    plt.title(
        f"ImageNet-val ResNet-50: Per-Channel Concept vs Random (N_PROBE={best_n})\nPoints above diagonal indicate meaningful concept channels"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "imagenet_val_resnet50_scatter_best.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Boxplot of per-channel concept scores across N_PROBE
try:
    data = [np.asarray(per_n[n]["concept_scores"]) for n in n_probe_values]
    plt.figure(figsize=(6, 4))
    plt.boxplot(data, labels=[str(n) for n in n_probe_values])
    plt.xlabel("N_PROBE")
    plt.ylabel("Per-channel concept score")
    plt.title(
        "ImageNet-val ResNet-50: Per-Channel Concept Score Distribution\nBoxplots across N_PROBE sweep values"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "imagenet_val_resnet50_boxplot_nprobe.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print summary metrics
try:
    for m in val_metrics:
        print(
            f"N_PROBE={m['N_PROBE']}: concept_mean={m['concept_consistency_score_mean']:.4f}, "
            f"random_mean={m['random_baseline_mean']:.4f}, delta={m['delta']:.4f}"
        )
    print(f"Best N_PROBE: {best_n}")
except Exception as e:
    print(f"Error printing metrics: {e}")
