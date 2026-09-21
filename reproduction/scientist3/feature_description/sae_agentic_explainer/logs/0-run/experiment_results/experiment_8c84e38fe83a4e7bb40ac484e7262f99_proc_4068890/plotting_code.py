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

DATASET_NAME = "gemma2_2b_gemmascope_res_16k"

try:
    data = experiment_data["N_ITER"][DATASET_NAME]
    np_mean = data["neuronpedia"]["mean_gen_acc"]
    runs = data["runs"]
    n_iter_values = sorted([int(k) for k in runs.keys()])
    features = data["features"]
    best_n_iter = data["summary"]["best_n_iter"]
except Exception as e:
    print(f"Error extracting: {e}")
    data = None

# Plot 1: Final gen_acc vs N_ITER
try:
    plt.figure(figsize=(8, 5))
    xs = n_iter_values
    ys = [runs[str(x)]["sage_final_gen_acc"] for x in xs]
    plt.plot(xs, ys, "o-", label="SAGE final gen_acc", linewidth=2, markersize=8)
    plt.axhline(
        np_mean,
        color="gray",
        linestyle="--",
        label=f"Neuronpedia baseline ({np_mean:.3f})",
    )
    plt.xlabel("N_ITER (SAGE refinement iterations)")
    plt.ylabel("Generative Accuracy")
    plt.title(
        "Gemma-2-2B GemmaScope Res-16k (Layer 12)\nFinal Generative Accuracy vs N_ITER"
    )
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    for x, y in zip(xs, ys):
        plt.text(x, y + 0.03, f"{y:.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{DATASET_NAME}_final_gen_acc_vs_niter.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Refinement trajectories per N_ITER
try:
    plt.figure(figsize=(9, 5))
    for n_iter in n_iter_values:
        means = runs[str(n_iter)]["sage_mean_gen_acc_per_iter"]
        plt.plot(range(len(means)), means, "o-", label=f"N_ITER={n_iter}")
    plt.axhline(
        np_mean, color="gray", linestyle="--", label=f"Neuronpedia ({np_mean:.3f})"
    )
    plt.xlabel("SAGE Iteration Index")
    plt.ylabel("Mean Generative Accuracy")
    plt.title(
        "Gemma-2-2B GemmaScope Res-16k (Layer 12)\nSAGE Refinement Trajectories Across N_ITER Values"
    )
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{DATASET_NAME}_refinement_trajectories.png"),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Per-feature final gen_acc across N_ITER values (grouped bar)
try:
    plt.figure(figsize=(10, 5))
    n_feats = len(features)
    n_niter = len(n_iter_values)
    width = 0.8 / n_niter
    x_base = np.arange(n_feats)
    for i, n_iter in enumerate(n_iter_values):
        per_feat = runs[str(n_iter)]["per_feature"]
        vals = [per_feat[fid]["iterations"][-1]["gen_acc"] for fid in features]
        plt.bar(x_base + i * width, vals, width, label=f"N_ITER={n_iter}")
    # Neuronpedia baseline per feature
    np_per_feat = data["neuronpedia"]["per_feature"]
    np_vals = [np_per_feat[fid]["gen_acc"] for fid in features]
    plt.plot(
        x_base + (n_niter - 1) * width / 2,
        np_vals,
        "k*--",
        markersize=12,
        label="Neuronpedia",
        linewidth=1.5,
    )
    plt.xticks(x_base + (n_niter - 1) * width / 2, [f"Feat {f}" for f in features])
    plt.ylabel("Generative Accuracy")
    plt.title(
        "Gemma-2-2B GemmaScope Res-16k (Layer 12)\nPer-Feature Final Gen Acc Across N_ITER Values (Bars: SAGE, Stars: Neuronpedia)"
    )
    plt.ylim(0, 1.1)
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{DATASET_NAME}_per_feature_niter_bars.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Best N_ITER SAGE vs Neuronpedia per feature
try:
    plt.figure(figsize=(8, 5))
    best_per_feat = runs[str(best_n_iter)]["per_feature"]
    sage_vals = [best_per_feat[fid]["iterations"][-1]["gen_acc"] for fid in features]
    np_vals = [data["neuronpedia"]["per_feature"][fid]["gen_acc"] for fid in features]
    x = np.arange(len(features))
    width = 0.35
    plt.bar(x - width / 2, np_vals, width, label="Neuronpedia", color="gray")
    plt.bar(
        x + width / 2,
        sage_vals,
        width,
        label=f"SAGE (best N_ITER={best_n_iter})",
        color="steelblue",
    )
    plt.xticks(x, [f"Feat {f}" for f in features])
    plt.ylabel("Generative Accuracy")
    plt.title(
        f"Gemma-2-2B GemmaScope Res-16k (Layer 12)\nLeft: Neuronpedia Baseline, Right: SAGE (Best N_ITER={best_n_iter})"
    )
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    for i, (nv, sv) in enumerate(zip(np_vals, sage_vals)):
        plt.text(i - width / 2, nv + 0.02, f"{nv:.2f}", ha="center", fontsize=8)
        plt.text(i + width / 2, sv + 0.02, f"{sv:.2f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, f"{DATASET_NAME}_best_sage_vs_neuronpedia.png"),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Print summary metrics
try:
    print("=== Summary ===")
    print(f"Neuronpedia mean gen_acc: {np_mean:.4f}")
    for n_iter in n_iter_values:
        s = runs[str(n_iter)]["sage_final_gen_acc"]
        print(f"  N_ITER={n_iter}: SAGE final gen_acc = {s:.4f}")
    print(f"Best N_ITER = {best_n_iter}")
    print(
        f"Best generative_accuracy = {data['summary']['best_sage_final_gen_acc']:.4f}"
    )
    print(f"Delta over Neuronpedia: {data['summary']['delta']:+.4f}")
except Exception as e:
    print(f"Error printing summary: {e}")
