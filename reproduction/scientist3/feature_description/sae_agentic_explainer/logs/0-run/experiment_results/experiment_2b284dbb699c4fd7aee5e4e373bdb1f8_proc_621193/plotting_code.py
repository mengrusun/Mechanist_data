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

ABL = "no_negative_contrastive_probes"
DKEY = "gemma2_2b_gemmascope_res_16k_multi"
try:
    data = experiment_data[ABL][DKEY]
    summary = data.get("summary", {})
    per_feature = data.get("per_feature", {})
    all_rows = data.get("all_rows", [])
    val_losses = data.get("losses", {}).get("val", [])
except Exception as e:
    print(f"Error accessing data: {e}")
    data, summary, per_feature, all_rows, val_losses = {}, {}, {}, [], []

# Plot 1: Generative & Predictive accuracy by layer/method
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    methods = ["Neuronpedia", "SingleShot", "SAGE(no-neg)"]
    mkeys = ["np", "ss", "sage"]
    layer_keys = [k for k in summary if k.startswith("L")] + (
        ["OVERALL"] if "OVERALL" in summary else []
    )
    x = np.arange(len(layer_keys))
    w = 0.25
    for i, m in enumerate(mkeys):
        vals = [summary[k][m + "_gen"]["mean"] for k in layer_keys]
        errs = [summary[k][m + "_gen"]["ci"] for k in layer_keys]
        axes[0].bar(x + i * w - w, vals, w, yerr=errs, label=methods[i], capsize=3)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(layer_keys)
    axes[0].set_ylabel("Generative Accuracy")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title("Left: Generative Accuracy by Layer")
    axes[0].legend()
    for i, m in enumerate(mkeys):
        vals = [summary[k][m + "_pred"]["mean"] for k in layer_keys]
        errs = [summary[k][m + "_pred"]["ci"] for k in layer_keys]
        axes[1].bar(x + i * w - w, vals, w, yerr=errs, label=methods[i], capsize=3)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layer_keys)
    axes[1].set_ylabel("Predictive Accuracy (pos-only)")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Right: Predictive Accuracy (pos-only proxy)")
    axes[1].legend()
    fig.suptitle(
        "Gemma-2-2b GemmaScope SAE (ablation: no negative probes)\nMethod comparison per layer"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "gemmascope_noneg_accuracy_by_layer.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: SAGE iterative refinement curve
try:
    max_iters = max(
        (len(pf.get("iterations", [])) for pf in per_feature.values()), default=0
    )
    iter_gens = [[] for _ in range(max_iters)]
    for pf in per_feature.values():
        best_g = -1
        for i, it in enumerate(pf.get("iterations", [])):
            g = max(c["gen_acc"] for c in it["candidates"])
            best_g = max(best_g, g)
            iter_gens[i].append(best_g)
        for i in range(len(pf.get("iterations", [])), max_iters):
            iter_gens[i].append(best_g)
    means = [np.mean(v) if v else 0 for v in iter_gens]
    stds = [np.std(v) if v else 0 for v in iter_gens]
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        range(max_iters),
        means,
        yerr=stds,
        marker="o",
        label="SAGE(no-neg) best-so-far",
        capsize=3,
    )
    if "OVERALL" in summary:
        plt.axhline(
            summary["OVERALL"]["np_gen"]["mean"],
            color="gray",
            linestyle="--",
            label=f"Neuronpedia ({summary['OVERALL']['np_gen']['mean']:.2f})",
        )
        plt.axhline(
            summary["OVERALL"]["ss_gen"]["mean"],
            color="orange",
            linestyle="--",
            label=f"SingleShot ({summary['OVERALL']['ss_gen']['mean']:.2f})",
        )
    plt.xlabel("SAGE Iteration")
    plt.ylabel("Generative Accuracy")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.title("Gemma-2-2b GemmaScope SAE (ablation)\nSAGE Iterative Refinement Curve")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "gemmascope_noneg_iteration_curve.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Validation loss curve across features
try:
    plt.figure(figsize=(10, 4))
    plt.plot(
        range(len(val_losses)),
        val_losses,
        marker=".",
        label="val loss (1 - SAGE gen_acc)",
    )
    plt.xlabel("Feature index (processing order)")
    plt.ylabel("Validation loss")
    plt.title(
        "Gemma-2-2b GemmaScope SAE (ablation: no negative probes)\nPer-feature Validation Loss"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "gemmascope_noneg_val_loss_curve.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Per-feature gen_acc comparison (scatter/paired)
try:
    if all_rows:
        np_gens = [r["np_gen"] for r in all_rows]
        ss_gens = [r["ss_gen"] for r in all_rows]
        sage_gens = [r["sage_gen"] for r in all_rows]
        idx = np.arange(len(all_rows))
        labels = [f"L{r['L']}_f{r['fid']}" for r in all_rows]
        plt.figure(figsize=(12, 5))
        plt.plot(idx, np_gens, "o-", label="Neuronpedia", alpha=0.7)
        plt.plot(idx, ss_gens, "s-", label="SingleShot", alpha=0.7)
        plt.plot(idx, sage_gens, "^-", label="SAGE(no-neg)", alpha=0.7)
        plt.xticks(idx, labels, rotation=90, fontsize=7)
        plt.ylabel("Generative Accuracy")
        plt.ylim(0, 1.05)
        plt.title(
            "Gemma-2-2b GemmaScope SAE (ablation)\nPer-feature Generative Accuracy: Methods Comparison"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemmascope_noneg_per_feature_gen_acc.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Feature frequency vs SAGE gen_acc scatter
try:
    if all_rows:
        freqs = [r["freq"] for r in all_rows]
        sage_gens = [r["sage_gen"] for r in all_rows]
        layers = [r["L"] for r in all_rows]
        plt.figure(figsize=(7, 5))
        for L in sorted(set(layers)):
            xs = [f for f, l in zip(freqs, layers) if l == L]
            ys = [g for g, l in zip(sage_gens, layers) if l == L]
            plt.scatter(xs, ys, label=f"Layer {L}", alpha=0.7)
        plt.xscale("log")
        plt.xlabel("Feature Activation Frequency (ref corpus)")
        plt.ylabel("SAGE(no-neg) Generative Accuracy")
        plt.ylim(0, 1.05)
        plt.title(
            "Gemma-2-2b GemmaScope SAE (ablation)\nFrequency vs SAGE Generative Accuracy"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemmascope_noneg_freq_vs_gen_acc.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print summary metric
try:
    if "OVERALL" in summary:
        print(
            f"Overall SAGE(no-neg) gen_acc: {summary['OVERALL']['sage_gen']['mean']:.4f}"
        )
        print(
            f"Overall Neuronpedia gen_acc: {summary['OVERALL']['np_gen']['mean']:.4f}"
        )
        print(f"Overall SingleShot gen_acc: {summary['OVERALL']['ss_gen']['mean']:.4f}")
except Exception as e:
    print(f"Error printing summary: {e}")
