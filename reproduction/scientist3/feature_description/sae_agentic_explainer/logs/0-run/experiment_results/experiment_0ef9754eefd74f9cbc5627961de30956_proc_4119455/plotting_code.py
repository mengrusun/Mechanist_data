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

ds_key = "gemma2_2b_gemmascope_res_16k_multi"
data = experiment_data.get(ds_key, {})
summary = data.get("summary", {})
per_feature = data.get("per_feature", {})
all_rows = data.get("all_rows", [])
layers = data.get("layers", [])
thresholds = data.get("thresholds", {})
freq_map = data.get("freq_map", {})

# Plot 1: Bar chart of gen & pred accuracy across layers (Neuronpedia/SingleShot/SAGE)
try:
    layer_keys = [k for k in summary if k.startswith("L")] + (
        ["OVERALL"] if "OVERALL" in summary else []
    )
    if layer_keys:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        methods = [("np", "Neuronpedia"), ("ss", "SingleShot"), ("sage", "SAGE")]
        x = np.arange(len(layer_keys))
        w = 0.25
        for i, (m, label) in enumerate(methods):
            vals = [summary[k].get(m + "_gen", {}).get("mean", 0) for k in layer_keys]
            errs = [summary[k].get(m + "_gen", {}).get("ci", 0) for k in layer_keys]
            axes[0].bar(x + i * w - w, vals, w, yerr=errs, label=label, capsize=3)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(layer_keys)
        axes[0].set_ylabel("Generative Accuracy")
        axes[0].set_title("Left: Generative Accuracy by Layer")
        axes[0].legend()
        axes[0].set_ylim(0, 1.05)

        for i, (m, label) in enumerate(methods):
            vals = [summary[k].get(m + "_pred", {}).get("mean", 0) for k in layer_keys]
            errs = [summary[k].get(m + "_pred", {}).get("ci", 0) for k in layer_keys]
            axes[1].bar(x + i * w - w, vals, w, yerr=errs, label=label, capsize=3)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(layer_keys)
        axes[1].set_ylabel("Predictive Accuracy (corr)")
        axes[1].set_title("Right: Predictive Accuracy by Layer")
        axes[1].legend()
        axes[1].axhline(0, color="k", lw=0.5)
        fig.suptitle("Method Comparison Across Layers — Gemma-2-2B GemmaScope-Res-16k")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemma2_2b_multi_method_comparison_by_layer.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Per-feature gen-acc delta (SAGE - Neuronpedia) vs feature frequency
try:
    if all_rows:
        fig, ax = plt.subplots(figsize=(8, 5))
        color_map = {5: "C0", 12: "C1", 20: "C2"}
        for r in all_rows:
            L = r["L"]
            ax.scatter(
                r["freq"],
                r["sage_gen"] - r["np_gen"],
                c=color_map.get(L, "gray"),
                alpha=0.75,
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Feature activation frequency (log)")
        ax.set_ylabel("Gen-acc Δ (SAGE − Neuronpedia)")
        ax.set_title(
            "Per-Feature Improvement vs Frequency\nDataset: Gemma-2-2B GemmaScope-Res-16k (multi-layer)"
        )
        for L, c in color_map.items():
            if any(r["L"] == L for r in all_rows):
                ax.scatter([], [], c=c, label=f"L{L}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemma2_2b_multi_delta_vs_freq.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: SAGE best-so-far iteration curve (aggregated), with baselines
try:
    if per_feature:
        max_iters = max(
            (len(pf.get("iterations", [])) for pf in per_feature.values()), default=0
        )
        if max_iters > 0:
            iter_vals = [[] for _ in range(max_iters)]
            for pf in per_feature.values():
                its = pf.get("iterations", [])
                best_g = -1
                for i, it in enumerate(its):
                    g = max((c["gen_acc"] for c in it.get("candidates", [])), default=0)
                    best_g = max(best_g, g)
                    iter_vals[i].append(best_g)
                for i in range(len(its), max_iters):
                    iter_vals[i].append(best_g)
            means = [np.mean(v) if v else 0 for v in iter_vals]
            stds = [np.std(v) / np.sqrt(len(v)) if v else 0 for v in iter_vals]

            fig, ax = plt.subplots(figsize=(8, 5))
            ax.errorbar(
                range(max_iters),
                means,
                yerr=stds,
                marker="o",
                label="SAGE best-so-far gen_acc",
                capsize=3,
            )
            np_mean = summary.get("OVERALL", {}).get("np_gen", {}).get("mean", None)
            ss_mean = summary.get("OVERALL", {}).get("ss_gen", {}).get("mean", None)
            if np_mean is not None:
                ax.axhline(
                    np_mean, color="gray", ls="--", label=f"Neuronpedia ({np_mean:.2f})"
                )
            if ss_mean is not None:
                ax.axhline(
                    ss_mean,
                    color="orange",
                    ls="--",
                    label=f"SingleShot ({ss_mean:.2f})",
                )
            ax.set_xlabel("SAGE iteration")
            ax.set_ylabel("Generative accuracy")
            ax.set_ylim(0, 1.05)
            ax.set_title(
                "SAGE Iterative Refinement (aggregated over features)\nDataset: Gemma-2-2B GemmaScope-Res-16k"
            )
            ax.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "gemma2_2b_multi_iteration_curve.png"),
                dpi=120,
            )
            plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Validation loss per feature
try:
    val_losses = data.get("losses", {}).get("val", [])
    feat_keys = list(per_feature.keys())
    if val_losses and feat_keys:
        n = min(len(val_losses), len(feat_keys))
        fig, ax = plt.subplots(figsize=(max(7, n * 0.35), 4))
        ax.plot(range(n), val_losses[:n], marker="s", color="C3")
        ax.set_xticks(range(n))
        ax.set_xticklabels(feat_keys[:n], rotation=90, fontsize=7)
        ax.set_ylabel("Validation Loss (1 − final SAGE gen_acc)")
        ax.set_title(
            "Per-Feature Validation Loss (Final SAGE Iteration)\nDataset: Gemma-2-2B GemmaScope-Res-16k (multi-layer)"
        )
        ax.set_ylim(-0.05, 1.05)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemma2_2b_multi_val_loss_per_feature.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Per-probe activation heatmap for a small sample of features (<=5)
try:
    feat_keys = list(per_feature.keys())
    sample = feat_keys[:: max(1, len(feat_keys) // 5)][:5] if feat_keys else []
    if sample:
        n_feats = len(sample)
        fig, axes = plt.subplots(1, n_feats, figsize=(4 * n_feats, 4.5))
        if n_feats == 1:
            axes = [axes]
        for ax, fk in zip(axes, sample):
            pf = per_feature.get(fk, {})
            iters = pf.get("iterations", [])
            rows, labels = [], []
            for i, it in enumerate(iters):
                best_c = max(
                    it.get("candidates", []),
                    key=lambda c: c.get("score", 0),
                    default=None,
                )
                if best_c is not None:
                    rows.append(best_c.get("per_probe_acts", []))
                    labels.append(f"SAGE it{i}")
            if not rows:
                continue
            maxlen = max(len(r) for r in rows)
            arr = np.array([r + [0.0] * (maxlen - len(r)) for r in rows], dtype=float)
            im = ax.imshow(arr, aspect="auto", cmap="viridis")
            thr = thresholds.get(fk, 0)
            ax.set_title(f"{fk}\nThr={thr:.3f}")
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Probe idx")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.suptitle(
            "Per-Probe Max Activations across SAGE Iterations\nDataset: Gemma-2-2B GemmaScope-Res-16k (multi-layer)"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemma2_2b_multi_per_probe_heatmap.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print summary metrics
try:
    overall = summary.get("OVERALL", {})
    print(f"SAGE gen (overall)       = {overall.get('sage_gen', {}).get('mean', 'NA')}")
    print(f"Neuronpedia gen (overall)= {overall.get('np_gen', {}).get('mean', 'NA')}")
    print(f"SingleShot gen (overall) = {overall.get('ss_gen', {}).get('mean', 'NA')}")
    print(
        f"SAGE pred (overall)      = {overall.get('sage_pred', {}).get('mean', 'NA')}"
    )
    print(
        f"generative_activation_success_rate = {data.get('generative_activation_success_rate', 'NA')}"
    )
except Exception as e:
    print(f"Error printing summary: {e}")
