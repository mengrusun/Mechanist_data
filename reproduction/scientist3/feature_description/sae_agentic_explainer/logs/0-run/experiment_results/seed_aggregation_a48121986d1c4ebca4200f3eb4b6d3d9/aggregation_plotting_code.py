import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-14_09-16-36_sae_agentic_explainer_attempt_3/logs/0-run/experiment_results/experiment_e57b15f0cf4c460d8db5fc8d4676e145_proc_621193/experiment_data.npy"
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        experiment_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

ds_key = "gemma2_2b_gemmascope_res_16k_multi"

# Gather per-run data
runs_data = [ed.get(ds_key, {}) for ed in all_experiment_data if ds_key in ed]
n_runs = len(runs_data)
print(f"Number of runs aggregated: {n_runs}")


def sem(vals):
    vals = np.array(vals, dtype=float)
    if len(vals) <= 1:
        return 0.0
    return np.std(vals, ddof=1) / np.sqrt(len(vals))


# Plot 1: Aggregated Bar chart of gen & pred accuracy across layers with SE across runs
try:
    # Collect layer keys union
    layer_keys_set = set()
    for rd in runs_data:
        s = rd.get("summary", {})
        for k in s:
            if k.startswith("L") or k == "OVERALL":
                layer_keys_set.add(k)
    layer_keys = sorted([k for k in layer_keys_set if k != "OVERALL"]) + (
        ["OVERALL"] if "OVERALL" in layer_keys_set else []
    )

    if layer_keys and n_runs > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        methods = [("np", "Neuronpedia"), ("ss", "SingleShot"), ("sage", "SAGE")]
        x = np.arange(len(layer_keys))
        w = 0.25

        for i, (m, label) in enumerate(methods):
            means, errs = [], []
            for lk in layer_keys:
                vals = []
                for rd in runs_data:
                    v = (
                        rd.get("summary", {})
                        .get(lk, {})
                        .get(m + "_gen", {})
                        .get("mean", None)
                    )
                    if v is not None:
                        vals.append(v)
                means.append(np.mean(vals) if vals else 0)
                errs.append(sem(vals) if vals else 0)
            axes[0].bar(
                x + i * w - w,
                means,
                w,
                yerr=errs,
                label=f"{label} (mean±SE)",
                capsize=3,
            )
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(layer_keys)
        axes[0].set_ylabel("Generative Accuracy")
        axes[0].set_title(f"Left: Generative Accuracy by Layer (agg. {n_runs} runs)")
        axes[0].legend()
        axes[0].set_ylim(0, 1.05)

        for i, (m, label) in enumerate(methods):
            means, errs = [], []
            for lk in layer_keys:
                vals = []
                for rd in runs_data:
                    v = (
                        rd.get("summary", {})
                        .get(lk, {})
                        .get(m + "_pred", {})
                        .get("mean", None)
                    )
                    if v is not None:
                        vals.append(v)
                means.append(np.mean(vals) if vals else 0)
                errs.append(sem(vals) if vals else 0)
            axes[1].bar(
                x + i * w - w,
                means,
                w,
                yerr=errs,
                label=f"{label} (mean±SE)",
                capsize=3,
            )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(layer_keys)
        axes[1].set_ylabel("Predictive Accuracy (corr)")
        axes[1].set_title(f"Right: Predictive Accuracy by Layer (agg. {n_runs} runs)")
        axes[1].legend()
        axes[1].axhline(0, color="k", lw=0.5)
        fig.suptitle(
            "Aggregated Method Comparison Across Layers — Gemma-2-2B GemmaScope-Res-16k"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "gemma2_2b_multi_agg_method_comparison_by_layer.png"
            ),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Aggregated SAGE best-so-far iteration curve across runs
try:
    # For each run, compute per-iteration mean over features, then aggregate across runs
    per_run_iter_means = []
    for rd in runs_data:
        per_feature = rd.get("per_feature", {})
        if not per_feature:
            continue
        max_iters = max(
            (len(pf.get("iterations", [])) for pf in per_feature.values()), default=0
        )
        if max_iters == 0:
            continue
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
        per_run_iter_means.append(means)

    if per_run_iter_means:
        max_len = max(len(m) for m in per_run_iter_means)
        # pad each run's mean list to max_len using last value
        padded = []
        for m in per_run_iter_means:
            last = m[-1] if m else 0
            padded.append(list(m) + [last] * (max_len - len(m)))
        arr = np.array(padded)  # shape (n_runs, max_len)
        agg_mean = arr.mean(axis=0)
        agg_se = (
            arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])
            if arr.shape[0] > 1
            else np.zeros(max_len)
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            range(max_len),
            agg_mean,
            yerr=agg_se,
            marker="o",
            label=f"SAGE best-so-far (mean±SE, {arr.shape[0]} runs)",
            capsize=3,
        )

        # aggregate baseline means across runs
        np_vals = [
            rd.get("summary", {}).get("OVERALL", {}).get("np_gen", {}).get("mean", None)
            for rd in runs_data
        ]
        np_vals = [v for v in np_vals if v is not None]
        ss_vals = [
            rd.get("summary", {}).get("OVERALL", {}).get("ss_gen", {}).get("mean", None)
            for rd in runs_data
        ]
        ss_vals = [v for v in ss_vals if v is not None]

        if np_vals:
            m = np.mean(np_vals)
            s = sem(np_vals)
            ax.axhline(m, color="gray", ls="--", label=f"Neuronpedia ({m:.2f}±{s:.2f})")
            ax.fill_between(range(max_len), m - s, m + s, color="gray", alpha=0.15)
        if ss_vals:
            m = np.mean(ss_vals)
            s = sem(ss_vals)
            ax.axhline(
                m, color="orange", ls="--", label=f"SingleShot ({m:.2f}±{s:.2f})"
            )
            ax.fill_between(range(max_len), m - s, m + s, color="orange", alpha=0.15)

        ax.set_xlabel("SAGE iteration")
        ax.set_ylabel("Generative accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            "Aggregated SAGE Iterative Refinement (mean±SE across runs)\nDataset: Gemma-2-2B GemmaScope-Res-16k"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemma2_2b_multi_agg_iteration_curve.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Aggregated per-feature validation loss across runs (mean ± SE)
try:
    # Collect union of feature keys and per-run val losses
    feat_to_vals = {}
    feat_order = []
    for rd in runs_data:
        val_losses = rd.get("losses", {}).get("val", [])
        feat_keys = list(rd.get("per_feature", {}).keys())
        n = min(len(val_losses), len(feat_keys))
        for i in range(n):
            fk = feat_keys[i]
            if fk not in feat_to_vals:
                feat_to_vals[fk] = []
                feat_order.append(fk)
            feat_to_vals[fk].append(val_losses[i])

    if feat_order:
        means = [np.mean(feat_to_vals[fk]) for fk in feat_order]
        errs = [sem(feat_to_vals[fk]) for fk in feat_order]
        n = len(feat_order)
        fig, ax = plt.subplots(figsize=(max(7, n * 0.35), 4))
        ax.errorbar(
            range(n),
            means,
            yerr=errs,
            marker="s",
            color="C3",
            capsize=3,
            label=f"Val loss (mean±SE, up to {n_runs} runs)",
        )
        ax.set_xticks(range(n))
        ax.set_xticklabels(feat_order, rotation=90, fontsize=7)
        ax.set_ylabel("Validation Loss (1 − final SAGE gen_acc)")
        ax.set_title(
            "Aggregated Per-Feature Validation Loss (mean±SE across runs)\nDataset: Gemma-2-2B GemmaScope-Res-16k"
        )
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemma2_2b_multi_agg_val_loss_per_feature.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Aggregated per-feature gen-acc delta (SAGE - Neuronpedia) vs frequency (mean over runs per feature)
try:
    # Merge rows by feature identity (using L and feature freq/key)
    from collections import defaultdict

    key_to_deltas = defaultdict(list)
    key_to_freq = {}
    key_to_L = {}
    for rd in runs_data:
        for r in rd.get("all_rows", []):
            # use (L, feature_id or freq)  - use tuple of L and index if id available
            fid = r.get("feature_id", r.get("feat", r.get("freq")))
            key = (r["L"], fid)
            key_to_deltas[key].append(r["sage_gen"] - r["np_gen"])
            key_to_freq[key] = r["freq"]
            key_to_L[key] = r["L"]

    if key_to_deltas:
        fig, ax = plt.subplots(figsize=(8, 5))
        color_map = {5: "C0", 12: "C1", 20: "C2"}
        Ls_present = set()
        for key, deltas in key_to_deltas.items():
            L = key_to_L[key]
            Ls_present.add(L)
            mean_d = np.mean(deltas)
            se_d = sem(deltas)
            ax.errorbar(
                key_to_freq[key],
                mean_d,
                yerr=se_d,
                fmt="o",
                color=color_map.get(L, "gray"),
                alpha=0.75,
                capsize=2,
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Feature activation frequency (log)")
        ax.set_ylabel("Gen-acc Δ (SAGE − Neuronpedia), mean±SE")
        ax.set_title(
            f"Aggregated Per-Feature Improvement vs Frequency ({n_runs} runs)\nDataset: Gemma-2-2B GemmaScope-Res-16k"
        )
        for L in sorted(Ls_present):
            ax.scatter([], [], c=color_map.get(L, "gray"), label=f"L{L}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemma2_2b_multi_agg_delta_vs_freq.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Print aggregated summary metrics
try:
    print("=== Aggregated OVERALL metrics (mean ± SE across runs) ===")
    for m, label in [
        ("sage_gen", "SAGE gen"),
        ("np_gen", "Neuronpedia gen"),
        ("ss_gen", "SingleShot gen"),
        ("sage_pred", "SAGE pred"),
    ]:
        vals = [
            rd.get("summary", {}).get("OVERALL", {}).get(m, {}).get("mean", None)
            for rd in runs_data
        ]
        vals = [v for v in vals if v is not None]
        if vals:
            print(
                f"{label:25s} = {np.mean(vals):.4f} ± {sem(vals):.4f}  (n={len(vals)})"
            )
        else:
            print(f"{label:25s} = NA")

    rates = [rd.get("generative_activation_success_rate", None) for rd in runs_data]
    rates = [v for v in rates if v is not None]
    if rates:
        print(
            f"generative_activation_success_rate = {np.mean(rates):.4f} ± {sem(rates):.4f}"
        )
except Exception as e:
    print(f"Error printing summary: {e}")
