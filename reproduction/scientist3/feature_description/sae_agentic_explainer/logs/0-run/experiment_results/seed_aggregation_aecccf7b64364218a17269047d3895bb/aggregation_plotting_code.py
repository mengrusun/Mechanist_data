import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-14_09-16-36_sae_agentic_explainer_attempt_3/logs/0-run/experiment_results/experiment_7211b09290e84e3db319eeba770b45be_proc_4119455/experiment_data.npy"
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

ds_key = "gemma2_2b_gemmascope_res_16k_multi"
n_runs = len(all_experiment_data)
print(f"Loaded {n_runs} runs")

# Gather per-run summaries
runs_data = [ed.get(ds_key, {}) for ed in all_experiment_data]

# --- Plot 1: Aggregated bar chart across runs ---
try:
    # Collect layer keys common across runs
    layer_key_sets = []
    for d in runs_data:
        s = d.get("summary", {})
        lks = [k for k in s if k.startswith("L")]
        if "OVERALL" in s:
            lks.append("OVERALL")
        layer_key_sets.append(lks)
    if layer_key_sets:
        layer_keys = sorted(
            (
                set.intersection(*[set(l) for l in layer_key_sets])
                if layer_key_sets
                else set()
            ),
            key=lambda x: (x == "OVERALL", x),
        )
        if not layer_keys and layer_key_sets:
            layer_keys = layer_key_sets[0]

        methods = [("np", "Neuronpedia"), ("ss", "SingleShot"), ("sage", "SAGE")]
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        x = np.arange(len(layer_keys))
        w = 0.25

        for kind_idx, kind in enumerate(["gen", "pred"]):
            ax = axes[kind_idx]
            for i, (m, label) in enumerate(methods):
                means_across_runs = []
                errs_across_runs = []
                for lk in layer_keys:
                    vals = []
                    for d in runs_data:
                        v = (
                            d.get("summary", {})
                            .get(lk, {})
                            .get(f"{m}_{kind}", {})
                            .get("mean", None)
                        )
                        if v is not None:
                            vals.append(v)
                    if vals:
                        means_across_runs.append(np.mean(vals))
                        errs_across_runs.append(
                            np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
                        )
                    else:
                        means_across_runs.append(0)
                        errs_across_runs.append(0)
                ax.bar(
                    x + i * w - w,
                    means_across_runs,
                    w,
                    yerr=errs_across_runs,
                    label=label,
                    capsize=3,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(layer_keys)
            ax.legend()
            if kind == "gen":
                ax.set_ylabel("Generative Accuracy")
                ax.set_title("Left: Generative Accuracy by Layer (mean ± SE)")
                ax.set_ylim(0, 1.05)
            else:
                ax.set_ylabel("Predictive Accuracy (corr)")
                ax.set_title("Right: Predictive Accuracy by Layer (mean ± SE)")
                ax.axhline(0, color="k", lw=0.5)
        fig.suptitle(
            f"Aggregated Method Comparison Across Layers (n={n_runs} runs)\nDataset: Gemma-2-2B GemmaScope-Res-16k"
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

# --- Plot 2: Aggregated delta vs frequency (scatter with all runs combined) ---
try:
    all_rows_combined = []
    for d in runs_data:
        for r in d.get("all_rows", []):
            all_rows_combined.append(r)
    if all_rows_combined:
        fig, ax = plt.subplots(figsize=(8, 5))
        color_map = {5: "C0", 12: "C1", 20: "C2"}
        for r in all_rows_combined:
            L = r["L"]
            ax.scatter(
                r["freq"],
                r["sage_gen"] - r["np_gen"],
                c=color_map.get(L, "gray"),
                alpha=0.6,
                s=25,
            )
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("Feature activation frequency (log)")
        ax.set_ylabel("Gen-acc Δ (SAGE − Neuronpedia)")
        ax.set_title(
            f"Per-Feature Improvement vs Frequency (aggregated, n={n_runs} runs)\nDataset: Gemma-2-2B GemmaScope-Res-16k"
        )
        for L, c in color_map.items():
            if any(r["L"] == L for r in all_rows_combined):
                ax.scatter([], [], c=c, label=f"L{L}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemma2_2b_multi_agg_delta_vs_freq.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# --- Plot 3: Aggregated iteration curve across runs ---
try:
    # For each run, compute mean best-so-far curve; then aggregate across runs
    run_curves = []
    for d in runs_data:
        per_feature = d.get("per_feature", {})
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
        means = np.array([np.mean(v) if v else 0 for v in iter_vals])
        run_curves.append(means)

    if run_curves:
        max_len = max(len(c) for c in run_curves)
        padded = []
        for c in run_curves:
            if len(c) < max_len:
                c = np.concatenate([c, np.full(max_len - len(c), c[-1])])
            padded.append(c)
        arr = np.stack(padded, axis=0)
        mean_curve = arr.mean(axis=0)
        se_curve = (
            arr.std(axis=0) / np.sqrt(arr.shape[0])
            if arr.shape[0] > 1
            else np.zeros(max_len)
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.errorbar(
            range(max_len),
            mean_curve,
            yerr=se_curve,
            marker="o",
            label="SAGE best-so-far (mean ± SE across runs)",
            capsize=3,
        )

        # Aggregated baselines
        np_vals = [
            d.get("summary", {}).get("OVERALL", {}).get("np_gen", {}).get("mean")
            for d in runs_data
        ]
        np_vals = [v for v in np_vals if v is not None]
        ss_vals = [
            d.get("summary", {}).get("OVERALL", {}).get("ss_gen", {}).get("mean")
            for d in runs_data
        ]
        ss_vals = [v for v in ss_vals if v is not None]
        if np_vals:
            npm = np.mean(np_vals)
            ax.axhline(
                npm, color="gray", ls="--", label=f"Neuronpedia mean ({npm:.2f})"
            )
        if ss_vals:
            ssm = np.mean(ss_vals)
            ax.axhline(
                ssm, color="orange", ls="--", label=f"SingleShot mean ({ssm:.2f})"
            )

        ax.set_xlabel("SAGE iteration")
        ax.set_ylabel("Generative accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            f"Aggregated SAGE Iterative Refinement (n={n_runs} runs)\nDataset: Gemma-2-2B GemmaScope-Res-16k"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemma2_2b_multi_agg_iteration_curve.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# --- Plot 4: Aggregated validation loss per feature (mean ± SE across runs) ---
try:
    # Align by feature key
    feat_to_vals = {}
    for d in runs_data:
        val_losses = d.get("losses", {}).get("val", [])
        feat_keys = list(d.get("per_feature", {}).keys())
        n = min(len(val_losses), len(feat_keys))
        for i in range(n):
            feat_to_vals.setdefault(feat_keys[i], []).append(val_losses[i])
    if feat_to_vals:
        feat_keys_sorted = list(feat_to_vals.keys())
        means = [np.mean(feat_to_vals[k]) for k in feat_keys_sorted]
        ses = [
            (
                np.std(feat_to_vals[k]) / np.sqrt(len(feat_to_vals[k]))
                if len(feat_to_vals[k]) > 1
                else 0
            )
            for k in feat_keys_sorted
        ]
        fig, ax = plt.subplots(figsize=(max(7, len(feat_keys_sorted) * 0.35), 4.5))
        ax.errorbar(
            range(len(feat_keys_sorted)),
            means,
            yerr=ses,
            marker="s",
            color="C3",
            linestyle="-",
            label="Mean ± SE across runs",
            capsize=3,
        )
        ax.set_xticks(range(len(feat_keys_sorted)))
        ax.set_xticklabels(feat_keys_sorted, rotation=90, fontsize=7)
        ax.set_ylabel("Validation Loss (1 − final SAGE gen_acc)")
        ax.set_title(
            f"Aggregated Per-Feature Validation Loss (n={n_runs} runs)\nDataset: Gemma-2-2B GemmaScope-Res-16k"
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
    print(f"Error creating plot4: {e}")
    plt.close()

# --- Plot 5: Aggregated overall metric bar chart (means ± SE across runs) ---
try:
    metric_names = [
        ("np_gen", "NP gen"),
        ("ss_gen", "SS gen"),
        ("sage_gen", "SAGE gen"),
        ("np_pred", "NP pred"),
        ("ss_pred", "SS pred"),
        ("sage_pred", "SAGE pred"),
    ]
    mean_vals, se_vals, labels = [], [], []
    for mk, lbl in metric_names:
        vals = []
        for d in runs_data:
            v = d.get("summary", {}).get("OVERALL", {}).get(mk, {}).get("mean")
            if v is not None:
                vals.append(v)
        if vals:
            mean_vals.append(np.mean(vals))
            se_vals.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
            labels.append(lbl)
    if mean_vals:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(labels))
        ax.bar(
            x,
            mean_vals,
            yerr=se_vals,
            capsize=4,
            color=["C0", "C1", "C2", "C0", "C1", "C2"][: len(labels)],
            label="Mean ± SE across runs",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Metric value")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(
            f"Aggregated Overall Metrics (n={n_runs} runs)\nDataset: Gemma-2-2B GemmaScope-Res-16k"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gemma2_2b_multi_agg_overall_metrics.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# --- Print aggregated summary metrics ---
try:
    print("\n=== Aggregated OVERALL metrics (mean ± SE across runs) ===")
    for mk, lbl in [
        ("sage_gen", "SAGE gen"),
        ("np_gen", "Neuronpedia gen"),
        ("ss_gen", "SingleShot gen"),
        ("sage_pred", "SAGE pred"),
        ("np_pred", "NP pred"),
        ("ss_pred", "SS pred"),
    ]:
        vals = []
        for d in runs_data:
            v = d.get("summary", {}).get("OVERALL", {}).get(mk, {}).get("mean")
            if v is not None:
                vals.append(v)
        if vals:
            m = np.mean(vals)
            se = np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0
            print(f"{lbl:20s}: {m:.4f} ± {se:.4f}  (n={len(vals)})")
        else:
            print(f"{lbl:20s}: NA")

    gen_rates = [d.get("generative_activation_success_rate") for d in runs_data]
    gen_rates = [v for v in gen_rates if v is not None]
    if gen_rates:
        print(
            f"\ngenerative_activation_success_rate: {np.mean(gen_rates):.4f} "
            f"± {np.std(gen_rates)/np.sqrt(len(gen_rates)) if len(gen_rates)>1 else 0:.4f}"
        )
except Exception as e:
    print(f"Error printing summary: {e}")
