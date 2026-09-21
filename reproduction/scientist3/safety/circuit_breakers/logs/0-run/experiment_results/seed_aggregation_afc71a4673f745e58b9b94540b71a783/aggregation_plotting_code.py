import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_02-20-28_circuit_breakers_attempt_1/logs/0-run/experiment_results/experiment_c9384ee6f64244f48a7b84a6d6bf3d91_proc_2293813/experiment_data.npy"
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# Collect hb dicts across runs
hb_list = [ed.get("harmbench", {}) for ed in all_experiment_data]
n_runs = len(hb_list)
print(f"Loaded {n_runs} run(s)")


def sem(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size <= 1:
        return np.zeros_like(arr.mean(axis=0)) if arr.ndim > 1 else 0.0
    return arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])


# Plot 1: Aggregated training loss curves (mean +/- SEM)
try:
    keys = ["loss", "harm_loss", "retain_ce", "rep_diff"]
    # Gather per-run arrays; align by step count (truncate to min)
    runs_data = []
    steps_ref = None
    for hb in hb_list:
        ll = hb.get("loss_log", [])
        if not ll:
            continue
        runs_data.append(ll)
    if runs_data:
        min_len = min(len(r) for r in runs_data)
        steps_ref = [r["step"] for r in runs_data[0][:min_len]]
        plt.figure(figsize=(9, 5))
        for k in keys:
            vals = np.array([[r[i][k] for i in range(min_len)] for r in runs_data])
            mean = vals.mean(axis=0)
            se = sem(vals)
            plt.plot(steps_ref, mean, label=f"{k} (mean)")
            plt.fill_between(
                steps_ref, mean - se, mean + se, alpha=0.25, label=f"{k} (±SEM)"
            )
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(
            f"HarmBench RR Aggregated Training Loss Curves (N={len(runs_data)} runs)\n"
            "Dataset: Circuit Breakers Train; shaded = ±SEM"
        )
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_agg_training_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curves: {e}")
    plt.close()

# Plot 2: Aggregated loss weight schedule
try:
    runs_data = [hb.get("loss_log", []) for hb in hb_list if hb.get("loss_log")]
    runs_data = [r for r in runs_data if r and "w_harm" in r[0]]
    if runs_data:
        min_len = min(len(r) for r in runs_data)
        steps_ref = [r["step"] for r in runs_data[0][:min_len]]
        plt.figure(figsize=(8, 4))
        for k, c in [("w_harm", "tab:blue"), ("w_retain", "tab:orange")]:
            vals = np.array([[r[i][k] for i in range(min_len)] for r in runs_data])
            mean = vals.mean(axis=0)
            se = sem(vals)
            plt.plot(steps_ref, mean, label=f"{k} (mean)", color=c)
            plt.fill_between(
                steps_ref,
                mean - se,
                mean + se,
                alpha=0.25,
                color=c,
                label=f"{k} (±SEM)",
            )
        plt.xlabel("Step")
        plt.ylabel("Weight")
        plt.title(
            f"HarmBench RR Aggregated Loss Weight Schedule (N={len(runs_data)} runs)\n"
            "Dataset: Circuit Breakers Train (cosine schedule); shaded = ±SEM"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_agg_loss_weight_schedule.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated weight schedule: {e}")
    plt.close()

# Plot 3: Aggregated Overall ASR (mean ± SEM) across runs
try:
    base_vals = [hb.get("base_asr") for hb in hb_list if hb.get("base_asr") is not None]
    rr_vals = [hb.get("rr_asr") for hb in hb_list if hb.get("rr_asr") is not None]
    if base_vals and rr_vals:
        base_mean = np.mean(base_vals)
        rr_mean = np.mean(rr_vals)
        base_se = sem(base_vals) if len(base_vals) > 1 else 0.0
        rr_se = sem(rr_vals) if len(rr_vals) > 1 else 0.0
        plt.figure(figsize=(5.5, 4.5))
        bars = plt.bar(
            ["Base Model", "RR Model"],
            [base_mean, rr_mean],
            yerr=[base_se, rr_se],
            capsize=8,
            color=["tab:red", "tab:green"],
            label="Mean ASR (±SEM)",
        )
        for b, v in zip(bars, [base_mean, rr_mean]):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
        plt.ylabel("Attack Success Rate")
        plt.ylim(0, max(1.0, max(base_mean, rr_mean) + 0.2))
        plt.title(
            f"HarmBench Aggregated Overall ASR: Base vs RR (N={len(base_vals)} runs)\n"
            "Dataset: HarmBench; error bars = SEM across runs"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_agg_asr_base_vs_rr.png"))
        print(f"Base ASR: {base_mean:.4f} ± {base_se:.4f}")
        print(f"RR   ASR: {rr_mean:.4f} ± {rr_se:.4f}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated overall ASR plot: {e}")
    plt.close()

# Plot 4: Aggregated ASR by attack style
try:
    # union of styles
    all_styles = []
    for hb in hb_list:
        for s in hb.get("base_asr_by_attack", {}).keys():
            if s not in all_styles:
                all_styles.append(s)
    if all_styles:
        base_matrix = []
        rr_matrix = []
        for hb in hb_list:
            b = hb.get("base_asr_by_attack", {})
            r = hb.get("rr_asr_by_attack", {})
            if not b or not r:
                continue
            base_matrix.append([b.get(s, np.nan) for s in all_styles])
            rr_matrix.append([r.get(s, np.nan) for s in all_styles])
        base_matrix = np.array(base_matrix, dtype=float)
        rr_matrix = np.array(rr_matrix, dtype=float)
        if base_matrix.size and rr_matrix.size:
            base_mean = np.nanmean(base_matrix, axis=0)
            rr_mean = np.nanmean(rr_matrix, axis=0)
            base_se = (
                (np.nanstd(base_matrix, axis=0, ddof=1) / np.sqrt(base_matrix.shape[0]))
                if base_matrix.shape[0] > 1
                else np.zeros_like(base_mean)
            )
            rr_se = (
                (np.nanstd(rr_matrix, axis=0, ddof=1) / np.sqrt(rr_matrix.shape[0]))
                if rr_matrix.shape[0] > 1
                else np.zeros_like(rr_mean)
            )
            x = np.arange(len(all_styles))
            w = 0.35
            plt.figure(figsize=(10, 5))
            plt.bar(
                x - w / 2,
                base_mean,
                w,
                yerr=base_se,
                capsize=4,
                label="Base (mean ±SEM)",
                color="tab:red",
            )
            plt.bar(
                x + w / 2,
                rr_mean,
                w,
                yerr=rr_se,
                capsize=4,
                label="RR (mean ±SEM)",
                color="tab:green",
            )
            for i in range(len(all_styles)):
                plt.text(
                    i - w / 2,
                    base_mean[i] + 0.02,
                    f"{base_mean[i]:.2f}",
                    ha="center",
                    fontsize=8,
                )
                plt.text(
                    i + w / 2,
                    rr_mean[i] + 0.02,
                    f"{rr_mean[i]:.2f}",
                    ha="center",
                    fontsize=8,
                )
            plt.xticks(x, all_styles, rotation=20, ha="right")
            plt.ylabel("Attack Success Rate")
            plt.ylim(0, 1.15)
            plt.title(
                f"HarmBench Aggregated ASR by Attack Style (N={base_matrix.shape[0]} runs)\n"
                "Left bars: Base, Right bars: RR (Dataset: HarmBench sub-styles); error bars = SEM"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "harmbench_agg_asr_by_attack_style.png")
            )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated per-style ASR plot: {e}")
    plt.close()

# Plot 5: Aggregated ASR delta (RR - Base) by style
try:
    all_styles = []
    for hb in hb_list:
        for s in hb.get("base_asr_by_attack", {}).keys():
            if s not in all_styles:
                all_styles.append(s)
    if all_styles:
        deltas_matrix = []
        for hb in hb_list:
            b = hb.get("base_asr_by_attack", {})
            r = hb.get("rr_asr_by_attack", {})
            if not b or not r:
                continue
            deltas_matrix.append(
                [r.get(s, np.nan) - b.get(s, np.nan) for s in all_styles]
            )
        deltas_matrix = np.array(deltas_matrix, dtype=float)
        if deltas_matrix.size:
            mean_d = np.nanmean(deltas_matrix, axis=0)
            se_d = (
                (
                    np.nanstd(deltas_matrix, axis=0, ddof=1)
                    / np.sqrt(deltas_matrix.shape[0])
                )
                if deltas_matrix.shape[0] > 1
                else np.zeros_like(mean_d)
            )
            colors = ["tab:green" if d < 0 else "tab:red" for d in mean_d]
            plt.figure(figsize=(9, 4.5))
            bars = plt.bar(
                all_styles,
                mean_d,
                yerr=se_d,
                capsize=4,
                color=colors,
                label="Mean Δ (±SEM)",
            )
            for b, v in zip(bars, mean_d):
                plt.text(
                    b.get_x() + b.get_width() / 2,
                    v + (0.01 if v >= 0 else -0.04),
                    f"{v:+.2f}",
                    ha="center",
                    fontsize=9,
                )
            plt.axhline(0, color="k", lw=0.6)
            plt.ylabel("ASR Δ (RR - Base)")
            plt.xticks(rotation=20, ha="right")
            plt.title(
                f"HarmBench Aggregated ASR Change by Attack Style (N={deltas_matrix.shape[0]} runs)\n"
                "Dataset: HarmBench sub-styles; negative = RR safer; error bars = SEM"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "harmbench_agg_asr_delta_by_style.png")
            )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated ASR delta plot: {e}")
    plt.close()

# Plot 6: Aggregated Generalization probe (pre vs post)
try:
    pre_vals = []
    post_vals = []
    for hb in hb_list:
        probe = hb.get("generalization_probe", {})
        if probe.get("pre_train_cos") is not None:
            pre_vals.append(probe["pre_train_cos"])
        if probe.get("post_train_cos") is not None:
            post_vals.append(probe["post_train_cos"])
    if pre_vals and post_vals:
        pre_mean, post_mean = np.mean(pre_vals), np.mean(post_vals)
        pre_se = sem(pre_vals) if len(pre_vals) > 1 else 0.0
        post_se = sem(post_vals) if len(post_vals) > 1 else 0.0
        plt.figure(figsize=(5.5, 4.5))
        bars = plt.bar(
            ["Pre-train", "Post-train"],
            [pre_mean, post_mean],
            yerr=[pre_se, post_se],
            capsize=8,
            color=["gray", "steelblue"],
            label="Mean cos (±SEM)",
        )
        for b, v in zip(bars, [pre_mean, post_mean]):
            plt.text(
                b.get_x() + b.get_width() / 2,
                v + (0.02 if v >= 0 else -0.05),
                f"{v:.3f}",
                ha="center",
            )
        plt.axhline(0, color="k", ls="--", lw=0.5)
        plt.ylabel("Mean cos(h_lora, h_base)")
        plt.title(
            f"Aggregated Generalization Probe (N={len(pre_vals)} runs)\n"
            "Dataset: Circuit Breakers held-out; error bars = SEM (lower = better rerouting)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "harmbench_agg_generalization_probe_cos.png")
        )
        print(f"Pre-train cos:  {pre_mean:.4f} ± {pre_se:.4f}")
        print(f"Post-train cos: {post_mean:.4f} ± {post_se:.4f}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated generalization probe plot: {e}")
    plt.close()

print("Aggregated plots saved to", working_dir)
