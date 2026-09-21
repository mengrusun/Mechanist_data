import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2026-07-15_02-20-28_circuit_breakers_attempt_1/logs/0-run/experiment_results/experiment_f4f9dbca15fb4aeabe21d555cc4a1216_proc_2544293/experiment_data.npy",
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

hbs = [ed.get("harmbench", {}) for ed in all_experiment_data]
n_runs = len(hbs)
print(f"Loaded {n_runs} runs")


def sem(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size <= 1:
        return 0.0
    return np.std(arr, ddof=1) / np.sqrt(arr.size)


# Plot 1: Aggregated training loss curves (mean +/- SE across runs)
try:
    all_logs = [hb.get("loss_log", []) for hb in hbs if hb.get("loss_log")]
    if all_logs:
        min_len = min(len(l) for l in all_logs)
        steps = [all_logs[0][i]["step"] for i in range(min_len)]
        keys = ["loss", "harm_loss", "retain_ce", "rep_diff"]
        plt.figure(figsize=(9, 5))
        for k in keys:
            mat = np.array([[l[i][k] for i in range(min_len)] for l in all_logs])
            mean = mat.mean(axis=0)
            se = (
                mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
                if mat.shape[0] > 1
                else np.zeros_like(mean)
            )
            plt.plot(steps, mean, label=f"{k} (mean)")
            plt.fill_between(
                steps,
                mean - se,
                mean + se,
                alpha=0.25,
                label=f"{k} ±SE" if n_runs > 1 else None,
            )
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(
            f"HarmBench RR Training Loss Curves (Aggregated over {n_runs} runs)\nDataset: Circuit Breakers Train (Llama-3-8B-Instruct)"
        )
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_agg_training_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# Plot 2: Overall ASR base vs RR with mean and SE across runs
try:
    base_vals = [hb.get("base_asr") for hb in hbs if hb.get("base_asr") is not None]
    rr_vals = [hb.get("rr_asr") for hb in hbs if hb.get("rr_asr") is not None]
    if base_vals and rr_vals:
        base_mean, rr_mean = np.mean(base_vals), np.mean(rr_vals)
        base_se, rr_se = sem(base_vals), sem(rr_vals)
        plt.figure(figsize=(6, 4))
        bars = plt.bar(
            ["Base Model", "RR Model"],
            [base_mean, rr_mean],
            yerr=[base_se, rr_se],
            capsize=8,
            color=["tab:red", "tab:green"],
            label=f"Mean ± SE (n={n_runs})",
        )
        for b, v in zip(bars, [base_mean, rr_mean]):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
        plt.ylabel("Attack Success Rate")
        plt.ylim(0, max(1.0, max(base_mean, rr_mean) + 0.2))
        plt.title(
            f"HarmBench Overall ASR: Base vs RR (Aggregated n={n_runs})\nDataset: HarmBench (lower is better, error bars = SE)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_agg_asr_base_vs_rr.png"))
        print(
            f"Base ASR: {base_mean:.3f} ± {base_se:.3f} | RR ASR: {rr_mean:.3f} ± {rr_se:.3f}"
        )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated ASR plot: {e}")
    plt.close()

# Plot 3: ASR by attack style, aggregated
try:
    all_base_by = [
        hb.get("base_asr_by_attack", {}) for hb in hbs if hb.get("base_asr_by_attack")
    ]
    all_rr_by = [
        hb.get("rr_asr_by_attack", {}) for hb in hbs if hb.get("rr_asr_by_attack")
    ]
    if all_base_by and all_rr_by:
        styles = list(all_base_by[0].keys())
        base_mat = np.array([[d.get(s, np.nan) for s in styles] for d in all_base_by])
        rr_mat = np.array([[d.get(s, np.nan) for s in styles] for d in all_rr_by])
        base_mean = np.nanmean(base_mat, axis=0)
        rr_mean = np.nanmean(rr_mat, axis=0)
        base_se = np.array(
            [sem(base_mat[:, i][~np.isnan(base_mat[:, i])]) for i in range(len(styles))]
        )
        rr_se = np.array(
            [sem(rr_mat[:, i][~np.isnan(rr_mat[:, i])]) for i in range(len(styles))]
        )
        x = np.arange(len(styles))
        w = 0.35
        plt.figure(figsize=(10, 5))
        plt.bar(
            x - w / 2,
            base_mean,
            w,
            yerr=base_se,
            capsize=4,
            label=f"Base (mean±SE, n={n_runs})",
            color="tab:red",
        )
        plt.bar(
            x + w / 2,
            rr_mean,
            w,
            yerr=rr_se,
            capsize=4,
            label=f"RR (mean±SE, n={n_runs})",
            color="tab:green",
        )
        plt.xticks(x, styles, rotation=20, ha="right")
        plt.ylabel("Attack Success Rate")
        plt.ylim(0, 1.15)
        plt.title(
            f"HarmBench ASR by Attack Style (Aggregated n={n_runs})\nLeft bars: Base, Right bars: RR (Dataset: HarmBench sub-styles)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_agg_asr_by_attack_style.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated ASR-by-style plot: {e}")
    plt.close()

# Plot 4: Generalization probe (pre vs post) aggregated
try:
    pre_vals = [hb.get("generalization_probe", {}).get("pre_train_cos") for hb in hbs]
    post_vals = [hb.get("generalization_probe", {}).get("post_train_cos") for hb in hbs]
    pre_vals = [v for v in pre_vals if v is not None]
    post_vals = [v for v in post_vals if v is not None]
    if pre_vals and post_vals:
        pre_mean, post_mean = np.mean(pre_vals), np.mean(post_vals)
        pre_se, post_se = sem(pre_vals), sem(post_vals)
        plt.figure(figsize=(5, 4))
        bars = plt.bar(
            ["Pre-train", "Post-train"],
            [pre_mean, post_mean],
            yerr=[pre_se, post_se],
            capsize=8,
            color=["gray", "steelblue"],
            label=f"Mean ± SE (n={n_runs})",
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
            f"Generalization Probe: Cosine Sim (Aggregated n={n_runs})\nDataset: Circuit Breakers held-out (lower = better rerouting)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_agg_generalization_probe.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated probe plot: {e}")
    plt.close()

# Plot 5: ASR delta (RR - Base) by attack style, aggregated
try:
    all_base_by = [
        hb.get("base_asr_by_attack", {}) for hb in hbs if hb.get("base_asr_by_attack")
    ]
    all_rr_by = [
        hb.get("rr_asr_by_attack", {}) for hb in hbs if hb.get("rr_asr_by_attack")
    ]
    if all_base_by and all_rr_by:
        styles = list(all_base_by[0].keys())
        delta_mat = np.array(
            [
                [
                    all_rr_by[r].get(s, np.nan) - all_base_by[r].get(s, np.nan)
                    for s in styles
                ]
                for r in range(len(all_base_by))
            ]
        )
        d_mean = np.nanmean(delta_mat, axis=0)
        d_se = np.array(
            [
                sem(delta_mat[:, i][~np.isnan(delta_mat[:, i])])
                for i in range(len(styles))
            ]
        )
        colors = ["tab:green" if d < 0 else "tab:red" for d in d_mean]
        plt.figure(figsize=(9, 4))
        bars = plt.bar(
            styles,
            d_mean,
            yerr=d_se,
            capsize=5,
            color=colors,
            label=f"Mean ± SE (n={n_runs})",
        )
        for b, v in zip(bars, d_mean):
            plt.text(
                b.get_x() + b.get_width() / 2,
                v + (0.01 if v >= 0 else -0.03),
                f"{v:+.2f}",
                ha="center",
                fontsize=9,
            )
        plt.axhline(0, color="k", lw=0.6)
        plt.ylabel("ASR Delta (RR - Base)")
        plt.xticks(rotation=20, ha="right")
        plt.title(
            f"HarmBench ASR Change by Attack Style (Aggregated n={n_runs})\nDataset: HarmBench sub-styles (negative = RR safer)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_agg_asr_delta_by_style.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated delta plot: {e}")
    plt.close()

print("Aggregated plots saved to", working_dir)
