import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_02-20-28_circuit_breakers_attempt_1/logs/0-run/experiment_results/experiment_78a5d30ae237402f80f53438fe3f97b6_proc_1950911/experiment_data.npy",
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

n_runs = len(all_experiment_data)
print(f"Loaded {n_runs} experiment runs")

# Gather harmbench data across runs
hb_runs = [ed.get("harmbench", {}) for ed in all_experiment_data]


def sem(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.shape[0] <= 1:
        return np.zeros(arr.shape[1:]) if arr.ndim > 1 else 0.0
    return arr.std(axis=0, ddof=1) / np.sqrt(arr.shape[0])


# Plot 1: Aggregated Training Loss Curves
try:
    all_steps, all_losses = [], {
        "loss": [],
        "harm_loss": [],
        "retain_ce": [],
        "rep_diff": [],
    }
    ref_steps = None
    for hb in hb_runs:
        loss_log = hb.get("loss_log", [])
        if not loss_log:
            continue
        steps = [l["step"] for l in loss_log]
        if ref_steps is None:
            ref_steps = steps
        # Only include if length matches ref
        if len(steps) != len(ref_steps):
            continue
        for k in all_losses.keys():
            all_losses[k].append([l[k] for l in loss_log])

    if ref_steps is not None and len(all_losses["loss"]) > 0:
        plt.figure(figsize=(9, 6))
        colors = {
            "loss": "tab:blue",
            "harm_loss": "tab:orange",
            "retain_ce": "tab:green",
            "rep_diff": "tab:red",
        }
        for k, vals in all_losses.items():
            arr = np.array(vals)
            mean = arr.mean(axis=0)
            se = sem(arr)
            plt.plot(ref_steps, mean, label=f"{k} (mean)", color=colors[k])
            plt.fill_between(
                ref_steps,
                mean - se,
                mean + se,
                alpha=0.25,
                color=colors[k],
                label=f"{k} ±SEM",
            )
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(
            f"HarmBench RR Aggregated Training Loss (n={len(all_losses['loss'])} runs)\n"
            "Dataset: Circuit Breakers Train (Llama-3-8B-Instruct) — Mean ± SEM"
        )
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_agg_training_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curve plot: {e}")
    plt.close()

# Plot 2: Aggregated ASR bar chart with SEM
try:
    base_vals = [hb.get("base_asr") for hb in hb_runs if hb.get("base_asr") is not None]
    rr_vals = [hb.get("rr_asr") for hb in hb_runs if hb.get("rr_asr") is not None]
    if base_vals and rr_vals:
        base_mean, rr_mean = np.mean(base_vals), np.mean(rr_vals)
        base_se = (
            np.std(base_vals, ddof=1) / np.sqrt(len(base_vals))
            if len(base_vals) > 1
            else 0.0
        )
        rr_se = (
            np.std(rr_vals, ddof=1) / np.sqrt(len(rr_vals)) if len(rr_vals) > 1 else 0.0
        )
        plt.figure(figsize=(6, 5))
        bars = plt.bar(
            ["Base Model", "RR Model"],
            [base_mean, rr_mean],
            yerr=[base_se, rr_se],
            capsize=8,
            color=["tab:red", "tab:green"],
            label="Mean ± SEM",
        )
        for b, v, s in zip(bars, [base_mean, rr_mean], [base_se, rr_se]):
            plt.text(
                b.get_x() + b.get_width() / 2,
                v + s + 0.02,
                f"{v:.3f}±{s:.3f}",
                ha="center",
                fontsize=9,
            )
        plt.ylabel("Attack Success Rate")
        plt.ylim(0, max(1.0, max(base_mean + base_se, rr_mean + rr_se) + 0.15))
        plt.title(
            f"HarmBench ASR: Base vs RR (n={len(base_vals)} runs)\n"
            "Dataset: HarmBench (lower is better) — Mean ± SEM"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_agg_asr_base_vs_rr.png"))
        print(f"Aggregated base_asr: {base_mean:.4f} ± {base_se:.4f}")
        print(f"Aggregated rr_asr:   {rr_mean:.4f} ± {rr_se:.4f}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated ASR bar plot: {e}")
    plt.close()

# Plot 3: Aggregated per-sample ASR
try:
    base_mat, rr_mat = [], []
    min_n = None
    for hb in hb_runs:
        sb = hb.get("samples_base", [])
        sr = hb.get("samples_rr", [])
        if not sb or not sr:
            continue
        n = min(len(sb), len(sr))
        min_n = n if min_n is None else min(min_n, n)
        base_mat.append([s["asr"] for s in sb])
        rr_mat.append([s["asr"] for s in sr])

    if base_mat and min_n:
        base_mat = np.array([row[:min_n] for row in base_mat], dtype=float)
        rr_mat = np.array([row[:min_n] for row in rr_mat], dtype=float)
        base_mean = base_mat.mean(axis=0)
        rr_mean = rr_mat.mean(axis=0)
        base_se = sem(base_mat)
        rr_se = sem(rr_mat)

        idx = np.arange(min_n)
        width = 0.4
        plt.figure(figsize=(max(9, min_n * 0.4), 5))
        plt.bar(
            idx - width / 2,
            base_mean,
            width,
            yerr=base_se,
            capsize=3,
            label=f"Base (mean±SEM, n={base_mat.shape[0]})",
            color="tab:red",
        )
        plt.bar(
            idx + width / 2,
            rr_mean,
            width,
            yerr=rr_se,
            capsize=3,
            label=f"RR (mean±SEM, n={rr_mat.shape[0]})",
            color="tab:green",
        )
        plt.xlabel("Sample Index")
        plt.ylabel("ASR (1=success, 0=refused)")
        plt.title(
            "HarmBench Aggregated Per-Sample Attack Success\n"
            "Left bars: Base, Right bars: RR (Dataset: HarmBench) — Mean ± SEM"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_agg_per_sample_asr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated per-sample ASR plot: {e}")
    plt.close()

print("Plots saved to", working_dir)
