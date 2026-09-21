import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_02-20-28_circuit_breakers_attempt_1/logs/0-run/experiment_results/experiment_9f6ddc7a6c2d48009b94b32a982cba4d_proc_395380/experiment_data.npy"
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

# Extract harmbench data from each run
hb_runs = [ed.get("harmbench", {}) for ed in all_experiment_data]
n_runs = len(hb_runs)
print(f"Number of runs loaded: {n_runs}")


def sem(arr, axis=0):
    arr = np.asarray(arr, dtype=float)
    if arr.shape[axis] <= 1:
        return np.zeros(arr.shape[1:] if arr.ndim > 1 else ())
    return np.std(arr, axis=axis, ddof=1) / np.sqrt(arr.shape[axis])


# ---------- Plot 1: Aggregated loss curves ----------
try:
    loss_keys = ["loss", "harm_loss", "retain_ce", "rep_diff"]
    # Collect per-run arrays aligned by step
    run_steps = []
    run_losses = {k: [] for k in loss_keys}
    for hb in hb_runs:
        loss_log = hb.get("loss_log", [])
        if not loss_log:
            continue
        steps = [l["step"] for l in loss_log]
        run_steps.append(steps)
        for k in loss_keys:
            run_losses[k].append([l.get(k, np.nan) for l in loss_log])

    if run_steps:
        # Use minimum common length
        min_len = min(len(s) for s in run_steps)
        steps_ref = run_steps[0][:min_len]

        plt.figure(figsize=(9, 6))
        colors = {
            "loss": "tab:blue",
            "harm_loss": "tab:orange",
            "retain_ce": "tab:green",
            "rep_diff": "tab:red",
        }
        for k in loss_keys:
            arr = np.array([r[:min_len] for r in run_losses[k]], dtype=float)
            mean = np.nanmean(arr, axis=0)
            se = sem(arr, axis=0) if arr.shape[0] > 1 else np.zeros_like(mean)
            plt.plot(steps_ref, mean, label=f"{k} (mean)", color=colors[k])
            if arr.shape[0] > 1:
                plt.fill_between(
                    steps_ref,
                    mean - se,
                    mean + se,
                    color=colors[k],
                    alpha=0.25,
                    label=f"{k} (±SE)",
                )
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(
            f"HarmBench RR Training Loss Curves (Aggregated over {n_runs} runs)\n"
            "Mean ± Standard Error | Dataset: Circuit Breakers Train (Llama-3-8B-Instruct)"
        )
        plt.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_aggregated_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated loss curve plot: {e}")
    plt.close()

# ---------- Plot 2: Aggregated ASR bar chart ----------
try:
    base_vals = [hb.get("base_asr") for hb in hb_runs if hb.get("base_asr") is not None]
    rr_vals = [hb.get("rr_asr") for hb in hb_runs if hb.get("rr_asr") is not None]

    if base_vals and rr_vals:
        base_arr = np.array(base_vals, dtype=float)
        rr_arr = np.array(rr_vals, dtype=float)
        base_mean, rr_mean = base_arr.mean(), rr_arr.mean()
        base_se = (
            base_arr.std(ddof=1) / np.sqrt(len(base_arr)) if len(base_arr) > 1 else 0.0
        )
        rr_se = rr_arr.std(ddof=1) / np.sqrt(len(rr_arr)) if len(rr_arr) > 1 else 0.0

        plt.figure(figsize=(6, 5))
        labels = ["Base Model", "RR Model"]
        means = [base_mean, rr_mean]
        ses = [base_se, rr_se]
        bars = plt.bar(
            labels,
            means,
            yerr=ses,
            capsize=8,
            color=["tab:red", "tab:green"],
            label="Mean ± SE",
        )
        for b, v in zip(bars, means):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
        plt.ylabel("Attack Success Rate")
        plt.ylim(0, max(1.0, max(means) + 0.15))
        plt.title(
            f"HarmBench ASR: Base vs RR (Aggregated over {n_runs} runs)\n"
            "Mean ± Standard Error | Dataset: HarmBench (lower is better)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_aggregated_asr_bar.png"))

        print(f"Base ASR: mean={base_mean:.4f}, SE={base_se:.4f}, n={len(base_arr)}")
        print(f"RR   ASR: mean={rr_mean:.4f}, SE={rr_se:.4f}, n={len(rr_arr)}")
    plt.close()
except Exception as e:
    print(f"Error creating aggregated ASR bar plot: {e}")
    plt.close()

# ---------- Plot 3: Aggregated per-sample ASR ----------
try:
    base_matrix = []
    rr_matrix = []
    for hb in hb_runs:
        sb = hb.get("samples_base", [])
        sr = hb.get("samples_rr", [])
        if sb and sr:
            n = min(len(sb), len(sr))
            base_matrix.append([s["asr"] for s in sb[:n]])
            rr_matrix.append([s["asr"] for s in sr[:n]])

    if base_matrix and rr_matrix:
        min_n = min(min(len(r) for r in base_matrix), min(len(r) for r in rr_matrix))
        base_arr = np.array([r[:min_n] for r in base_matrix], dtype=float)
        rr_arr = np.array([r[:min_n] for r in rr_matrix], dtype=float)

        base_mean = base_arr.mean(axis=0)
        rr_mean = rr_arr.mean(axis=0)
        base_se = sem(base_arr, axis=0) if base_arr.shape[0] > 1 else np.zeros(min_n)
        rr_se = sem(rr_arr, axis=0) if rr_arr.shape[0] > 1 else np.zeros(min_n)

        idx = np.arange(min_n)
        width = 0.4
        plt.figure(figsize=(max(9, min_n * 0.3), 5))
        plt.bar(
            idx - width / 2,
            base_mean,
            width,
            yerr=base_se,
            capsize=3,
            label="Base (mean ± SE)",
            color="tab:red",
        )
        plt.bar(
            idx + width / 2,
            rr_mean,
            width,
            yerr=rr_se,
            capsize=3,
            label="RR (mean ± SE)",
            color="tab:green",
        )
        plt.xlabel("Sample Index")
        plt.ylabel("ASR (1=success, 0=refused/gibberish)")
        plt.title(
            f"HarmBench Per-Sample ASR (Aggregated over {n_runs} runs)\n"
            "Left bars: Base, Right bars: RR | Dataset: HarmBench"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "harmbench_aggregated_per_sample_asr.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated per-sample ASR plot: {e}")
    plt.close()

# ---------- Plot 4: Summary text of aggregated metrics ----------
try:
    base_vals = [hb.get("base_asr") for hb in hb_runs if hb.get("base_asr") is not None]
    rr_vals = [hb.get("rr_asr") for hb in hb_runs if hb.get("rr_asr") is not None]
    if base_vals and rr_vals:
        base_arr = np.array(base_vals, dtype=float)
        rr_arr = np.array(rr_vals, dtype=float)
        reduction = base_arr.mean() - rr_arr.mean()

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        summary = (
            f"Aggregated Summary Across {n_runs} Run(s)\n"
            f"------------------------------------------\n"
            f"Base ASR : mean = {base_arr.mean():.4f}, "
            f"SE = {(base_arr.std(ddof=1)/np.sqrt(len(base_arr))) if len(base_arr)>1 else 0.0:.4f}\n"
            f"RR   ASR : mean = {rr_arr.mean():.4f}, "
            f"SE = {(rr_arr.std(ddof=1)/np.sqrt(len(rr_arr))) if len(rr_arr)>1 else 0.0:.4f}\n"
            f"ASR reduction (Base - RR) = {reduction:.4f}"
        )
        ax.text(
            0.02, 0.9, summary, va="top", ha="left", fontsize=11, family="monospace"
        )
        plt.title("HarmBench Aggregated Metrics Summary\nDataset: HarmBench")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_aggregated_summary.png"))
    plt.close()
except Exception as e:
    print(f"Error creating summary plot: {e}")
    plt.close()

print("Aggregated plots saved to", working_dir)
