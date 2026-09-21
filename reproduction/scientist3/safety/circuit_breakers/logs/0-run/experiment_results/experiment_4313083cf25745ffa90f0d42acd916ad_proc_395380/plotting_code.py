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

hb = experiment_data.get("harmbench", {})

# Plot 1: Training loss curves
try:
    loss_log = hb.get("loss_log", [])
    if loss_log:
        steps = [l["step"] for l in loss_log]
        plt.figure(figsize=(8, 5))
        plt.plot(steps, [l["loss"] for l in loss_log], label="total loss")
        plt.plot(steps, [l["harm_loss"] for l in loss_log], label="harm_loss")
        plt.plot(steps, [l["retain_ce"] for l in loss_log], label="retain_ce")
        plt.plot(steps, [l["rep_diff"] for l in loss_log], label="rep_diff")
        plt.xlabel("Step")
        plt.ylabel("Loss")
        plt.title(
            "HarmBench RR Training Loss Curves\nDataset: Circuit Breakers Train (Llama-3-8B-Instruct)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_training_loss_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating loss curve plot: {e}")
    plt.close()

# Plot 2: ASR bar chart (base vs RR)
try:
    base_asr = hb.get("base_asr")
    rr_asr = hb.get("rr_asr")
    if base_asr is not None and rr_asr is not None:
        plt.figure(figsize=(5, 4))
        bars = plt.bar(
            ["Base Model", "RR Model"],
            [base_asr, rr_asr],
            color=["tab:red", "tab:green"],
        )
        for b, v in zip(bars, [base_asr, rr_asr]):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
        plt.ylabel("Attack Success Rate")
        plt.ylim(0, max(1.0, max(base_asr, rr_asr) + 0.1))
        plt.title(
            "HarmBench ASR: Base vs Representation-Rerouted\nDataset: HarmBench (lower is better)"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_asr_base_vs_rr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating ASR bar plot: {e}")
    plt.close()

# Plot 3: Per-sample ASR breakdown
try:
    samples_base = hb.get("samples_base", [])
    samples_rr = hb.get("samples_rr", [])
    if samples_base and samples_rr:
        n = min(len(samples_base), len(samples_rr))
        idx = np.arange(n)
        base_scores = [s["asr"] for s in samples_base[:n]]
        rr_scores = [s["asr"] for s in samples_rr[:n]]
        width = 0.4
        plt.figure(figsize=(9, 4))
        plt.bar(idx - width / 2, base_scores, width, label="Base", color="tab:red")
        plt.bar(idx + width / 2, rr_scores, width, label="RR", color="tab:green")
        plt.xlabel("Sample Index")
        plt.ylabel("ASR (1=success, 0=refused/gibberish)")
        plt.title(
            "HarmBench Per-Sample Attack Success\nLeft bars: Base, Right bars: RR (Dataset: HarmBench)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_per_sample_asr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating per-sample ASR plot: {e}")
    plt.close()

# Plot 4: Benign response text comparison
try:
    before = hb.get("benign_before", "")
    after = hb.get("benign_after", "")
    if before or after:
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.axis("off")
        text = (
            f"Benign Prompt: 'What is the capital of France?'\n\n"
            f"BEFORE RR:\n{before}\n\n"
            f"AFTER RR:\n{after}"
        )
        ax.text(
            0.02,
            0.98,
            text,
            va="top",
            ha="left",
            fontsize=9,
            family="monospace",
            wrap=True,
        )
        plt.title(
            "Benign Response Comparison (Left/Top: Before RR, Right/Bottom: After RR)\nDataset: Llama-3-8B-Instruct on benign query"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "harmbench_benign_response_comparison.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating benign comparison plot: {e}")
    plt.close()

print("Plots saved to", working_dir)
print(f"base_asr: {hb.get('base_asr')}, rr_asr: {hb.get('rr_asr')}")
