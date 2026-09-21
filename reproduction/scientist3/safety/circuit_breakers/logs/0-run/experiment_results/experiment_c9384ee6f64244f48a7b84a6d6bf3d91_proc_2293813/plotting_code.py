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

# Plot 2: Loss weight schedule
try:
    loss_log = hb.get("loss_log", [])
    if loss_log and "w_harm" in loss_log[0]:
        steps = [l["step"] for l in loss_log]
        plt.figure(figsize=(8, 4))
        plt.plot(steps, [l["w_harm"] for l in loss_log], label="w_harm")
        plt.plot(steps, [l["w_retain"] for l in loss_log], label="w_retain")
        plt.xlabel("Step")
        plt.ylabel("Weight")
        plt.title(
            "HarmBench RR Loss Weight Schedule\nDataset: Circuit Breakers Train (cosine schedule)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_loss_weight_schedule.png"))
    plt.close()
except Exception as e:
    print(f"Error creating weight schedule plot: {e}")
    plt.close()

# Plot 3: Overall ASR bar chart (base vs RR) with CI
try:
    base_asr = hb.get("base_asr")
    rr_asr = hb.get("rr_asr")
    base_ci = hb.get("base_ci")
    rr_ci = hb.get("rr_ci")
    if base_asr is not None and rr_asr is not None:
        plt.figure(figsize=(5, 4))
        vals = [base_asr, rr_asr]
        errs = None
        if base_ci and rr_ci:
            errs = [
                [base_asr - base_ci[0], rr_asr - rr_ci[0]],
                [base_ci[1] - base_asr, rr_ci[1] - rr_asr],
            ]
        bars = plt.bar(
            ["Base Model", "RR Model"],
            vals,
            yerr=errs,
            capsize=6,
            color=["tab:red", "tab:green"],
        )
        for b, v in zip(bars, vals):
            plt.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center")
        plt.ylabel("Attack Success Rate")
        plt.ylim(0, max(1.0, max(vals) + 0.15))
        plt.title(
            "HarmBench Overall ASR: Base vs Representation-Rerouted\nDataset: HarmBench (lower is better, error bars = 95% CI)"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_asr_base_vs_rr.png"))
    plt.close()
except Exception as e:
    print(f"Error creating ASR bar plot: {e}")
    plt.close()

# Plot 4: ASR by attack style (comparison across attack "datasets")
try:
    base_by = hb.get("base_asr_by_attack", {})
    rr_by = hb.get("rr_asr_by_attack", {})
    if base_by and rr_by:
        styles = list(base_by.keys())
        x = np.arange(len(styles))
        w = 0.35
        plt.figure(figsize=(9, 5))
        plt.bar(
            x - w / 2, [base_by[s] for s in styles], w, label="Base", color="tab:red"
        )
        plt.bar(x + w / 2, [rr_by[s] for s in styles], w, label="RR", color="tab:green")
        for i, s in enumerate(styles):
            plt.text(
                i - w / 2,
                base_by[s] + 0.01,
                f"{base_by[s]:.2f}",
                ha="center",
                fontsize=8,
            )
            plt.text(
                i + w / 2, rr_by[s] + 0.01, f"{rr_by[s]:.2f}", ha="center", fontsize=8
            )
        plt.xticks(x, styles, rotation=20, ha="right")
        plt.ylabel("Attack Success Rate")
        plt.ylim(0, 1.1)
        plt.title(
            "HarmBench ASR by Attack Style\nLeft bars: Base, Right bars: RR (Dataset: HarmBench sub-styles)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_asr_by_attack_style.png"))
    plt.close()
except Exception as e:
    print(f"Error creating per-style ASR plot: {e}")
    plt.close()

# Plot 5: Per-sample ASR breakdown
try:
    samples_base = hb.get("samples_base", [])
    samples_rr = hb.get("samples_rr", [])
    if samples_base and samples_rr:
        n = min(len(samples_base), len(samples_rr))
        idx = np.arange(n)
        base_scores = [s["asr"] for s in samples_base[:n]]
        rr_scores = [s["asr"] for s in samples_rr[:n]]
        width = 0.4
        plt.figure(figsize=(10, 4))
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

# Plot 6: Generalization probe (cosine similarity pre vs post)
try:
    probe = hb.get("generalization_probe", {})
    pre_cos = probe.get("pre_train_cos")
    post_cos = probe.get("post_train_cos")
    if pre_cos is not None and post_cos is not None:
        plt.figure(figsize=(5, 4))
        bars = plt.bar(
            ["Pre-train", "Post-train"],
            [pre_cos, post_cos],
            color=["gray", "steelblue"],
        )
        for b, v in zip(bars, [pre_cos, post_cos]):
            plt.text(
                b.get_x() + b.get_width() / 2,
                v + (0.02 if v >= 0 else -0.05),
                f"{v:.3f}",
                ha="center",
            )
        plt.axhline(0, color="k", ls="--", lw=0.5)
        plt.ylabel("Mean cos(h_lora, h_base)")
        plt.title(
            "Generalization Probe: Held-out Harmful Cosine Sim\nDataset: Circuit Breakers held-out (lower = better rerouting)"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_generalization_probe_cos.png"))
    plt.close()
except Exception as e:
    print(f"Error creating generalization probe plot: {e}")
    plt.close()

# Plot 7: Benign response text comparison
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
            "Benign Response Comparison (Top: Before RR, Bottom: After RR)\nDataset: Llama-3-8B-Instruct on benign query"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "harmbench_benign_response_comparison.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating benign comparison plot: {e}")
    plt.close()

# Plot 8: Cross-comparison summary (ASR delta by attack style)
try:
    base_by = hb.get("base_asr_by_attack", {})
    rr_by = hb.get("rr_asr_by_attack", {})
    if base_by and rr_by:
        styles = list(base_by.keys())
        deltas = [rr_by[s] - base_by[s] for s in styles]
        colors = ["tab:green" if d < 0 else "tab:red" for d in deltas]
        plt.figure(figsize=(8, 4))
        bars = plt.bar(styles, deltas, color=colors)
        for b, v in zip(bars, deltas):
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
            "HarmBench ASR Change by Attack Style (RR - Base)\nDataset: HarmBench sub-styles (negative = RR safer)"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_asr_delta_by_style.png"))
    plt.close()
except Exception as e:
    print(f"Error creating ASR delta plot: {e}")
    plt.close()

print("Plots saved to", working_dir)
print(f"base_asr: {hb.get('base_asr')}, rr_asr: {hb.get('rr_asr')}")
