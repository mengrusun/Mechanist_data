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

root = experiment_data.get("single_vs_multi_layer_RR", {})
configs = ["multi_layer", "single_layer"]

# Plot 1: Training loss curves
try:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, cfg in enumerate(configs):
        rec = root.get(cfg, {}).get("harmbench", {})
        losses_log = rec.get("loss_log", [])
        ax = axes[i]
        if losses_log:
            steps = [l["step"] for l in losses_log]
            ax.plot(steps, [l["harm_loss"] for l in losses_log], label="harm_loss")
            ax.plot(steps, [l["retain_ce"] for l in losses_log], label="retain_ce")
            ax.plot(steps, [l["rep_diff"] for l in losses_log], label="rep_diff")
            ax.plot(
                steps,
                [l["loss"] for l in losses_log],
                label="total",
                alpha=0.5,
                ls="--",
            )
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title(f"{cfg}")
        ax.legend(fontsize=8)
    fig.suptitle(
        "HarmBench RR Ablation: Training Loss Curves (Left: multi_layer, Right: single_layer)"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "harmbench_rr_training_losses.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating loss curves plot: {e}")
    plt.close()

# Plot 2: ASR by attack style, base vs RR
try:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, cfg in enumerate(configs):
        rec = root.get(cfg, {}).get("harmbench", {})
        base_by = rec.get("base_asr_by_attack", {})
        rr_by = rec.get("rr_asr_by_attack", {})
        styles_list = list(base_by.keys())
        ax = axes[i]
        if styles_list:
            x = np.arange(len(styles_list))
            w = 0.35
            ax.bar(
                x - w / 2,
                [base_by[s] for s in styles_list],
                w,
                label="base",
                color="tomato",
            )
            ax.bar(
                x + w / 2,
                [rr_by.get(s, 0) for s in styles_list],
                w,
                label="RR",
                color="steelblue",
            )
            ax.set_xticks(x)
            ax.set_xticklabels(styles_list, rotation=30, ha="right")
        ax.set_ylabel("ASR")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{cfg}")
        ax.legend()
    fig.suptitle(
        "HarmBench: ASR by Attack Style (Left: multi_layer, Right: single_layer)"
    )
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "harmbench_asr_by_attack_style.png"), dpi=120)
    plt.close()
except Exception as e:
    print(f"Error creating ASR by style plot: {e}")
    plt.close()

# Plot 3: Representation probe cosine similarity pre/post
try:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(configs))
    w = 0.35
    pre_vals = [
        root.get(c, {})
        .get("harmbench", {})
        .get("generalization_probe", {})
        .get("pre_train_cos", 0)
        for c in configs
    ]
    post_vals = [
        root.get(c, {})
        .get("harmbench", {})
        .get("generalization_probe", {})
        .get("post_train_cos", 0)
        for c in configs
    ]
    ax.bar(x - w / 2, pre_vals, w, label="pre-train", color="gray")
    ax.bar(x + w / 2, post_vals, w, label="post-train", color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Cosine similarity (LoRA vs base)")
    ax.axhline(0, color="k", ls="--", lw=0.5)
    ax.legend()
    ax.set_title(
        "HarmBench RR Ablation: Held-out Harm Representation Probe\n(Pre vs Post training cosine similarity)"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "harmbench_representation_probe_cosine.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating probe plot: {e}")
    plt.close()

# Plot 4: Overall ASR base vs RR across configs, with CI
try:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(configs))
    w = 0.35
    base_asrs = [
        root.get(c, {}).get("harmbench", {}).get("base_asr", 0) for c in configs
    ]
    rr_asrs = [root.get(c, {}).get("harmbench", {}).get("rr_asr", 0) for c in configs]
    base_cis = [
        root.get(c, {}).get("harmbench", {}).get("base_ci", (0, 0)) for c in configs
    ]
    rr_cis = [
        root.get(c, {}).get("harmbench", {}).get("rr_ci", (0, 0)) for c in configs
    ]
    base_err = np.array(
        [[a - ci[0], ci[1] - a] for a, ci in zip(base_asrs, base_cis)]
    ).T
    rr_err = np.array([[a - ci[0], ci[1] - a] for a, ci in zip(rr_asrs, rr_cis)]).T
    ax.bar(
        x - w / 2, base_asrs, w, yerr=base_err, label="base", color="tomato", capsize=4
    )
    ax.bar(x + w / 2, rr_asrs, w, yerr=rr_err, label="RR", color="steelblue", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("Overall ASR")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(
        "HarmBench: Overall ASR with 95% CI (Base vs RR)\nSingle vs Multi-layer Representation Rerouting"
    )
    for i, (b, r) in enumerate(zip(base_asrs, rr_asrs)):
        ax.text(i, max(b, r) + 0.05, f"Δ={r-b:+.2f}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "harmbench_overall_asr_comparison.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating overall ASR plot: {e}")
    plt.close()

# Print summary metrics
try:
    print("\n=== Summary Metrics ===")
    for cfg in configs:
        rec = root.get(cfg, {}).get("harmbench", {})
        b = rec.get("base_asr", None)
        r = rec.get("rr_asr", None)
        print(
            f"{cfg}: base_asr={b}, rr_asr={r}, delta={None if (b is None or r is None) else r-b}"
        )
        print(f"  base_ci={rec.get('base_ci')}, rr_ci={rec.get('rr_ci')}")
        print(
            f"  probe pre={rec.get('generalization_probe', {}).get('pre_train_cos')}, "
            f"post={rec.get('generalization_probe', {}).get('post_train_cos')}"
        )
except Exception as e:
    print(f"Error printing summary: {e}")
