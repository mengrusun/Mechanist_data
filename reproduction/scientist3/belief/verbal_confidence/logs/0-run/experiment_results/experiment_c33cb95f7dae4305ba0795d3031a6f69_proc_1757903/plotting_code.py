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

ed = experiment_data.get("probe_max_iter", {}).get("triviaqa_gemma3_27b_pt", {})
per_config = ed.get("per_config", {})
layer_indices = ed.get("layer_indices", [])
maj_conf = ed.get("majority_baseline_conf", None)
maj_corr = ed.get("majority_baseline_corr", None)
mi_keys = sorted(per_config.keys(), key=lambda x: int(x))

# Plot 1: Confidence probe accuracy vs layer for each max_iter
try:
    plt.figure(figsize=(10, 6))
    for mi in mi_keys:
        cfg = per_config[mi]
        plt.plot(
            cfg["layer_indices"],
            cfg["per_layer_confidence_acc"],
            "o-",
            label=f"max_iter={mi}",
        )
    if maj_conf is not None:
        plt.axhline(
            maj_conf, color="gray", linestyle=":", label=f"Majority ({maj_conf:.3f})"
        )
    plt.xlabel("Layer")
    plt.ylabel("Probe Accuracy")
    plt.title(
        "TriviaQA (Gemma3-27B-PT): Confidence Probe Accuracy vs Layer\nAcross max_iter values"
    )
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, "triviaqa_gemma3_27b_pt_confidence_probe_by_layer.png"
        ),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error plot1: {e}")
    plt.close()

# Plot 2: Correctness probe accuracy vs layer
try:
    plt.figure(figsize=(10, 6))
    for mi in mi_keys:
        cfg = per_config[mi]
        plt.plot(
            cfg["layer_indices"],
            cfg["per_layer_correctness_acc"],
            "o-",
            label=f"max_iter={mi}",
        )
    if maj_corr is not None:
        plt.axhline(
            maj_corr, color="gray", linestyle=":", label=f"Majority ({maj_corr:.3f})"
        )
    plt.xlabel("Layer")
    plt.ylabel("Probe Accuracy")
    plt.title(
        "TriviaQA (Gemma3-27B-PT): Correctness Probe Accuracy vs Layer\nAcross max_iter values"
    )
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, "triviaqa_gemma3_27b_pt_correctness_probe_by_layer.png"
        ),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error plot2: {e}")
    plt.close()

# Plot 3: Post-answer vs Pre-answer (control) at best max_iter
try:
    best_mi = ed.get("best_max_iter", None)
    if best_mi is not None:
        cfg = per_config[str(best_mi)]
        plt.figure(figsize=(10, 6))
        plt.plot(
            cfg["layer_indices"],
            cfg["per_layer_confidence_acc"],
            "o-",
            label="Post-answer (conf)",
        )
        plt.plot(
            cfg["layer_indices"],
            cfg["per_layer_control_conf_acc"],
            "s--",
            label="Pre-answer (control)",
        )
        if maj_conf is not None:
            plt.axhline(
                maj_conf,
                color="gray",
                linestyle=":",
                label=f"Majority ({maj_conf:.3f})",
            )
        plt.xlabel("Layer")
        plt.ylabel("Probe Accuracy")
        plt.title(
            f"TriviaQA (Gemma3-27B-PT): Post-answer vs Pre-answer Confidence Probe\nBest max_iter={best_mi}"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "triviaqa_gemma3_27b_pt_post_vs_pre_control.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error plot3: {e}")
    plt.close()

# Plot 4: Best probe accuracy vs max_iter
try:
    mis = [int(k) for k in mi_keys]
    best_conf = [per_config[k]["cache_probe_accuracy"] for k in mi_keys]
    best_corr = [per_config[k]["corr_probe_accuracy"] for k in mi_keys]
    plt.figure(figsize=(8, 5))
    plt.plot(mis, best_conf, "o-", label="Best confidence probe acc")
    plt.plot(mis, best_corr, "s-", label="Best correctness probe acc")
    plt.xscale("log")
    plt.xlabel("max_iter")
    plt.ylabel("Best Probe Accuracy (over layers)")
    plt.title("TriviaQA (Gemma3-27B-PT): Best Probe Accuracy vs max_iter")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_gemma3_27b_pt_best_acc_vs_max_iter.png"),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error plot4: {e}")
    plt.close()

# Plot 5: Distribution of raw confidences by correctness
try:
    confs = np.array(ed.get("raw_confidences", []))
    corrs = np.array(ed.get("raw_correctness", []))
    if len(confs) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(
            [confs[corrs == 1], confs[corrs == 0]],
            bins=20,
            label=["Correct", "Incorrect"],
            stacked=False,
        )
        axes[0].set_xlabel("Continuous Confidence")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Left: Confidence Distribution by Correctness")
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        # Accuracy per confidence bin
        bins = np.linspace(confs.min(), confs.max(), 8)
        bin_idx = np.digitize(confs, bins)
        bin_centers = []
        bin_acc = []
        for b in range(1, len(bins)):
            mask = bin_idx == b
            if mask.sum() > 0:
                bin_centers.append((bins[b - 1] + bins[b]) / 2)
                bin_acc.append(corrs[mask].mean())
        axes[1].plot(bin_centers, bin_acc, "o-")
        axes[1].set_xlabel("Confidence Bin Center")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Right: Calibration (Accuracy vs Confidence)")
        axes[1].grid(alpha=0.3)
        plt.suptitle("TriviaQA (Gemma3-27B-PT): Confidence Calibration")
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "triviaqa_gemma3_27b_pt_confidence_calibration.png"
            ),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error plot5: {e}")
    plt.close()

# Print summary metrics
try:
    print(f"Best max_iter: {ed.get('best_max_iter')}")
    print(f"Best cache_probe_accuracy: {ed.get('best_cache_probe_accuracy')}")
    print(f"Majority baseline conf: {maj_conf}")
    print(f"Majority baseline corr: {maj_corr}")
    for k in mi_keys:
        cfg = per_config[k]
        print(
            f"max_iter={k}: conf_acc={cfg['cache_probe_accuracy']:.4f} corr_acc={cfg['corr_probe_accuracy']:.4f}"
        )
except Exception as e:
    print(f"Error summary: {e}")
