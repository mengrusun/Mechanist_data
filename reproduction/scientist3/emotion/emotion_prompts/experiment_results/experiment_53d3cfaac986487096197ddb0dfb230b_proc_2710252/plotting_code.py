import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = None

if experiment_data is not None:
    data = experiment_data["max_new_tokens"]["GSM8K"]
    templates = list(data["templates"].keys())
    hparam_values = data["hparam_values"]
    hparam_keys = [f"mnt_{m}" for m in hparam_values]
    acc_pt = data["accuracy_per_template"]
    acc_ph = data["accuracy_per_hparam"]
    n = data["n_samples"]

    # Plot 1: Grouped bar chart accuracy by template x hparam
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(templates))
        width = 0.8 / len(hparam_keys)
        for i, hk in enumerate(hparam_keys):
            vals = [acc_pt[hk][t] for t in templates]
            ax.bar(x + i * width, vals, width, label=hk)
        ax.set_xticks(x + width * (len(hparam_keys) - 1) / 2)
        ax.set_xticklabels(templates, rotation=20)
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"GSM8K: Accuracy by Emotional Prefix × max_new_tokens (N={n})")
        ax.legend(title="max_new_tokens")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gsm8k_accuracy_by_template_and_mnt.png"), dpi=120
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot1: {e}")
        plt.close()

    # Plot 2: Mean accuracy across templates per hparam
    try:
        plt.figure(figsize=(7, 4))
        means = [acc_ph[hk] for hk in hparam_keys]
        plt.plot(hparam_values, means, marker="o")
        for xv, yv in zip(hparam_values, means):
            plt.text(xv, yv + 0.005, f"{yv:.3f}", ha="center")
        plt.xlabel("max_new_tokens")
        plt.ylabel("Mean Accuracy")
        plt.title(
            f"GSM8K: Mean Accuracy Across Emotional Templates vs max_new_tokens (N={n})"
        )
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gsm8k_mean_accuracy_vs_mnt.png"), dpi=120
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot2: {e}")
        plt.close()

    # Plot 3: Heatmap of accuracy (templates x hparams)
    try:
        mat = np.array([[acc_pt[hk][t] for hk in hparam_keys] for t in templates])
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(len(hparam_keys)))
        ax.set_xticklabels(hparam_keys)
        ax.set_yticks(range(len(templates)))
        ax.set_yticklabels(templates)
        for i in range(len(templates)):
            for j in range(len(hparam_keys)):
                ax.text(
                    j,
                    i,
                    f"{mat[i,j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if mat[i, j] < 0.5 else "black",
                )
        ax.set_title(
            f"GSM8K Accuracy Heatmap: Emotional Templates × max_new_tokens (N={n})"
        )
        plt.colorbar(im, ax=ax, label="Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "gsm8k_accuracy_heatmap.png"), dpi=120)
        plt.close()
    except Exception as e:
        print(f"Error creating plot3: {e}")
        plt.close()

    # Plot 4: Line plot per-template accuracy across hparams
    try:
        plt.figure(figsize=(9, 5))
        for t in templates:
            vals = [acc_pt[hk][t] for hk in hparam_keys]
            plt.plot(hparam_values, vals, marker="o", label=t)
        plt.xlabel("max_new_tokens")
        plt.ylabel("Accuracy")
        plt.title(f"GSM8K: Per-Template Accuracy vs max_new_tokens (N={n})")
        plt.legend(fontsize=8, loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gsm8k_per_template_accuracy_lines.png"), dpi=120
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot4: {e}")
        plt.close()

    # Print summary metrics
    try:
        print("=== GSM8K Accuracy Summary ===")
        print(f"N samples: {n}")
        for hk in hparam_keys:
            print(f"  {hk}: mean_acc = {acc_ph[hk]:.4f}")
        print("\nPer-template best hparam:")
        for t in templates:
            best_hk = max(hparam_keys, key=lambda hk: acc_pt[hk][t])
            print(f"  {t}: best={best_hk} ({acc_pt[best_hk][t]:.4f})")
    except Exception as e:
        print(f"Error printing summary: {e}")
