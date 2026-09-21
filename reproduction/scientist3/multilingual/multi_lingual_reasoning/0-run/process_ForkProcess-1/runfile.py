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
    experiment_data = None

if experiment_data is not None:
    mgsm = experiment_data.get("mgsm", {})
    conditions = mgsm.get("conditions", {})
    cond_names = ["baseline", "lang_suppression", "random_control"]
    cond_names = [c for c in cond_names if c in conditions]
    colors = ["gray", "tab:blue", "tab:orange"]

    # Plot 1: Overall accuracy bar chart
    try:
        plt.figure(figsize=(6, 4))
        vals = [conditions[c].get("overall", 0.0) for c in cond_names]
        plt.bar(cond_names, vals, color=colors[: len(cond_names)])
        for i, v in enumerate(vals):
            plt.text(i, v + 0.01, f"{v:.3f}", ha="center")
        plt.ylabel("multilingual_reasoning_accuracy")
        plt.title(
            "MGSM Overall Accuracy by Condition\nDataset: MGSM (Qwen3-4B-Thinking)"
        )
        plt.ylim(0, max(vals) * 1.3 + 0.05 if max(vals) > 0 else 1.0)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "mgsm_overall_accuracy_bar.png"), dpi=120)
        plt.close()
    except Exception as e:
        print(f"Error creating overall accuracy plot: {e}")
        plt.close()

    # Plot 2: Per-language grouped bar chart
    try:
        langs = sorted(
            {l for c in cond_names for l in conditions[c].get("per_lang", {}).keys()}
        )
        if langs:
            x = np.arange(len(langs))
            width = 0.8 / max(1, len(cond_names))
            plt.figure(figsize=(max(8, len(langs) * 0.8), 4.5))
            for i, c in enumerate(cond_names):
                per_lang = conditions[c].get("per_lang", {})
                yvals = [per_lang.get(l, 0.0) for l in langs]
                plt.bar(
                    x + i * width - 0.4 + width / 2,
                    yvals,
                    width,
                    label=c,
                    color=colors[i],
                )
            plt.xticks(x, langs)
            plt.ylabel("accuracy")
            plt.xlabel("language")
            plt.title(
                "MGSM Per-Language Accuracy by Condition\nDataset: MGSM (Qwen3-4B-Thinking)"
            )
            plt.legend()
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "mgsm_per_language_grouped_bar.png"), dpi=120
            )
            plt.close()
    except Exception as e:
        print(f"Error creating per-language grouped bar: {e}")
        plt.close()

    # Plot 3: Heatmap of per-language accuracy
    try:
        langs = sorted(
            {l for c in cond_names for l in conditions[c].get("per_lang", {}).keys()}
        )
        if langs:
            mat = np.array(
                [
                    [conditions[c].get("per_lang", {}).get(l, 0.0) for l in langs]
                    for c in cond_names
                ]
            )
            plt.figure(figsize=(max(8, len(langs) * 0.7), 3 + 0.4 * len(cond_names)))
            im = plt.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            plt.colorbar(im, label="accuracy")
            plt.yticks(range(len(cond_names)), cond_names)
            plt.xticks(range(len(langs)), langs)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    plt.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if mat[i, j] < 0.5 else "black",
                        fontsize=8,
                    )
            plt.title(
                "MGSM Per-Language Accuracy Heatmap\nRows: Conditions, Cols: Languages (Qwen3-4B-Thinking)"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "mgsm_per_language_heatmap.png"), dpi=120
            )
            plt.close()
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        plt.close()

    # Plot 4: Validation "loss" (1 - accuracy) across conditions/epochs
    try:
        val_metrics = mgsm.get("metrics", {}).get("val", [])
        if val_metrics:
            epochs = [m["epoch"] for m in val_metrics]
            losses = [1.0 - m["multilingual_reasoning_accuracy"] for m in val_metrics]
            labels = [m["condition"] for m in val_metrics]
            plt.figure(figsize=(6, 4))
            plt.plot(epochs, losses, marker="o", color="crimson")
            for e, l, lab in zip(epochs, losses, labels):
                plt.text(e, l + 0.01, lab, ha="center", fontsize=8)
            plt.xlabel("epoch (condition index)")
            plt.ylabel("validation_loss (1 - accuracy)")
            plt.title(
                "MGSM Validation Loss across Conditions\nDataset: MGSM (Qwen3-4B-Thinking)"
            )
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "mgsm_validation_loss_curve.png"), dpi=120
            )
            plt.close()
    except Exception as e:
        print(f"Error creating validation loss plot: {e}")
        plt.close()

    # Print evaluation metrics
    try:
        print("=== MGSM multilingual_reasoning_accuracy ===")
        for c in cond_names:
            print(f"  {c}: {conditions[c].get('overall', 0.0):.4f}")
    except Exception as e:
        print(f"Error printing metrics: {e}")
