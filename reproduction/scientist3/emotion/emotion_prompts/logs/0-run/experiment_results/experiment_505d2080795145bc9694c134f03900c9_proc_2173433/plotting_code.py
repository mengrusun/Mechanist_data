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
    data = experiment_data.get("GSM8K", {})
    acc_dict = data.get("accuracy_per_template", {})
    preds = data.get("predictions", {})
    gts = data.get("ground_truth", [])
    raw = data.get("raw_outputs", {})
    n = data.get("n_samples", len(gts))

    def norm(s):
        try:
            f = float(s)
            if f == int(f):
                return str(int(f))
            return f"{f:.4f}".rstrip("0").rstrip(".")
        except:
            return str(s).strip()

    # Plot 1: Accuracy per template
    try:
        names = list(acc_dict.keys())
        vals = [acc_dict[k] for k in names]
        plt.figure(figsize=(9, 5))
        colors = ["gray" if nm == "neutral" else f"C{i}" for i, nm in enumerate(names)]
        plt.bar(names, vals, color=colors)
        neutral = acc_dict.get("neutral", 0)
        plt.axhline(
            neutral,
            color="black",
            linestyle="--",
            alpha=0.5,
            label=f"neutral baseline ({neutral:.2f})",
        )
        for i, v in enumerate(vals):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1.0)
        plt.title(f"GSM8K: Accuracy by Emotional Prefix (Qwen, N={n})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "GSM8K_accuracy_by_emotion_prefix.png"), dpi=120
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot1: {e}")
        plt.close()

    # Plot 2: Accuracy delta vs neutral
    try:
        names = [k for k in acc_dict.keys() if k != "neutral"]
        neutral = acc_dict.get("neutral", 0)
        deltas = [acc_dict[k] - neutral for k in names]
        plt.figure(figsize=(9, 5))
        colors = ["green" if d >= 0 else "red" for d in deltas]
        plt.bar(names, deltas, color=colors)
        plt.axhline(0, color="black", linewidth=0.8)
        for i, v in enumerate(deltas):
            plt.text(i, v + (0.005 if v >= 0 else -0.015), f"{v:+.2f}", ha="center")
        plt.ylabel("Accuracy Δ vs neutral")
        plt.title(
            f"GSM8K: Accuracy Change vs Neutral Prefix (Qwen, N={n})\nPositive=emotion helps, Negative=emotion hurts"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "GSM8K_accuracy_delta_vs_neutral.png"), dpi=120
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot2: {e}")
        plt.close()

    # Plot 3: Correctness heatmap (templates x samples)
    try:
        names = list(preds.keys())
        if names and gts:
            mat = np.zeros((len(names), len(gts)))
            for i, nm in enumerate(names):
                for j, (p, g) in enumerate(zip(preds[nm], gts)):
                    mat[i, j] = 1.0 if norm(p) == norm(g) else 0.0
            plt.figure(figsize=(12, 4))
            plt.imshow(
                mat,
                aspect="auto",
                cmap="RdYlGn",
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            plt.yticks(range(len(names)), names)
            plt.xlabel("Sample index")
            plt.ylabel("Emotion template")
            plt.title(
                f"GSM8K: Per-Sample Correctness Heatmap\n(Green=Correct, Red=Incorrect, N={n})"
            )
            plt.colorbar(label="Correct")
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "GSM8K_correctness_heatmap.png"), dpi=120
            )
            plt.close()
    except Exception as e:
        print(f"Error creating plot3: {e}")
        plt.close()

    # Plot 4: Output length distribution per template
    try:
        names = list(raw.keys())
        if names:
            lengths = [[len(o) for o in raw[nm]] for nm in names]
            plt.figure(figsize=(9, 5))
            plt.boxplot(lengths, labels=names)
            plt.ylabel("Output length (chars)")
            plt.title(
                f"GSM8K: Output Length Distribution by Emotional Prefix (Qwen, N={n})"
            )
            plt.xticks(rotation=20)
            plt.tight_layout()
            plt.savefig(
                os.path.join(working_dir, "GSM8K_output_length_distribution.png"),
                dpi=120,
            )
            plt.close()
    except Exception as e:
        print(f"Error creating plot4: {e}")
        plt.close()

    # Print summary metric
    print("=== GSM8K Accuracy Summary ===")
    for k, v in acc_dict.items():
        print(f"  {k}: {v:.4f}")
