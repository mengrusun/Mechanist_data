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
    hb = experiment_data["N_STEPS"]["harmbench"]
    n_steps_list = hb["n_steps_values"]
    runs = hb["runs"]
    base_asr = hb.get("base_asr", None)

    # Plot 1: total training loss curves
    try:
        plt.figure(figsize=(8, 5))
        for ns in n_steps_list:
            r = runs[str(ns)]
            losses = [l["loss"] for l in r["loss_log"]]
            plt.plot(losses, label=f"N_STEPS={ns}")
        plt.xlabel("Step")
        plt.ylabel("Total Loss")
        plt.title(
            "HarmBench Dataset: Total Training Loss Curves\nAcross Different N_STEPS Values"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_total_loss_curves.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot1: {e}")
        plt.close()

    # Plot 2: component losses (harm, retain_ce, rep_diff) in subplots
    try:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for ns in n_steps_list:
            r = runs[str(ns)]
            axes[0].plot([l["harm_loss"] for l in r["loss_log"]], label=f"n={ns}")
            axes[1].plot([l["retain_ce"] for l in r["loss_log"]], label=f"n={ns}")
            axes[2].plot([l["rep_diff"] for l in r["loss_log"]], label=f"n={ns}")
        axes[0].set_title("Left: Harm Loss")
        axes[1].set_title("Middle: Retain CE")
        axes[2].set_title("Right: Rep Diff")
        for ax in axes:
            ax.set_xlabel("Step")
            ax.legend(fontsize=8)
        fig.suptitle("HarmBench Dataset: Component Loss Curves across N_STEPS")
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_component_losses.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot2: {e}")
        plt.close()

    # Plot 3: ASR bar chart (base vs RR at each N_STEPS)
    try:
        plt.figure(figsize=(8, 5))
        asrs = [runs[str(ns)]["rr_asr"] for ns in n_steps_list]
        labels = ["base"] + [f"RR n={ns}" for ns in n_steps_list]
        vals = [base_asr if base_asr is not None else 0] + asrs
        colors = ["gray"] + ["steelblue"] * len(n_steps_list)
        plt.bar(labels, vals, color=colors)
        plt.ylabel("Attack Success Rate")
        plt.title(
            "HarmBench Dataset: ASR Comparison\n(Base Model vs RR-Tuned at Various N_STEPS, lower is better)"
        )
        plt.xticks(rotation=30, ha="right")
        for i, v in enumerate(vals):
            plt.text(i, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_asr_comparison.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot3: {e}")
        plt.close()

    # Plot 4: actual steps run vs configured N_STEPS (early abort/stop visualization)
    try:
        plt.figure(figsize=(7, 5))
        configured = n_steps_list
        actual = [runs[str(ns)]["actual_steps"] for ns in n_steps_list]
        aborted = [runs[str(ns)]["aborted"] for ns in n_steps_list]
        x = np.arange(len(configured))
        width = 0.35
        plt.bar(x - width / 2, configured, width, label="Configured")
        plt.bar(x + width / 2, actual, width, label="Actual")
        plt.xticks(x, [str(ns) for ns in configured])
        plt.xlabel("Configured N_STEPS")
        plt.ylabel("Training Steps")
        plt.title(
            "HarmBench Dataset: Configured vs Actual Training Steps\n(early stop / abort may reduce actual)"
        )
        for i, ab in enumerate(aborted):
            if ab:
                plt.text(
                    i + width / 2,
                    actual[i] + 2,
                    "aborted",
                    ha="center",
                    color="red",
                    fontsize=8,
                )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "harmbench_configured_vs_actual_steps.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot4: {e}")
        plt.close()

    # Plot 5: ASR vs N_STEPS line plot with base reference
    try:
        plt.figure(figsize=(7, 5))
        asrs = [runs[str(ns)]["rr_asr"] for ns in n_steps_list]
        plt.plot(n_steps_list, asrs, marker="o", label="RR-tuned ASR")
        if base_asr is not None:
            plt.axhline(
                base_asr, color="red", linestyle="--", label=f"Base ASR={base_asr:.2f}"
            )
        plt.xlabel("N_STEPS")
        plt.ylabel("Attack Success Rate")
        plt.title("HarmBench Dataset: RR ASR vs N_STEPS\n(lower is stronger defense)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "harmbench_asr_vs_nsteps.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot5: {e}")
        plt.close()

    # Print summary metrics
    try:
        print(f"Base ASR: {base_asr}")
        for ns in n_steps_list:
            r = runs[str(ns)]
            print(
                f"N_STEPS={ns}: rr_asr={r['rr_asr']:.3f}, actual_steps={r['actual_steps']}, aborted={r['aborted']}"
            )
        best_ns = min(n_steps_list, key=lambda ns: runs[str(ns)]["rr_asr"])
        print(f"Best N_STEPS={best_ns}, ASR={runs[str(best_ns)]['rr_asr']:.3f}")
    except Exception as e:
        print(f"Error printing metrics: {e}")
