import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-14_01-42-41_emotion_prompts_attempt_1/logs/0-run/experiment_results/experiment_0b05169d1f2144d6a98835d2827efd8f_proc_2710252/experiment_data.npy"
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        experiment_data = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

if len(all_experiment_data) > 0:
    # Assume all runs share same templates and hparam_values
    first = all_experiment_data[0]["max_new_tokens"]["GSM8K"]
    templates = list(first["templates"].keys())
    hparam_values = first["hparam_values"]
    hparam_keys = [f"mnt_{m}" for m in hparam_values]
    n = first["n_samples"]
    n_runs = len(all_experiment_data)

    # Build arrays: shape (n_runs, n_templates, n_hparams) and (n_runs, n_hparams)
    acc_pt_arr = np.zeros((n_runs, len(templates), len(hparam_keys)))
    acc_ph_arr = np.zeros((n_runs, len(hparam_keys)))
    for r, ed in enumerate(all_experiment_data):
        d = ed["max_new_tokens"]["GSM8K"]
        for i, t in enumerate(templates):
            for j, hk in enumerate(hparam_keys):
                acc_pt_arr[r, i, j] = d["accuracy_per_template"][hk][t]
        for j, hk in enumerate(hparam_keys):
            acc_ph_arr[r, j] = d["accuracy_per_hparam"][hk]

    # Means and SE
    acc_pt_mean = acc_pt_arr.mean(axis=0)
    acc_pt_se = (
        acc_pt_arr.std(axis=0, ddof=1) / np.sqrt(n_runs)
        if n_runs > 1
        else np.zeros_like(acc_pt_mean)
    )
    acc_ph_mean = acc_ph_arr.mean(axis=0)
    acc_ph_se = (
        acc_ph_arr.std(axis=0, ddof=1) / np.sqrt(n_runs)
        if n_runs > 1
        else np.zeros_like(acc_ph_mean)
    )

    # Plot 1: Mean accuracy vs max_new_tokens with SE error bars (aggregated across templates & runs)
    try:
        # Aggregate across templates: mean over templates per run, then across runs
        per_run_mean_over_templates = acc_pt_arr.mean(axis=1)  # (n_runs, n_hparams)
        agg_mean = per_run_mean_over_templates.mean(axis=0)
        agg_se = (
            per_run_mean_over_templates.std(axis=0, ddof=1) / np.sqrt(n_runs)
            if n_runs > 1
            else np.zeros_like(agg_mean)
        )
        plt.figure(figsize=(7, 4))
        plt.errorbar(
            hparam_values,
            agg_mean,
            yerr=agg_se,
            marker="o",
            capsize=4,
            label=f"Mean ± SE (n_runs={n_runs})",
        )
        plt.xlabel("max_new_tokens")
        plt.ylabel("Mean Accuracy")
        plt.title(
            f"GSM8K: Mean Accuracy vs max_new_tokens\nAggregated across Emotional Templates (N={n})"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gsm8k_agg_mean_accuracy_vs_mnt.png"), dpi=120
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot1: {e}")
        plt.close()

    # Plot 2: Per-template accuracy vs max_new_tokens with error bars
    try:
        plt.figure(figsize=(10, 6))
        for i, t in enumerate(templates):
            plt.errorbar(
                hparam_values,
                acc_pt_mean[i],
                yerr=acc_pt_se[i],
                marker="o",
                capsize=3,
                label=t,
            )
        plt.xlabel("max_new_tokens")
        plt.ylabel("Accuracy (Mean ± SE)")
        plt.title(
            f"GSM8K: Per-Template Accuracy vs max_new_tokens\n(Mean ± SE across {n_runs} run(s), N={n})"
        )
        plt.legend(fontsize=8, loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gsm8k_agg_per_template_accuracy_lines.png"),
            dpi=120,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot2: {e}")
        plt.close()

    # Plot 3: Grouped bar chart with error bars (template x hparam)
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(templates))
        width = 0.8 / len(hparam_keys)
        for j, hk in enumerate(hparam_keys):
            means = acc_pt_mean[:, j]
            ses = acc_pt_se[:, j]
            ax.bar(x + j * width, means, width, yerr=ses, capsize=3, label=hk)
        ax.set_xticks(x + width * (len(hparam_keys) - 1) / 2)
        ax.set_xticklabels(templates, rotation=20)
        ax.set_ylabel("Accuracy (Mean ± SE)")
        ax.set_ylim(0, 1.0)
        ax.set_title(
            f"GSM8K: Accuracy by Emotional Prefix × max_new_tokens\n(Mean ± SE across {n_runs} run(s), N={n})"
        )
        ax.legend(title="max_new_tokens")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "gsm8k_agg_accuracy_by_template_and_mnt.png"),
            dpi=120,
        )
        plt.close()
    except Exception as e:
        print(f"Error creating plot3: {e}")
        plt.close()

    # Plot 4: Mean accuracy per hparam (from acc_ph) with error bars
    try:
        plt.figure(figsize=(7, 4))
        plt.errorbar(
            hparam_values,
            acc_ph_mean,
            yerr=acc_ph_se,
            marker="s",
            capsize=4,
            color="darkorange",
            label=f"acc_per_hparam Mean ± SE (n_runs={n_runs})",
        )
        for xv, yv in zip(hparam_values, acc_ph_mean):
            plt.text(xv, yv + 0.005, f"{yv:.3f}", ha="center", fontsize=8)
        plt.xlabel("max_new_tokens")
        plt.ylabel("Accuracy")
        plt.title(
            f"GSM8K: Aggregated Accuracy per max_new_tokens\n(from accuracy_per_hparam, N={n})"
        )
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "gsm8k_agg_acc_per_hparam.png"), dpi=120)
        plt.close()
    except Exception as e:
        print(f"Error creating plot4: {e}")
        plt.close()

    # Print summary
    try:
        print(f"=== GSM8K Aggregated Accuracy Summary (n_runs={n_runs}, N={n}) ===")
        for j, hk in enumerate(hparam_keys):
            print(f"  {hk}: mean_acc = {acc_ph_mean[j]:.4f} ± {acc_ph_se[j]:.4f} (SE)")
        print("\nPer-template best hparam (by mean):")
        for i, t in enumerate(templates):
            best_j = int(np.argmax(acc_pt_mean[i]))
            print(
                f"  {t}: best={hparam_keys[best_j]} "
                f"({acc_pt_mean[i, best_j]:.4f} ± {acc_pt_se[i, best_j]:.4f})"
            )
    except Exception as e:
        print(f"Error printing summary: {e}")
