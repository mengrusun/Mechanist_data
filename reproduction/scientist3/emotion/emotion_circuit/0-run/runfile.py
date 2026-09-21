import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-14_16-45-26_emotion_circuit_attempt_8/logs/0-run/experiment_results/experiment_035ddc0b26324fa097b5b671184e40ee_proc_2301608/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        experiment_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(experiment_data)
    print(f"Loaded {len(all_experiment_data)} experiment(s).")
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


# Collect data across runs
runs_sev = []
for exp in all_experiment_data:
    sev = exp.get("strength", {}).get("SEV", {})
    if sev:
        runs_sev.append(sev)

if len(runs_sev) == 0:
    print("No SEV data available.")
else:
    # Use strengths from first run (assume same across runs)
    strengths = runs_sev[0].get("strength_values", [])

    # Build arrays: accuracy per run per strength
    acc_matrix = []
    for sev in runs_sev:
        per_str_res = sev.get("per_strength_results", {})
        try:
            row = [per_str_res[str(s)]["accuracy"] for s in strengths]
            acc_matrix.append(row)
        except Exception as e:
            print(f"Skipping run due to missing strength data: {e}")
    acc_matrix = np.array(acc_matrix)  # shape (n_runs, n_strengths)

    n_runs = acc_matrix.shape[0]
    mean_acc = acc_matrix.mean(axis=0)
    sem_acc = (
        acc_matrix.std(axis=0, ddof=1) / np.sqrt(n_runs)
        if n_runs > 1
        else np.zeros_like(mean_acc)
    )

    # Plot 1: Aggregated accuracy vs strength with SEM
    try:
        plt.figure(figsize=(8, 5))
        plt.errorbar(
            strengths,
            mean_acc,
            yerr=sem_acc,
            marker="o",
            color="steelblue",
            ecolor="lightsteelblue",
            capsize=4,
            label=f"Mean accuracy (n={n_runs})",
        )
        plt.fill_between(
            strengths,
            mean_acc - sem_acc,
            mean_acc + sem_acc,
            color="steelblue",
            alpha=0.2,
            label="± SEM",
        )
        plt.axhline(1 / 7, color="r", linestyle="--", label="Random (1/7)")
        plt.xlabel("Steering strength")
        plt.ylabel("Accuracy")
        plt.title(
            "SEV Dataset: Aggregated Steering Accuracy vs Strength\n(Mean ± SEM across runs)"
        )
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SEV_agg_accuracy_vs_strength.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot1: {e}")
        plt.close()

    # Plot 2: Aggregated loss (1-acc) vs strength
    try:
        loss_matrix = 1 - acc_matrix
        mean_loss = loss_matrix.mean(axis=0)
        sem_loss = (
            loss_matrix.std(axis=0, ddof=1) / np.sqrt(n_runs)
            if n_runs > 1
            else np.zeros_like(mean_loss)
        )
        plt.figure(figsize=(8, 5))
        plt.errorbar(
            strengths,
            mean_loss,
            yerr=sem_loss,
            marker="s",
            color="darkorange",
            ecolor="navajowhite",
            capsize=4,
            label=f"Mean loss (n={n_runs})",
        )
        plt.fill_between(
            strengths,
            mean_loss - sem_loss,
            mean_loss + sem_loss,
            color="darkorange",
            alpha=0.2,
            label="± SEM",
        )
        plt.xlabel("Steering strength")
        plt.ylabel("Loss (1 - accuracy)")
        plt.title(
            "SEV Dataset: Aggregated Validation Loss vs Steering Strength\n(Mean ± SEM across runs, lower is better)"
        )
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SEV_agg_loss_vs_strength.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot2: {e}")
        plt.close()

    # Plot 3: Aggregated per-emotion accuracy at best strength across runs
    try:
        # Collect per-emotion accuracy at each run's best strength
        emo_keys = None
        per_emo_matrix = []
        for sev in runs_sev:
            best_per_emo = sev.get("per_emotion_accuracy", {})
            if not best_per_emo:
                continue
            if emo_keys is None:
                emo_keys = list(best_per_emo.keys())
            vals = [best_per_emo.get(e, np.nan) for e in emo_keys]
            per_emo_matrix.append(vals)
        per_emo_matrix = np.array(per_emo_matrix, dtype=float)
        mean_emo = np.nanmean(per_emo_matrix, axis=0)
        sem_emo = (
            np.nanstd(per_emo_matrix, axis=0, ddof=1) / np.sqrt(per_emo_matrix.shape[0])
            if per_emo_matrix.shape[0] > 1
            else np.zeros_like(mean_emo)
        )

        plt.figure(figsize=(9, 5))
        x = np.arange(len(emo_keys))
        plt.bar(
            x,
            mean_emo,
            yerr=sem_emo,
            capsize=4,
            color="steelblue",
            ecolor="black",
            label=f"Mean ± SEM (n={per_emo_matrix.shape[0]})",
        )
        plt.axhline(1 / 7, color="r", linestyle="--", label="Random (1/7)")
        plt.xticks(x, emo_keys, rotation=30)
        plt.ylabel("Accuracy")
        plt.title(
            "SEV Dataset: Aggregated Per-Emotion Accuracy at Best Strength\n(Mean ± SEM across runs)"
        )
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SEV_agg_per_emotion_best_strength.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot3: {e}")
        plt.close()

    # Plot 4: Aggregated heatmap of per-emotion accuracy vs strength (mean across runs)
    try:
        # collect per-emotion matrix per run: shape (n_runs, n_strengths, n_emotions)
        emo_keys2 = None
        cubes = []
        for sev in runs_sev:
            per_str_res = sev.get("per_strength_results", {})
            try:
                first_key = str(strengths[0])
                if emo_keys2 is None:
                    emo_keys2 = list(
                        per_str_res[first_key]["per_emotion_accuracy"].keys()
                    )
                mat = np.array(
                    [
                        [
                            per_str_res[str(s)]["per_emotion_accuracy"][e]
                            for e in emo_keys2
                        ]
                        for s in strengths
                    ]
                )
                cubes.append(mat)
            except Exception:
                continue
        cube = np.array(cubes)  # (n_runs, n_strengths, n_emotions)
        mean_mat = cube.mean(axis=0)

        plt.figure(figsize=(9, 5))
        im = plt.imshow(mean_mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        plt.colorbar(im, label="Mean Accuracy")
        plt.xticks(range(len(emo_keys2)), emo_keys2, rotation=30)
        plt.yticks(range(len(strengths)), strengths)
        plt.xlabel("Emotion")
        plt.ylabel("Steering strength")
        plt.title(
            f"SEV Dataset: Aggregated Per-Emotion Accuracy Heatmap\n(Mean across n={cube.shape[0]} runs)"
        )
        for i in range(mean_mat.shape[0]):
            for j in range(mean_mat.shape[1]):
                plt.text(
                    j,
                    i,
                    f"{mean_mat[i,j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if mean_mat[i, j] < 0.5 else "black",
                    fontsize=8,
                )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "SEV_agg_per_emotion_heatmap.png"))
        plt.close()
    except Exception as e:
        print(f"Error creating plot4: {e}")
        plt.close()

    # Print summary metrics
    try:
        print("\n=== Aggregated Summary (SEV) ===")
        print(f"Number of runs: {n_runs}")
        for i, s in enumerate(strengths):
            print(f"  strength={s}: mean_acc={mean_acc[i]:.4f} ± SEM {sem_acc[i]:.4f}")
        best_idx = int(np.argmax(mean_acc))
        print(
            f"Best mean strength: {strengths[best_idx]} with mean_acc={mean_acc[best_idx]:.4f}"
        )
    except Exception as e:
        print(f"Error printing summary: {e}")
