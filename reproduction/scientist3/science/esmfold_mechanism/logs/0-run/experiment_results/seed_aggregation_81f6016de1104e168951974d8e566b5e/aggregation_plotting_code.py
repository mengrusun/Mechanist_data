import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/run_1/working/experiment_data.npy",
        "experiments/run_2/working/experiment_data.npy",
        "experiments/run_3/working/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        try:
            ed = np.load(full_path, allow_pickle=True).item()
            all_experiment_data.append(ed)
        except Exception as e:
            print(f"Could not load {full_path}: {e}")
    if not all_experiment_data:
        # fallback to local file
        ed = np.load(
            os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
        ).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

ds_key = "esmfold_hairpin_panel"

# Aggregate across runs
names = None
gts_list, preds_list, plddts_list, correct_list = [], [], [], []
acc_list = []
n_eval_list, n_correct_list = [], []

for ed in all_experiment_data:
    ds = ed.get(ds_key, {})
    per_prot = ds.get("per_protein", [])
    if not per_prot:
        continue
    cur_names = [p["name"] for p in per_prot]
    if names is None:
        names = cur_names
    # align by name
    name_to_idx = {n: i for i, n in enumerate(cur_names)}
    order = [name_to_idx[n] for n in names if n in name_to_idx]
    if len(order) != len(names):
        continue
    gts_list.append(np.array([int(per_prot[i]["gt_hairpin"]) for i in order]))
    preds_list.append(np.array([int(per_prot[i]["pred_hairpin"]) for i in order]))
    plddts_list.append(np.array([per_prot[i].get("plddt", np.nan) for i in order]))
    correct_list.append(np.array([int(per_prot[i]["correct"]) for i in order]))
    val_metrics = ds.get("metrics", {}).get("val", [{}])
    if val_metrics:
        acc_list.append(val_metrics[0].get("hairpin_dssp_accuracy", np.nan))
        n_eval_list.append(val_metrics[0].get("n_eval", len(per_prot)))
        n_correct_list.append(
            val_metrics[0].get("n_correct", int(correct_list[-1].sum()))
        )

n_runs = len(gts_list)
print(f"Aggregating across {n_runs} runs")

if n_runs > 0:
    gts_arr = np.stack(gts_list)  # (runs, proteins)
    preds_arr = np.stack(preds_list)
    plddts_arr = np.stack(plddts_list)
    correct_arr = np.stack(correct_list)

    preds_mean = preds_arr.mean(axis=0)
    preds_sem = (
        preds_arr.std(axis=0, ddof=1) / np.sqrt(n_runs)
        if n_runs > 1
        else np.zeros_like(preds_mean)
    )
    plddt_mean = np.nanmean(plddts_arr, axis=0)
    plddt_sem = (
        (np.nanstd(plddts_arr, axis=0, ddof=1) / np.sqrt(n_runs))
        if n_runs > 1
        else np.zeros_like(plddt_mean)
    )
    gt_mean = gts_arr.mean(axis=0)  # should be constant across runs

    acc_arr = np.array(acc_list, dtype=float)
    acc_mean = np.nanmean(acc_arr)
    acc_sem = (np.nanstd(acc_arr, ddof=1) / np.sqrt(n_runs)) if n_runs > 1 else 0.0

# Plot 1: GT vs mean predicted hairpin with SEM
try:
    if n_runs > 0:
        plt.figure(figsize=(10, 4))
        x = np.arange(len(names))
        plt.bar(x - 0.2, gt_mean, 0.4, label="Ground Truth", color="steelblue")
        plt.bar(
            x + 0.2,
            preds_mean,
            0.4,
            yerr=preds_sem,
            capsize=3,
            label=f"Predicted mean ± SEM (n={n_runs})",
            color="orange",
        )
        plt.xticks(x, names, rotation=30, ha="right")
        plt.ylabel("Hairpin (0/1)")
        plt.title(
            f"ESMFold Hairpin Panel: GT vs Aggregated Predicted Hairpin\n"
            f"(Left: Ground Truth, Right: Mean Prediction ± SEM) acc={acc_mean:.3f}±{acc_sem:.3f}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "esmfold_hairpin_panel_agg_gt_vs_pred.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Mean pLDDT per protein with SEM
try:
    if n_runs > 0:
        plt.figure(figsize=(10, 4))
        x = np.arange(len(names))
        mean_correct = correct_arr.mean(axis=0)
        colors = [plt.cm.RdYlGn(c) for c in mean_correct]
        plt.bar(
            x,
            plddt_mean,
            yerr=plddt_sem,
            capsize=3,
            color=colors,
            label=f"Mean pLDDT ± SEM (n={n_runs})",
        )
        plt.xticks(x, names, rotation=30, ha="right")
        plt.ylabel("mean pLDDT")
        plt.title(
            "ESMFold Hairpin Panel: Aggregated Per-Protein pLDDT\n"
            "(Bar color: fraction of runs with correct hairpin call, Green=1, Red=0)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "esmfold_hairpin_panel_agg_plddt.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Aggregated confusion matrix (summed across runs)
try:
    if n_runs > 0:
        cm = np.zeros((2, 2), dtype=int)
        for gts, preds in zip(gts_list, preds_list):
            for g, p in zip(gts, preds):
                cm[g, p] += 1
        plt.figure(figsize=(4.5, 4.5))
        plt.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                plt.text(
                    j,
                    i,
                    str(cm[i, j]),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=14,
                )
        plt.xticks([0, 1], ["Pred: No", "Pred: Yes"])
        plt.yticks([0, 1], ["GT: No", "GT: Yes"])
        plt.title(
            f"ESMFold Hairpin Panel: Aggregated Confusion Matrix (n={n_runs} runs)\n"
            "(Rows: Ground Truth, Cols: Predicted)"
        )
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "esmfold_hairpin_panel_agg_confusion_matrix.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Accuracy summary with mean ± SEM
try:
    if n_runs > 0:
        plt.figure(figsize=(5, 4))
        labels = ["Accuracy", "Loss (1-acc)"]
        means = [acc_mean, 1.0 - acc_mean]
        sems = [acc_sem, acc_sem]
        plt.bar(
            labels,
            means,
            yerr=sems,
            capsize=5,
            color=["seagreen", "salmon"],
            label=f"Mean ± SEM (n={n_runs})",
        )
        # scatter individual runs
        for a in acc_arr:
            plt.scatter(0, a, color="black", zorder=3, s=15)
            plt.scatter(1, 1 - a, color="black", zorder=3, s=15)
        plt.ylim(0, 1)
        plt.title(
            f"ESMFold Hairpin Panel: Aggregated Validation Summary\n"
            f"Mean acc = {acc_mean:.3f} ± {acc_sem:.3f} (SEM, n={n_runs})"
        )
        plt.ylabel("value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "esmfold_hairpin_panel_agg_accuracy_summary.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

if n_runs > 0:
    print(
        f"hairpin_dssp_accuracy (mean ± SEM) = {acc_mean:.4f} ± {acc_sem:.4f} over {n_runs} runs"
    )
    print(f"Individual run accuracies: {acc_arr}")
else:
    print("No runs available for aggregation.")
