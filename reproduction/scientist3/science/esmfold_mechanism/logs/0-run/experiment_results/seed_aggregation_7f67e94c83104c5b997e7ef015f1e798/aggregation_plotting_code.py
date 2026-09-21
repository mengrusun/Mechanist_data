import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-16_10-56-30_esmfold_mechanism_attempt_2/logs/0-run/experiment_results/experiment_d2a6deb2e3764ef2b454f407baf1967c_proc_702553/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        ed = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

ds_key = "esmfold_hairpin_panel"

# Collect aggregated per-protein data
name_order = None
gts_runs, preds_runs, plddts_runs, correct_runs = [], [], [], []
acc_runs = []
n_eval_runs, n_correct_runs = [], []

for ed in all_experiment_data:
    ds = ed.get(ds_key, {})
    per_prot = ds.get("per_protein", [])
    if not per_prot:
        continue
    names = [p["name"] for p in per_prot]
    if name_order is None:
        name_order = names
    idx = {n: i for i, n in enumerate(names)}
    order_idx = [idx[n] for n in name_order if n in idx]
    gts = np.array([int(per_prot[i]["gt_hairpin"]) for i in order_idx])
    preds = np.array([int(per_prot[i]["pred_hairpin"]) for i in order_idx])
    plddts = np.array(
        [per_prot[i].get("plddt", np.nan) for i in order_idx], dtype=float
    )
    correct = np.array([int(per_prot[i]["correct"]) for i in order_idx])
    gts_runs.append(gts)
    preds_runs.append(preds)
    plddts_runs.append(plddts)
    correct_runs.append(correct)

    val_metrics = ds.get("metrics", {}).get("val", [{}])
    vm = val_metrics[0] if val_metrics else {}
    acc_runs.append(
        vm.get(
            "hairpin_dssp_accuracy", float(correct.mean()) if len(correct) else np.nan
        )
    )
    n_eval_runs.append(vm.get("n_eval", len(order_idx)))
    n_correct_runs.append(vm.get("n_correct", int(correct.sum())))

n_runs = len(gts_runs)
if n_runs > 0 and name_order is not None:
    gts_arr = np.stack(gts_runs)  # (R, N)
    preds_arr = np.stack(preds_runs)
    plddts_arr = np.stack(plddts_runs)
    correct_arr = np.stack(correct_runs)

    gts_mean = gts_arr.mean(0)
    gts_se = gts_arr.std(0, ddof=0) / np.sqrt(n_runs)
    preds_mean = preds_arr.mean(0)
    preds_se = preds_arr.std(0, ddof=0) / np.sqrt(n_runs)
    plddts_mean = np.nanmean(plddts_arr, 0)
    plddts_se = np.nanstd(plddts_arr, 0, ddof=0) / np.sqrt(n_runs)
    correct_mean = correct_arr.mean(0)

    acc_mean = float(np.mean(acc_runs))
    acc_se = float(np.std(acc_runs, ddof=0) / np.sqrt(n_runs))
else:
    acc_mean = float("nan")
    acc_se = float("nan")

# Plot 1: GT vs mean Predicted hairpin per protein with SE bars
try:
    if n_runs > 0:
        plt.figure(figsize=(9, 4))
        x = np.arange(len(name_order))
        plt.bar(
            x - 0.2,
            gts_mean,
            0.4,
            yerr=gts_se,
            capsize=3,
            label=f"Ground Truth (mean±SE, n={n_runs})",
            color="steelblue",
        )
        plt.bar(
            x + 0.2,
            preds_mean,
            0.4,
            yerr=preds_se,
            capsize=3,
            label=f"Predicted (mean±SE, n={n_runs})",
            color="orange",
        )
        plt.xticks(x, name_order, rotation=30, ha="right")
        plt.ylabel("Hairpin probability (0-1)")
        plt.title(
            f"ESMFold Hairpin Panel: Aggregated GT vs Predicted Hairpin\n"
            f"(Left bar: Ground Truth, Right bar: Predicted) mean acc={acc_mean:.3f} ± {acc_se:.3f}"
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

# Plot 2: Mean pLDDT per protein with SE bars colored by mean correctness
try:
    if n_runs > 0:
        plt.figure(figsize=(9, 4))
        x = np.arange(len(name_order))
        colors = [(1 - c, c, 0.3) for c in correct_mean]  # red->green by correctness
        plt.bar(
            x,
            plddts_mean,
            yerr=plddts_se,
            capsize=3,
            color=colors,
            label=f"mean pLDDT ± SE (n={n_runs})",
        )
        plt.xticks(x, name_order, rotation=30, ha="right")
        plt.ylabel("mean pLDDT")
        plt.title(
            "ESMFold Hairpin Panel: Aggregated Per-Protein pLDDT\n"
            "(Color: mean correctness across runs, Green=Correct, Red=Incorrect)"
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
        for gts, preds in zip(gts_runs, preds_runs):
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

# Plot 4: Accuracy summary across runs with SE
try:
    if n_runs > 0:
        plt.figure(figsize=(4.5, 4))
        plt.bar(
            ["Accuracy", "1 - Accuracy"],
            [acc_mean, 1.0 - acc_mean],
            yerr=[acc_se, acc_se],
            capsize=5,
            color=["seagreen", "salmon"],
            label=f"mean ± SE (n={n_runs})",
        )
        plt.ylim(0, 1)
        plt.title(
            f"ESMFold Hairpin Panel: Aggregated Validation Summary\n"
            f"mean {np.mean(n_correct_runs):.1f}/{np.mean(n_eval_runs):.1f} correct across {n_runs} run(s)"
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

print(f"Aggregated hairpin_dssp_accuracy = {acc_mean:.4f} ± {acc_se:.4f} (n={n_runs})")
