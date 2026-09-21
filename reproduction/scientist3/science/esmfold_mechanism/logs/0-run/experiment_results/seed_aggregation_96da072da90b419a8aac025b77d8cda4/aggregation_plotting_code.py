import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-16_10-56-30_esmfold_mechanism_attempt_2/logs/0-run/experiment_results/experiment_89d32df76e2d4c94945265bf7c5be5b4_proc_631077/experiment_data.npy",
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

# Collect per-run per-protein arrays keyed by name
names_ref = None
gts_ref = None
preds_runs = []  # list of arrays (per protein)
plddts_runs = []
correct_runs = []
acc_runs = []
n_eval_list = []
n_correct_list = []

for ed in all_experiment_data:
    ds = ed.get(ds_key, {})
    per_prot = ds.get("per_protein", [])
    if not per_prot:
        continue
    names = [p["name"] for p in per_prot]
    gts = np.array([int(p["gt_hairpin"]) for p in per_prot])
    preds = np.array([int(p["pred_hairpin"]) for p in per_prot])
    plddts = np.array([p.get("plddt", np.nan) for p in per_prot], dtype=float)
    correct = np.array([int(p["correct"]) for p in per_prot])
    if names_ref is None:
        names_ref = names
        gts_ref = gts
    preds_runs.append(preds)
    plddts_runs.append(plddts)
    correct_runs.append(correct)

    val_metrics = ds.get("metrics", {}).get("val", [{}])
    vm = val_metrics[0] if val_metrics else {}
    acc = vm.get(
        "hairpin_dssp_accuracy", float(correct.mean()) if len(correct) else float("nan")
    )
    acc_runs.append(acc)
    n_eval_list.append(vm.get("n_eval", len(per_prot)))
    n_correct_list.append(vm.get("n_correct", int(correct.sum())))

n_runs = len(preds_runs)
print(f"Number of runs aggregated: {n_runs}")

if n_runs > 0:
    preds_arr = np.stack(preds_runs, axis=0)  # (runs, proteins)
    plddts_arr = np.stack(plddts_runs, axis=0)
    correct_arr = np.stack(correct_runs, axis=0)

    preds_mean = preds_arr.mean(axis=0)
    preds_se = (
        preds_arr.std(axis=0, ddof=0) / np.sqrt(n_runs)
        if n_runs > 1
        else np.zeros_like(preds_mean)
    )

    plddts_mean = np.nanmean(plddts_arr, axis=0)
    if n_runs > 1:
        plddts_se = np.nanstd(plddts_arr, axis=0, ddof=0) / np.sqrt(n_runs)
    else:
        plddts_se = np.zeros_like(plddts_mean)

    acc_arr = np.array(acc_runs, dtype=float)
    acc_mean = float(np.nanmean(acc_arr))
    acc_se = float(np.nanstd(acc_arr, ddof=0) / np.sqrt(n_runs)) if n_runs > 1 else 0.0
    print(
        f"Aggregated hairpin_dssp_accuracy: mean={acc_mean:.4f}  SE={acc_se:.4f}  (n_runs={n_runs})"
    )
else:
    preds_mean = preds_se = plddts_mean = plddts_se = np.array([])
    acc_mean, acc_se = float("nan"), 0.0

# Plot 1: GT vs Mean Predicted with SE bars
try:
    if n_runs > 0:
        plt.figure(figsize=(10, 4))
        x = np.arange(len(names_ref))
        plt.bar(x - 0.2, gts_ref, 0.4, label="Ground Truth", color="steelblue")
        plt.bar(
            x + 0.2,
            preds_mean,
            0.4,
            yerr=preds_se,
            capsize=3,
            label=f"Predicted (mean ± SE, n={n_runs})",
            color="orange",
            error_kw=dict(ecolor="black", lw=1),
        )
        plt.xticks(x, names_ref, rotation=30, ha="right")
        plt.ylabel("Hairpin call (0/1)")
        plt.title(
            "ESMFold Hairpin Panel: GT vs Aggregated Predicted Hairpin\n"
            f"Left bar: Ground Truth, Right bar: Mean Predicted ± SE (acc={acc_mean:.3f} ± {acc_se:.3f})"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "esmfold_hairpin_panel_gt_vs_pred_agg.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Mean pLDDT per protein with SE bars, colored by mean correctness
try:
    if n_runs > 0:
        plt.figure(figsize=(10, 4))
        x = np.arange(len(names_ref))
        mean_correct = correct_arr.mean(axis=0)
        colors = ["green" if mc >= 0.5 else "red" for mc in mean_correct]
        plt.bar(
            x,
            plddts_mean,
            yerr=plddts_se,
            capsize=3,
            color=colors,
            error_kw=dict(ecolor="black", lw=1),
            label=f"mean pLDDT ± SE (n={n_runs})",
        )
        plt.xticks(x, names_ref, rotation=30, ha="right")
        plt.ylabel("mean pLDDT")
        plt.title(
            "ESMFold Hairpin Panel: Aggregated Per-Protein pLDDT\n"
            "Green: mostly correct hairpin call, Red: mostly incorrect (bars = mean ± SE across runs)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "esmfold_hairpin_panel_plddt_agg.png"), dpi=120
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Aggregated confusion matrix (summed across runs)
try:
    if n_runs > 0:
        cm = np.zeros((2, 2), dtype=int)
        for r in range(n_runs):
            for g, p in zip(gts_ref, preds_runs[r]):
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
            "ESMFold Hairpin Panel: Aggregated Confusion Matrix\n"
            f"Rows: Ground Truth, Cols: Predicted (summed over {n_runs} runs)"
        )
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "esmfold_hairpin_panel_confusion_matrix_agg.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Accuracy summary with SE across runs
try:
    if n_runs > 0:
        plt.figure(figsize=(5, 4))
        vals = [acc_mean, 1.0 - acc_mean]
        errs = [acc_se, acc_se]
        plt.bar(
            ["Accuracy", "Loss (1-acc)"],
            vals,
            yerr=errs,
            capsize=5,
            color=["seagreen", "salmon"],
            error_kw=dict(ecolor="black", lw=1),
            label=f"mean ± SE (n_runs={n_runs})",
        )
        plt.ylim(0, 1)
        plt.ylabel("value")
        plt.title(
            "ESMFold Hairpin Panel: Aggregated Validation Summary\n"
            f"Mean accuracy = {acc_mean:.3f} ± {acc_se:.3f} SE across {n_runs} run(s)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "esmfold_hairpin_panel_accuracy_summary_agg.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

print(
    f"Aggregated hairpin_dssp_accuracy = {acc_mean:.4f} (SE={acc_se:.4f}, n_runs={n_runs})"
)
