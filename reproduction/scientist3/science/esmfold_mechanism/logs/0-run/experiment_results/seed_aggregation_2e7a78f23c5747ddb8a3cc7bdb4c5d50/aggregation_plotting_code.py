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
        try:
            full_path = os.path.join(
                os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
            )
            experiment_data = np.load(full_path, allow_pickle=True).item()
            all_experiment_data.append(experiment_data)
        except Exception as e:
            print(f"Could not load {experiment_data_path}: {e}")
    if not all_experiment_data:
        # fallback: try local working dir
        local_path = os.path.join(working_dir, "experiment_data.npy")
        if os.path.exists(local_path):
            all_experiment_data.append(np.load(local_path, allow_pickle=True).item())
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

ds_key = "esmfold_hairpin_panel"

# Aggregate per-protein data across runs
try:
    # Use first run to get protein names ordering
    ref_names = None
    for ed in all_experiment_data:
        per_prot = ed.get(ds_key, {}).get("per_protein", [])
        if per_prot:
            ref_names = [p["name"] for p in per_prot]
            break
    if ref_names is None:
        ref_names = []

    gts_runs, preds_runs, plddts_runs, correct_runs = [], [], [], []
    acc_runs = []
    cm_total = np.zeros((2, 2), dtype=int)

    for ed in all_experiment_data:
        ds = ed.get(ds_key, {})
        per_prot = ds.get("per_protein", [])
        name_to_p = {p["name"]: p for p in per_prot}
        gts, preds, plddts, corrs = [], [], [], []
        for n in ref_names:
            p = name_to_p.get(n, {})
            gts.append(int(p.get("gt_hairpin", 0)))
            preds.append(int(p.get("pred_hairpin", 0)))
            plddts.append(float(p.get("plddt", np.nan)))
            corrs.append(int(p.get("correct", 0)))
            g = int(p.get("gt_hairpin", 0))
            pr = int(p.get("pred_hairpin", 0))
            cm_total[g, pr] += 1
        gts_runs.append(gts)
        preds_runs.append(preds)
        plddts_runs.append(plddts)
        correct_runs.append(corrs)
        val_metrics = ds.get("metrics", {}).get("val", [{}])
        acc = (
            val_metrics[0].get("hairpin_dssp_accuracy", np.nan)
            if val_metrics
            else np.nan
        )
        acc_runs.append(acc)

    gts_arr = np.array(gts_runs, dtype=float) if gts_runs else np.zeros((0, 0))
    preds_arr = np.array(preds_runs, dtype=float) if preds_runs else np.zeros((0, 0))
    plddts_arr = np.array(plddts_runs, dtype=float) if plddts_runs else np.zeros((0, 0))
    correct_arr = (
        np.array(correct_runs, dtype=float) if correct_runs else np.zeros((0, 0))
    )
    acc_arr = np.array(acc_runs, dtype=float)
    n_runs = len(all_experiment_data)
except Exception as e:
    print(f"Error aggregating data: {e}")
    ref_names, gts_arr, preds_arr, plddts_arr, correct_arr, acc_arr = (
        [],
        None,
        None,
        None,
        None,
        None,
    )
    cm_total = np.zeros((2, 2), dtype=int)
    n_runs = 0

# Plot 1: Mean GT vs Predicted with SEM
try:
    if ref_names and gts_arr.size > 0:
        plt.figure(figsize=(9, 4))
        x = np.arange(len(ref_names))
        gt_mean = gts_arr.mean(axis=0)
        gt_sem = (
            gts_arr.std(axis=0, ddof=1) / np.sqrt(n_runs)
            if n_runs > 1
            else np.zeros_like(gt_mean)
        )
        pred_mean = preds_arr.mean(axis=0)
        pred_sem = (
            preds_arr.std(axis=0, ddof=1) / np.sqrt(n_runs)
            if n_runs > 1
            else np.zeros_like(pred_mean)
        )
        plt.bar(
            x - 0.2,
            gt_mean,
            0.4,
            yerr=gt_sem,
            label="Ground Truth (mean±SEM)",
            color="steelblue",
            capsize=3,
        )
        plt.bar(
            x + 0.2,
            pred_mean,
            0.4,
            yerr=pred_sem,
            label="Predicted (mean±SEM)",
            color="orange",
            capsize=3,
        )
        plt.xticks(x, ref_names, rotation=30, ha="right")
        plt.ylabel("Hairpin (0/1)")
        plt.title(
            f"ESMFold Hairpin Panel: Aggregated GT vs Predicted across {n_runs} runs\n"
            f"(Left bar: Ground Truth, Right bar: Predicted; error bars = SEM)"
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
    if ref_names and plddts_arr.size > 0:
        plt.figure(figsize=(9, 4))
        x = np.arange(len(ref_names))
        plddt_mean = np.nanmean(plddts_arr, axis=0)
        if n_runs > 1:
            plddt_sem = np.nanstd(plddts_arr, axis=0, ddof=1) / np.sqrt(n_runs)
        else:
            plddt_sem = np.zeros_like(plddt_mean)
        corr_mean = correct_arr.mean(axis=0)
        colors = ["green" if c >= 0.5 else "red" for c in corr_mean]
        plt.bar(
            x,
            plddt_mean,
            yerr=plddt_sem,
            color=colors,
            capsize=3,
            label="mean pLDDT ± SEM",
        )
        plt.xticks(x, ref_names, rotation=30, ha="right")
        plt.ylabel("mean pLDDT")
        plt.title(
            f"ESMFold Hairpin Panel: Aggregated per-Protein pLDDT across {n_runs} runs\n"
            "(Green: majority correct, Red: majority incorrect; error bars = SEM)"
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

# Plot 3: Aggregate confusion matrix
try:
    plt.figure(figsize=(4.5, 4.5))
    plt.imshow(cm_total, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                str(cm_total[i, j]),
                ha="center",
                va="center",
                color="white" if cm_total[i, j] > cm_total.max() / 2 else "black",
                fontsize=14,
            )
    plt.xticks([0, 1], ["Pred: No", "Pred: Yes"])
    plt.yticks([0, 1], ["GT: No", "GT: Yes"])
    plt.title(
        f"ESMFold Hairpin Panel: Aggregate Confusion Matrix\n"
        f"(Summed across {n_runs} runs; Rows: GT, Cols: Predicted)"
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

# Plot 4: Accuracy across runs with mean ± SEM
try:
    if acc_arr is not None and acc_arr.size > 0:
        plt.figure(figsize=(6, 4))
        run_ids = np.arange(1, len(acc_arr) + 1)
        plt.bar(run_ids, acc_arr, color="lightsteelblue", label="per-run accuracy")
        mean_acc = np.nanmean(acc_arr)
        sem_acc = (
            np.nanstd(acc_arr, ddof=1) / np.sqrt(len(acc_arr))
            if len(acc_arr) > 1
            else 0.0
        )
        plt.axhline(
            mean_acc,
            color="darkred",
            linestyle="--",
            label=f"mean = {mean_acc:.3f} ± {sem_acc:.3f} (SEM)",
        )
        plt.fill_between(
            [0.5, len(acc_arr) + 0.5],
            mean_acc - sem_acc,
            mean_acc + sem_acc,
            color="darkred",
            alpha=0.15,
            label="±1 SEM",
        )
        plt.xticks(run_ids)
        plt.xlabel("Run")
        plt.ylabel("Hairpin DSSP Accuracy")
        plt.ylim(0, 1.05)
        plt.title(
            f"ESMFold Hairpin Panel: Accuracy across {n_runs} runs\n(Mean ± SEM overlaid)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "esmfold_hairpin_panel_agg_accuracy.png"), dpi=120
        )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Print aggregate metric
try:
    if acc_arr is not None and acc_arr.size > 0:
        mean_acc = np.nanmean(acc_arr)
        sem_acc = (
            np.nanstd(acc_arr, ddof=1) / np.sqrt(len(acc_arr))
            if len(acc_arr) > 1
            else 0.0
        )
        print(
            f"Aggregated hairpin_dssp_accuracy across {n_runs} runs: "
            f"mean = {mean_acc:.4f}, SEM = {sem_acc:.4f}"
        )
        print(f"Per-run accuracies: {acc_arr}")
    else:
        print("No accuracy data available for aggregation.")
except Exception as e:
    print(f"Error printing metrics: {e}")
