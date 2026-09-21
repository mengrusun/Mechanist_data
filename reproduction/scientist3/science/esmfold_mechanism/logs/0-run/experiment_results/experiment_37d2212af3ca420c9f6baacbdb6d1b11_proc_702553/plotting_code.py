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
    experiment_data = {}

ed = experiment_data.get("num_recycles", {}).get("esmfold_hairpin_panel", {})
summary = ed.get("per_setting_summary", [])
nrs = [s["num_recycles"] for s in summary]
accs = [s["accuracy"] for s in summary]
plddts = [s["mean_plddt"] for s in summary]
per_protein = ed.get("per_protein", {})
predictions = ed.get("predictions", {})
gt = ed.get("ground_truth", [])

# Plot 1: accuracy + mean pLDDT vs num_recycles
try:
    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.bar([str(n) for n in nrs], accs, color="steelblue", alpha=0.7)
    ax1.set_ylabel("hairpin_dssp_accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("num_recycles")
    ax2 = ax1.twinx()
    ax2.plot([str(n) for n in nrs], plddts, "ro-")
    ax2.set_ylabel("mean pLDDT")
    plt.title(
        "ESMFold Hairpin Panel: Accuracy (bars) & Mean pLDDT (line) vs num_recycles"
    )
    fig.tight_layout()
    plt.savefig(
        os.path.join(
            working_dir, "esmfold_hairpin_panel_accuracy_plddt_vs_num_recycles.png"
        ),
        dpi=120,
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: per-protein hairpin predictions with GT
try:
    keys = sorted(per_protein.keys(), key=lambda k: int(k.split("=")[1]))
    if keys:
        names = [p["name"] for p in per_protein[keys[0]]]
        x = np.arange(len(names))
        width = 0.8 / (len(keys) + 1)
        plt.figure(figsize=(11, 4))
        gt_vals = (
            gt
            if len(gt) == len(names)
            else [int(p["gt_hairpin"]) for p in per_protein[keys[0]]]
        )
        plt.bar(x - 0.4 + width / 2, gt_vals, width, label="Ground Truth", color="gray")
        for i, k in enumerate(keys):
            preds = [int(p["pred_hairpin"]) for p in per_protein[k]]
            plt.bar(x - 0.4 + width * (i + 1) + width / 2, preds, width, label=k)
        plt.xticks(x, names, rotation=30, ha="right")
        plt.ylabel("hairpin (0/1)")
        plt.title(
            "ESMFold Hairpin Panel: Per-protein Hairpin Predictions\nLeft-to-right groups: Ground Truth then each num_recycles setting"
        )
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "esmfold_hairpin_panel_per_protein_predictions.png"
            ),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: per-protein pLDDT heatmap across settings
try:
    keys = sorted(per_protein.keys(), key=lambda k: int(k.split("=")[1]))
    if keys:
        names = [p["name"] for p in per_protein[keys[0]]]
        mat = np.array(
            [[p["plddt"] for p in per_protein[k]] for k in keys], dtype=float
        )
        plt.figure(figsize=(10, 3.5))
        im = plt.imshow(mat, aspect="auto", cmap="viridis")
        plt.colorbar(im, label="pLDDT")
        plt.yticks(range(len(keys)), keys)
        plt.xticks(range(len(names)), names, rotation=30, ha="right")
        plt.title("ESMFold Hairpin Panel: Per-protein Mean pLDDT across num_recycles")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "esmfold_hairpin_panel_plddt_heatmap.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: validation "loss" (1-accuracy) curve
try:
    losses = ed.get("losses", {}).get("val", [])
    if losses:
        xs = [l["num_recycles"] for l in losses]
        ys = [l["loss"] for l in losses]
        plt.figure(figsize=(6, 4))
        plt.plot(xs, ys, "bo-")
        plt.xlabel("num_recycles")
        plt.ylabel("validation loss (1 - accuracy)")
        plt.title("ESMFold Hairpin Panel: Validation Loss vs num_recycles")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "esmfold_hairpin_panel_val_loss_curve.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: correctness matrix
try:
    keys = sorted(per_protein.keys(), key=lambda k: int(k.split("=")[1]))
    if keys:
        names = [p["name"] for p in per_protein[keys[0]]]
        mat = np.array([[int(p["correct"]) for p in per_protein[k]] for k in keys])
        plt.figure(figsize=(10, 3.5))
        im = plt.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        plt.colorbar(im, label="correct (1) / wrong (0)")
        plt.yticks(range(len(keys)), keys)
        plt.xticks(range(len(names)), names, rotation=30, ha="right")
        plt.title("ESMFold Hairpin Panel: Per-protein Correctness across num_recycles")
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "esmfold_hairpin_panel_correctness_matrix.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print metrics summary
try:
    print("num_recycles sweep summary:")
    for s in summary:
        print(
            f"  nr={s['num_recycles']}: acc={s['accuracy']:.4f} ({s['n_correct']}/{s['n_eval']}) mean_plddt={s['mean_plddt']:.2f}"
        )
except Exception as e:
    print(f"Error printing summary: {e}")
