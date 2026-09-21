import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_06-08-42_propositional_logic_circuit_attempt_1/logs/0-run/experiment_results/experiment_b86fb5c0fd5c43d7928a18fb9413bb37_proc_3265112/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        full_path = os.path.join(
            os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path
        )
        experiment_data = np.load(full_path, allow_pickle=True).item()
        all_experiment_data.append(experiment_data)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

# Extract per-run arrays
runs = []
for ed in all_experiment_data:
    data = ed.get("prop_logic_synth", {})
    preds = np.array(data.get("predictions", []))
    labels = np.array(data.get("ground_truth", []))
    tl = np.array(data.get("true_logits", []))
    fl = np.array(data.get("false_logits", []))
    val_metrics = data.get("metrics", {}).get("val", [])
    val_losses = data.get("losses", {}).get("val", [])
    acc = (
        val_metrics[0]["task_accuracy"]
        if val_metrics
        else (float((preds == labels).mean()) if len(preds) else 0.0)
    )
    vloss = val_losses[0]["val_loss"] if val_losses else None
    runs.append(dict(preds=preds, labels=labels, tl=tl, fl=fl, acc=acc, vloss=vloss))

n_runs = len(runs)
print(f"Loaded {n_runs} runs")


def sem(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return 0.0
    return np.std(x, ddof=1) / np.sqrt(len(x))


# Plot 1: Aggregated metrics (accuracy and val loss) with SEM error bars
try:
    accs = [r["acc"] for r in runs]
    vlosses = [r["vloss"] for r in runs if r["vloss"] is not None]
    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    acc_mean, acc_sem = np.mean(accs), sem(accs)
    ax[0].bar(
        ["accuracy"],
        [acc_mean],
        yerr=[acc_sem],
        color="steelblue",
        capsize=8,
        label=f"Mean ± SEM (n={n_runs})",
    )
    ax[0].set_ylim(0, 1)
    ax[0].set_title("Task Accuracy")
    ax[0].text(0, acc_mean + 0.02, f"{acc_mean:.3f}±{acc_sem:.3f}", ha="center")
    ax[0].legend()
    if vlosses:
        vl_mean, vl_sem = np.mean(vlosses), sem(vlosses)
        ax[1].bar(
            ["val_loss"],
            [vl_mean],
            yerr=[vl_sem],
            color="indianred",
            capsize=8,
            label=f"Mean ± SEM (n={len(vlosses)})",
        )
        ax[1].set_title("Validation Loss")
        ax[1].text(0, vl_mean * 1.02, f"{vl_mean:.3f}±{vl_sem:.3f}", ha="center")
        ax[1].legend()
    fig.suptitle(
        "Prop Logic Synth: Aggregated Evaluation Metrics\n(Left: Accuracy, Right: Val Loss; Mean ± SEM)"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "prop_logic_synth_aggregated_metrics.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Aggregated logit gap histogram (pooled across runs)
try:
    plt.figure(figsize=(7, 4))
    all_diffs_true, all_diffs_false = [], []
    for r in runs:
        diffs = r["tl"] - r["fl"]
        lab = r["labels"].astype(int)
        all_diffs_true.append(diffs[lab == 1])
        all_diffs_false.append(diffs[lab == 0])
    dt = np.concatenate(all_diffs_true) if all_diffs_true else np.array([])
    df = np.concatenate(all_diffs_false) if all_diffs_false else np.array([])
    plt.hist(
        dt, bins=25, alpha=0.6, label=f"True label (n={len(dt)}, μ={dt.mean():.2f})"
    )
    plt.hist(
        df, bins=25, alpha=0.6, label=f"False label (n={len(df)}, μ={df.mean():.2f})"
    )
    plt.axvline(0, color="k", linestyle="--")
    plt.axvline(dt.mean(), color="C0", linestyle=":", label="Mean True")
    plt.axvline(df.mean(), color="C1", linestyle=":", label="Mean False")
    plt.xlabel("True_logit - False_logit")
    plt.ylabel("Count")
    plt.legend()
    plt.title(
        f"Prop Logic Synth: Aggregated Logit Gap Distribution\n(Pooled across {n_runs} runs)"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "prop_logic_synth_aggregated_logit_gap.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Aggregated confusion matrix (mean per-run counts with SEM annotation)
try:
    cms = []
    for r in runs:
        cm = np.zeros((2, 2), dtype=float)
        for l, p in zip(r["labels"].astype(int), r["preds"].astype(int)):
            cm[l, p] += 1
        cms.append(cm)
    if cms:
        cms_arr = np.stack(cms, axis=0)
        cm_mean = cms_arr.mean(axis=0)
        cm_sem = np.array([[sem(cms_arr[:, i, j]) for j in range(2)] for i in range(2)])
        plt.figure(figsize=(5.5, 4.5))
        plt.imshow(cm_mean, cmap="Blues")
        plt.colorbar()
        plt.xticks([0, 1], ["Pred False", "Pred True"])
        plt.yticks([0, 1], ["True False", "True True"])
        for i in range(2):
            for j in range(2):
                txt = f"{cm_mean[i,j]:.1f}\n±{cm_sem[i,j]:.1f}"
                plt.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    color="white" if cm_mean[i, j] > cm_mean.max() / 2 else "black",
                )
        plt.title(
            f"Prop Logic Synth: Aggregated Confusion Matrix\n(Mean ± SEM counts across {n_runs} runs)"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(
                working_dir, "prop_logic_synth_aggregated_confusion_matrix.png"
            ),
            dpi=100,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Per-run accuracy with mean ± SEM line
try:
    plt.figure(figsize=(7, 4))
    accs = [r["acc"] for r in runs]
    xs = np.arange(1, n_runs + 1)
    plt.bar(xs, accs, color="lightsteelblue", label="Per-run accuracy")
    m, s = np.mean(accs), sem(accs)
    plt.axhline(m, color="red", linestyle="--", label=f"Mean = {m:.3f}")
    plt.fill_between(
        [0.5, n_runs + 0.5],
        m - s,
        m + s,
        color="red",
        alpha=0.2,
        label=f"± SEM ({s:.3f})",
    )
    plt.xticks(xs)
    plt.xlim(0.5, n_runs + 0.5)
    plt.ylim(0, 1)
    plt.xlabel("Run index")
    plt.ylabel("Task accuracy")
    plt.legend()
    plt.title(f"Prop Logic Synth: Per-Run Accuracy with Mean ± SEM\n(n={n_runs} runs)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "prop_logic_synth_per_run_accuracy.png"), dpi=100
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Print aggregated evaluation metrics
if runs:
    accs = [r["acc"] for r in runs]
    vlosses = [r["vloss"] for r in runs if r["vloss"] is not None]
    print(f"Aggregated Accuracy: mean={np.mean(accs):.4f}, sem={sem(accs):.4f}")
    if vlosses:
        print(
            f"Aggregated Val Loss: mean={np.mean(vlosses):.4f}, sem={sem(vlosses):.4f}"
        )
