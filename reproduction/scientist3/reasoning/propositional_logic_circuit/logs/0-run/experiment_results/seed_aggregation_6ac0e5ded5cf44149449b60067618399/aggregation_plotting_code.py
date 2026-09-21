import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_06-08-42_propositional_logic_circuit_attempt_1/logs/0-run/experiment_results/experiment_ee490aacf0c04a89b4be81cbfcd294e1_proc_3610328/experiment_data.npy"
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

# Aggregate: for each max_length, gather accuracy and val_loss across runs
try:
    all_mls = set()
    for exp in all_experiment_data:
        tuning = exp.get("max_length_tuning", {})
        for k in tuning.keys():
            all_mls.add(int(k))
    mls = sorted(all_mls)
except Exception as e:
    print(f"Error extracting mls: {e}")
    mls = []


def sem(a):
    a = np.asarray(a, dtype=float)
    if len(a) <= 1:
        return 0.0
    return np.std(a, ddof=1) / np.sqrt(len(a))


# Plot 1: Aggregated accuracy vs max_length with SEM
try:
    means, sems = [], []
    for m in mls:
        vals = []
        for exp in all_experiment_data:
            t = exp.get("max_length_tuning", {}).get(str(m))
            if t is not None:
                vals.append(t["metrics"]["val"][0]["task_accuracy"])
        means.append(np.mean(vals) if vals else np.nan)
        sems.append(sem(vals))
    plt.figure(figsize=(6, 4))
    plt.errorbar(mls, means, yerr=sems, fmt="o-", capsize=4, label="Mean ± SEM")
    plt.xscale("log", base=2)
    plt.xlabel("max_length")
    plt.ylabel("Accuracy")
    plt.title(
        "Synthetic Logic Dataset: Aggregated Accuracy vs max_length\nMean ± SEM across runs"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "synthetic_logic_agg_accuracy_vs_max_length.png"),
        dpi=100,
    )
    plt.close()
    print("Aggregated accuracy:")
    for m, mu, se in zip(mls, means, sems):
        print(f"  max_length={m}: acc={mu:.4f} ± {se:.4f}")
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
    plt.close()

# Plot 2: Aggregated val loss vs max_length with SEM
try:
    means, sems = [], []
    for m in mls:
        vals = []
        for exp in all_experiment_data:
            t = exp.get("max_length_tuning", {}).get(str(m))
            if t is not None:
                vals.append(t["losses"]["val"][0]["val_loss"])
        means.append(np.mean(vals) if vals else np.nan)
        sems.append(sem(vals))
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        mls, means, yerr=sems, fmt="o-", color="orange", capsize=4, label="Mean ± SEM"
    )
    plt.xscale("log", base=2)
    plt.xlabel("max_length")
    plt.ylabel("Validation Loss (NLL)")
    plt.title(
        "Synthetic Logic Dataset: Aggregated Val Loss vs max_length\nMean ± SEM across runs"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "synthetic_logic_agg_valloss_vs_max_length.png"),
        dpi=100,
    )
    plt.close()
    print("Aggregated val loss:")
    for m, mu, se in zip(mls, means, sems):
        print(f"  max_length={m}: loss={mu:.4f} ± {se:.4f}")
except Exception as e:
    print(f"Error creating aggregated loss plot: {e}")
    plt.close()

# Plot 3: Pooled True vs False logit distributions per max_length (across runs)
try:
    if len(mls) > 0:
        fig, axes = plt.subplots(1, len(mls), figsize=(4 * len(mls), 4), squeeze=False)
        for i, m in enumerate(mls):
            tl_all, fl_all = [], []
            for exp in all_experiment_data:
                t = exp.get("max_length_tuning", {}).get(str(m))
                if t is not None:
                    tl_all.extend(list(t["true_logits"]))
                    fl_all.extend(list(t["false_logits"]))
            tl_all = np.array(tl_all)
            fl_all = np.array(fl_all)
            ax = axes[0, i]
            ax.hist(
                tl_all,
                bins=25,
                alpha=0.5,
                label=f"True (μ={tl_all.mean():.2f})",
                color="green",
            )
            ax.hist(
                fl_all,
                bins=25,
                alpha=0.5,
                label=f"False (μ={fl_all.mean():.2f})",
                color="red",
            )
            ax.axvline(tl_all.mean(), color="green", linestyle="--", alpha=0.7)
            ax.axvline(fl_all.mean(), color="red", linestyle="--", alpha=0.7)
            ax.set_title(f"max_length={m}")
            ax.set_xlabel("Logit value")
            ax.set_ylabel("Count")
            ax.legend()
        fig.suptitle(
            "Synthetic Logic Dataset: Pooled True vs False Logit Distributions\n(Aggregated across runs; dashed = mean)"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "synthetic_logic_agg_logit_distributions.png"),
            dpi=100,
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated logit distribution plot: {e}")
    plt.close()

# Plot 4: Aggregated margin mean ± SEM per max_length, split by correctness
try:
    correct_means, correct_sems = [], []
    incorrect_means, incorrect_sems = [], []
    for m in mls:
        c_margins, i_margins = [], []
        for exp in all_experiment_data:
            t = exp.get("max_length_tuning", {}).get(str(m))
            if t is None:
                continue
            tl = np.array(t["true_logits"])
            fl = np.array(t["false_logits"])
            preds = np.array(t["predictions"])
            gts = np.array(t["ground_truth"])
            margin = tl - fl
            mask = preds == gts
            c_margins.extend(list(margin[mask]))
            i_margins.extend(list(margin[~mask]))
        correct_means.append(np.mean(c_margins) if c_margins else np.nan)
        correct_sems.append(sem(c_margins) if c_margins else 0)
        incorrect_means.append(np.mean(i_margins) if i_margins else np.nan)
        incorrect_sems.append(sem(i_margins) if i_margins else 0)
    plt.figure(figsize=(7, 4))
    x = np.arange(len(mls))
    w = 0.35
    plt.bar(
        x - w / 2,
        correct_means,
        w,
        yerr=correct_sems,
        capsize=4,
        label="Correct",
        color="blue",
        alpha=0.7,
    )
    plt.bar(
        x + w / 2,
        incorrect_means,
        w,
        yerr=incorrect_sems,
        capsize=4,
        label="Incorrect",
        color="red",
        alpha=0.7,
    )
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.xticks(x, [str(m) for m in mls])
    plt.xlabel("max_length")
    plt.ylabel("Mean margin (True - False logit)")
    plt.title(
        "Synthetic Logic Dataset: Aggregated Prediction Margin\nby Correctness (Mean ± SEM, pooled across runs)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "synthetic_logic_agg_margin_by_correctness.png"),
        dpi=100,
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated margin plot: {e}")
    plt.close()

# Plot 5: Aggregated confusion matrix (summed across runs) per max_length
try:
    if len(mls) > 0:
        fig, axes = plt.subplots(
            1, len(mls), figsize=(3.5 * len(mls), 3.5), squeeze=False
        )
        for i, m in enumerate(mls):
            cm = np.zeros((2, 2), dtype=int)
            for exp in all_experiment_data:
                t = exp.get("max_length_tuning", {}).get(str(m))
                if t is None:
                    continue
                preds = np.array(t["predictions"])
                gts = np.array(t["ground_truth"])
                for p, g in zip(preds, gts):
                    cm[int(g), int(p)] += 1
            ax = axes[0, i]
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["False", "True"])
            ax.set_yticklabels(["False", "True"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Ground Truth")
            ax.set_title(f"max_length={m}")
            for r in range(2):
                for c in range(2):
                    ax.text(
                        c,
                        r,
                        str(cm[r, c]),
                        ha="center",
                        va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black",
                    )
        fig.suptitle(
            "Synthetic Logic Dataset: Aggregated Confusion Matrices\n(Summed across runs; Rows: GT, Cols: Predicted)"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "synthetic_logic_agg_confusion_matrices.png"),
            dpi=100,
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated confusion matrix plot: {e}")
    plt.close()

print(f"\nNumber of runs aggregated: {len(all_experiment_data)}")
