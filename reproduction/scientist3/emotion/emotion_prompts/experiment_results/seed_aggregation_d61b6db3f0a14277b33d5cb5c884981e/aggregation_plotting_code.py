import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-14_01-42-41_emotion_prompts_attempt_1/logs/0-run/experiment_results/experiment_505d2080795145bc9694c134f03900c9_proc_2173433/experiment_data.npy",
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


def norm(s):
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return f"{f:.4f}".rstrip("0").rstrip(".")
    except Exception:
        return str(s).strip()


# Aggregate accuracy per template across runs
agg_acc = {}
n_samples_list = []
agg_lengths = {}
for exp in all_experiment_data:
    data = exp.get("GSM8K", {})
    acc_dict = data.get("accuracy_per_template", {})
    n_samples_list.append(data.get("n_samples", 0))
    for k, v in acc_dict.items():
        agg_acc.setdefault(k, []).append(v)
    raw = data.get("raw_outputs", {})
    for k, outs in raw.items():
        agg_lengths.setdefault(k, []).extend([len(o) for o in outs])

n_runs = len(all_experiment_data)
total_n = sum(n_samples_list)

# Plot 1: Mean accuracy per template with SEM error bars
try:
    names = list(agg_acc.keys())
    means = np.array([np.mean(agg_acc[k]) for k in names])
    sems = np.array(
        [
            (
                np.std(agg_acc[k], ddof=1) / np.sqrt(len(agg_acc[k]))
                if len(agg_acc[k]) > 1
                else 0.0
            )
            for k in names
        ]
    )
    plt.figure(figsize=(10, 5))
    colors = ["gray" if nm == "neutral" else f"C{i}" for i, nm in enumerate(names)]
    plt.bar(names, means, yerr=sems, color=colors, capsize=5, label="Mean ± SEM")
    neutral_mean = np.mean(agg_acc.get("neutral", [0]))
    plt.axhline(
        neutral_mean,
        color="black",
        linestyle="--",
        alpha=0.5,
        label=f"neutral mean ({neutral_mean:.2f})",
    )
    for i, v in enumerate(means):
        plt.text(i, v + 0.02, f"{v:.2f}", ha="center")
    plt.ylabel("Accuracy (mean across runs)")
    plt.ylim(0, 1.05)
    plt.title(
        f"GSM8K: Mean Accuracy by Emotional Prefix\n(Qwen, runs={n_runs}, total N={total_n})"
    )
    plt.legend()
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "GSM8K_mean_accuracy_by_emotion_prefix.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Mean delta vs neutral with error bars
try:
    names = [k for k in agg_acc.keys() if k != "neutral"]
    neutral_runs = np.array(agg_acc.get("neutral", [0]))
    delta_means = []
    delta_sems = []
    for k in names:
        vals = np.array(agg_acc[k])
        # Pair with neutral run-by-run when possible
        if len(vals) == len(neutral_runs):
            diffs = vals - neutral_runs
        else:
            diffs = vals - np.mean(neutral_runs)
        delta_means.append(np.mean(diffs))
        delta_sems.append(
            np.std(diffs, ddof=1) / np.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
        )
    delta_means = np.array(delta_means)
    delta_sems = np.array(delta_sems)
    plt.figure(figsize=(10, 5))
    colors = ["green" if d >= 0 else "red" for d in delta_means]
    plt.bar(
        names,
        delta_means,
        yerr=delta_sems,
        color=colors,
        capsize=5,
        label="Mean Δ ± SEM",
    )
    plt.axhline(0, color="black", linewidth=0.8)
    for i, v in enumerate(delta_means):
        plt.text(i, v + (0.01 if v >= 0 else -0.02), f"{v:+.2f}", ha="center")
    plt.ylabel("Accuracy Δ vs neutral")
    plt.title(
        f"GSM8K: Mean Accuracy Change vs Neutral Prefix\n(Qwen, runs={n_runs}) Positive=emotion helps"
    )
    plt.legend()
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "GSM8K_mean_accuracy_delta_vs_neutral.png"), dpi=120
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Aggregated output length distribution per template
try:
    names = list(agg_lengths.keys())
    if names:
        length_lists = [agg_lengths[nm] for nm in names]
        means_len = [np.mean(l) for l in length_lists]
        sems_len = [
            np.std(l, ddof=1) / np.sqrt(len(l)) if len(l) > 1 else 0.0
            for l in length_lists
        ]
        plt.figure(figsize=(10, 5))
        plt.bar(
            names,
            means_len,
            yerr=sems_len,
            capsize=5,
            color="skyblue",
            label="Mean length ± SEM",
        )
        plt.ylabel("Output length (chars)")
        plt.title(
            f"GSM8K: Mean Output Length by Emotional Prefix\n(Qwen, aggregated across {n_runs} runs)"
        )
        plt.legend()
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "GSM8K_mean_output_length_by_emotion.png"),
            dpi=120,
        )
        plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Print summary metrics
print(f"=== GSM8K Aggregated Accuracy Summary (runs={n_runs}) ===")
for k in agg_acc:
    vals = np.array(agg_acc[k])
    m = np.mean(vals)
    sem = np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    print(f"  {k}: mean={m:.4f}  sem={sem:.4f}  n_runs={len(vals)}")
