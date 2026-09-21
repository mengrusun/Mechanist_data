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

key = "r1_distill_llama_8b_steering"
data = experiment_data.get(key, {})
steering_results = data.get("steering_results", {})
behaviours = list(steering_results.keys())


# Parse coeff keys (stored as strings)
def parse_coeffs(behaviour_results):
    coeffs = sorted([float(k) for k in behaviour_results.keys()])
    return coeffs


# Plot 1: Baseline behaviour counts
try:
    plt.figure(figsize=(7, 4))
    bc = data.get("behaviour_counts", {}).get("baseline_total", {})
    if bc:
        names = list(bc.keys())
        vals = [bc[n] for n in names]
        plt.bar(names, vals, color="steelblue")
        plt.ylabel("Total occurrences across baseline chains")
        plt.title(
            "R1-Distill-Llama-8B: Baseline Behaviour Counts\n(Aggregated across all reasoning chains)"
        )
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "r1_llama8b_baseline_behaviour_counts.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating baseline behaviour count plot: {e}")
    plt.close()

# Plot 2: Mean behaviour count per chain vs steering coefficient
try:
    if behaviours:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(behaviours))
        coeffs = parse_coeffs(steering_results[behaviours[0]])
        w = 0.8 / len(coeffs)
        for i, coeff in enumerate(coeffs):
            vals = [steering_results[b][str(coeff)]["mean"] for b in behaviours]
            ax.bar(
                x + (i - (len(coeffs) - 1) / 2) * w,
                vals,
                w,
                label=f"coeff={coeff:+.1f}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(behaviours, rotation=20)
        ax.set_ylabel("Mean behaviour count / chain")
        ax.set_title(
            "R1-Distill-Llama-8B: Behaviour Frequency vs Steering Coefficient\n(Higher coeff should amplify targeted behaviour)"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "r1_llama8b_steering_behaviour_counts.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating behaviour count vs coeff plot: {e}")
    plt.close()

# Plot 3: Task accuracy under steering
try:
    if behaviours:
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(behaviours))
        coeffs = parse_coeffs(steering_results[behaviours[0]])
        w = 0.8 / len(coeffs)
        for i, coeff in enumerate(coeffs):
            accs = [steering_results[b][str(coeff)]["acc"] for b in behaviours]
            ax.bar(
                x + (i - (len(coeffs) - 1) / 2) * w,
                accs,
                w,
                label=f"coeff={coeff:+.1f}",
            )
        baseline_acc = data.get("baseline_accuracy", None)
        if baseline_acc is not None:
            ax.axhline(
                baseline_acc,
                color="red",
                linestyle="--",
                label=f"baseline acc={baseline_acc:.2f}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(behaviours, rotation=20)
        ax.set_ylabel("Task accuracy")
        ax.set_ylim(0, 1.05)
        ax.set_title(
            "R1-Distill-Llama-8B: Task Accuracy under Steering\n(Steering should preserve task performance)"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "r1_llama8b_steering_task_accuracy.png"))
    plt.close()
except Exception as e:
    print(f"Error creating task accuracy plot: {e}")
    plt.close()

# Plot 4: Per-behaviour dose-response curves (mean count vs coefficient)
try:
    if behaviours:
        fig, ax = plt.subplots(figsize=(7, 5))
        for b in behaviours:
            coeffs = parse_coeffs(steering_results[b])
            means = [steering_results[b][str(c)]["mean"] for c in coeffs]
            ax.plot(coeffs, means, marker="o", label=b)
        ax.set_xlabel("Steering coefficient")
        ax.set_ylabel("Mean behaviour count / chain")
        ax.set_title(
            "R1-Distill-Llama-8B: Dose-Response of Steering\n(Curves show behaviour amplification/suppression)"
        )
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "r1_llama8b_steering_dose_response.png"))
    plt.close()
except Exception as e:
    print(f"Error creating dose-response plot: {e}")
    plt.close()

# Plot 5: Pos/Neg pool sizes per behaviour
try:
    bc = data.get("behaviour_counts", {})
    beh_names = ["hedging", "backtracking", "self_correction", "example_gen"]
    pos_vals = [bc.get(f"{b}_pos", 0) for b in beh_names]
    neg_vals = [bc.get(f"{b}_neg", 0) for b in beh_names]
    if any(pos_vals) or any(neg_vals):
        fig, ax = plt.subplots(figsize=(7, 4))
        x = np.arange(len(beh_names))
        w = 0.35
        ax.bar(x - w / 2, pos_vals, w, label="Positive examples", color="green")
        ax.bar(x + w / 2, neg_vals, w, label="Negative examples", color="gray")
        ax.set_xticks(x)
        ax.set_xticklabels(beh_names, rotation=20)
        ax.set_ylabel("Number of sentence activations")
        ax.set_title(
            "R1-Distill-Llama-8B: Pos/Neg Pool Sizes for Steering Vector Extraction\n(Left: Positive matches, Right: Negative samples)"
        )
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "r1_llama8b_pool_sizes.png"))
    plt.close()
except Exception as e:
    print(f"Error creating pool sizes plot: {e}")
    plt.close()

# Print summary metric
rate = data.get("behaviour_steering_success_rate", None)
baseline_acc = data.get("baseline_accuracy", None)
print(f"Baseline accuracy: {baseline_acc}")
print(f"Behaviour steering success rate: {rate}")
