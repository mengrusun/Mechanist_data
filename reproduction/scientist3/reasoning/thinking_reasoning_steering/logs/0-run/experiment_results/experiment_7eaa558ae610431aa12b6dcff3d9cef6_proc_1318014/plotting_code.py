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

root = experiment_data.get("max_new_tokens", {}).get("r1_distill_llama_8b_steering", {})
runs = root.get("runs", {})
mnts = sorted([int(k) for k in runs.keys()])
COEFFS = [-1.0, 0.0, 1.0]

# Plot 1: tuning curves
try:
    rates = [runs[str(m)]["behaviour_steering_success_rate"] for m in mnts]
    accs = [runs[str(m)]["baseline_accuracy"] for m in mnts]
    plt.figure(figsize=(7, 4))
    plt.plot(mnts, rates, "o-", label="steering success rate")
    plt.plot(mnts, accs, "s-", label="baseline accuracy")
    plt.xlabel("max_new_tokens")
    plt.ylabel("value")
    plt.title(
        "R1-Distill-Llama-8B Steering: Tuning over max_new_tokens\nSuccess Rate vs Baseline Accuracy"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "r1_llama8b_tuning_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating tuning curve plot: {e}")
    plt.close()

# Plot 2: behaviour counts per MNT
for m in mnts:
    try:
        sr = runs[str(m)].get("steering_results", {})
        behs = list(sr.keys())
        if not behs:
            continue
        plt.figure(figsize=(8, 5))
        x = np.arange(len(behs))
        w = 0.25
        for i, coeff in enumerate(COEFFS):
            vals = [sr[b].get(str(coeff), {}).get("mean", 0.0) for b in behs]
            plt.bar(x + (i - 1) * w, vals, w, label=f"coeff={coeff}")
        plt.xticks(x, behs, rotation=20)
        plt.ylabel("Mean behaviour count per task")
        plt.title(
            f"R1-Distill-Llama-8B (Math tasks): Behaviour Counts by Steering Coefficient\nmax_new_tokens={m}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"r1_llama8b_behaviour_counts_mnt{m}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating behaviour counts plot for MNT={m}: {e}")
        plt.close()

# Plot 3: accuracy per behaviour steering per MNT
for m in mnts:
    try:
        sr = runs[str(m)].get("steering_results", {})
        behs = list(sr.keys())
        if not behs:
            continue
        plt.figure(figsize=(8, 5))
        x = np.arange(len(behs))
        w = 0.25
        for i, coeff in enumerate(COEFFS):
            vals = [sr[b].get(str(coeff), {}).get("acc", 0.0) for b in behs]
            plt.bar(x + (i - 1) * w, vals, w, label=f"coeff={coeff}")
        plt.xticks(x, behs, rotation=20)
        plt.ylabel("Accuracy on test subset")
        plt.ylim(0, 1.05)
        plt.title(
            f"R1-Distill-Llama-8B (Math tasks): Accuracy under Steering\nmax_new_tokens={m}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"r1_llama8b_steering_accuracy_mnt{m}.png")
        )
        plt.close()
    except Exception as e:
        print(f"Error creating accuracy plot for MNT={m}: {e}")
        plt.close()

# Plot 4: baseline behaviour totals per MNT
try:
    plt.figure(figsize=(8, 5))
    behaviours = None
    for m in mnts:
        bc = runs[str(m)].get("behaviour_counts", {}).get("baseline_total", {})
        if not bc:
            continue
        if behaviours is None:
            behaviours = list(bc.keys())
    if behaviours:
        x = np.arange(len(behaviours))
        w = 0.8 / max(1, len(mnts))
        for i, m in enumerate(mnts):
            bc = runs[str(m)].get("behaviour_counts", {}).get("baseline_total", {})
            vals = [bc.get(b, 0) for b in behaviours]
            plt.bar(x + (i - (len(mnts) - 1) / 2) * w, vals, w, label=f"MNT={m}")
        plt.xticks(x, behaviours, rotation=20)
        plt.ylabel("Total occurrences (baseline)")
        plt.title(
            "R1-Distill-Llama-8B (Math tasks): Baseline Behaviour Totals by max_new_tokens"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "r1_llama8b_baseline_behaviour_totals.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating baseline totals plot: {e}")
    plt.close()

# Plot 5: predictions vs ground truth for best run
try:
    best_val = root.get("best_value", None)
    preds = root.get("predictions", [])
    gts = root.get("ground_truth", [])
    if preds and gts:

        def parse_num(p):
            try:
                import re

                s = re.sub(r"[^0-9\.\-]", "", str(p))
                return float(s) if s not in ("", "-", ".", "-.") else None
            except:
                return None

        correct_flags = []
        for p, g in zip(preds, gts):
            v = parse_num(p)
            correct_flags.append(
                1 if (v is not None and abs(v - float(g)) < 1e-3) else 0
            )
        plt.figure(figsize=(10, 3))
        idx = np.arange(len(correct_flags))
        colors = ["green" if c == 1 else "red" for c in correct_flags]
        plt.bar(idx, correct_flags, color=colors)
        plt.xlabel("Task index")
        plt.ylabel("Correct (1) / Incorrect (0)")
        plt.title(
            f"R1-Distill-Llama-8B (Math tasks): Per-Task Correctness at Best max_new_tokens={best_val}\nGreen=correct, Red=incorrect"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "r1_llama8b_best_run_correctness.png"))
    plt.close()
except Exception as e:
    print(f"Error creating correctness plot: {e}")
    plt.close()

# Print eval metrics
try:
    print(f"Best max_new_tokens: {root.get('best_value')}")
    print(f"Best success_rate: {root.get('best_success_rate')}")
    for m in mnts:
        r = runs[str(m)]
        print(
            f"MNT={m}: baseline_acc={r.get('baseline_accuracy'):.3f}, success_rate={r.get('behaviour_steering_success_rate'):.3f}"
        )
except Exception as e:
    print(f"Error printing metrics: {e}")
