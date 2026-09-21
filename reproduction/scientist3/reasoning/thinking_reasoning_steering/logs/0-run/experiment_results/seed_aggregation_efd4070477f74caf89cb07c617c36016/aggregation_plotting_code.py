import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_07-29-12_thinking_reasoning_steering_attempt_1/logs/0-run/experiment_results/experiment_7eaa558ae610431aa12b6dcff3d9cef6_proc_1318014/experiment_data.npy",
    ]
    all_experiment_data = []
    for experiment_data_path in experiment_data_path_list:
        ed = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), experiment_data_path),
            allow_pickle=True,
        ).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

COEFFS = [-1.0, 0.0, 1.0]


def sem(arr):
    arr = np.asarray(arr, dtype=float)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


# Collect aggregated data across runs
# Structure: for each MNT: list of baseline_acc, success_rate over runs
agg = (
    {}
)  # mnt -> {'baseline_acc': [], 'success_rate': [], 'steering_results': {beh: {coeff: {'mean': [], 'acc': []}}}, 'baseline_total': {beh: []}}
all_mnts = set()
for ed in all_experiment_data:
    root = ed.get("max_new_tokens", {}).get("r1_distill_llama_8b_steering", {})
    runs = root.get("runs", {})
    for k, r in runs.items():
        m = int(k)
        all_mnts.add(m)
        d = agg.setdefault(
            m,
            {
                "baseline_acc": [],
                "success_rate": [],
                "steering_results": {},
                "baseline_total": {},
            },
        )
        if "baseline_accuracy" in r:
            d["baseline_acc"].append(r["baseline_accuracy"])
        if "behaviour_steering_success_rate" in r:
            d["success_rate"].append(r["behaviour_steering_success_rate"])
        sr = r.get("steering_results", {})
        for beh, cdict in sr.items():
            b_entry = d["steering_results"].setdefault(beh, {})
            for coeff in COEFFS:
                c_entry = b_entry.setdefault(str(coeff), {"mean": [], "acc": []})
                cv = cdict.get(str(coeff), {})
                if "mean" in cv:
                    c_entry["mean"].append(cv["mean"])
                if "acc" in cv:
                    c_entry["acc"].append(cv["acc"])
        bt = r.get("behaviour_counts", {}).get("baseline_total", {})
        for beh, v in bt.items():
            d["baseline_total"].setdefault(beh, []).append(v)

mnts = sorted(all_mnts)
n_runs = len(all_experiment_data)

# Plot 1: aggregated tuning curves with SEM
try:
    if mnts:
        rate_means = [
            np.mean(agg[m]["success_rate"]) if agg[m]["success_rate"] else 0
            for m in mnts
        ]
        rate_sems = [sem(agg[m]["success_rate"]) for m in mnts]
        acc_means = [
            np.mean(agg[m]["baseline_acc"]) if agg[m]["baseline_acc"] else 0
            for m in mnts
        ]
        acc_sems = [sem(agg[m]["baseline_acc"]) for m in mnts]
        plt.figure(figsize=(7, 4))
        plt.errorbar(
            mnts,
            rate_means,
            yerr=rate_sems,
            fmt="o-",
            capsize=4,
            label=f"steering success rate (mean ± SEM, n={n_runs})",
        )
        plt.errorbar(
            mnts,
            acc_means,
            yerr=acc_sems,
            fmt="s-",
            capsize=4,
            label=f"baseline accuracy (mean ± SEM, n={n_runs})",
        )
        plt.xlabel("max_new_tokens")
        plt.ylabel("value")
        plt.title(
            "R1-Distill-Llama-8B Steering (Math tasks): Aggregated Tuning Curves\nSuccess Rate vs Baseline Accuracy across runs"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "r1_llama8b_agg_tuning_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated tuning curve plot: {e}")
    plt.close()

# Plot 2: aggregated behaviour counts per MNT (select up to 5 MNTs)
try:
    sel_mnts = mnts[:: max(1, len(mnts) // 5)][:5] if mnts else []
    for m in sel_mnts:
        sr = agg[m]["steering_results"]
        behs = list(sr.keys())
        if not behs:
            continue
        plt.figure(figsize=(8, 5))
        x = np.arange(len(behs))
        w = 0.25
        for i, coeff in enumerate(COEFFS):
            means = [
                np.mean(sr[b][str(coeff)]["mean"]) if sr[b][str(coeff)]["mean"] else 0
                for b in behs
            ]
            sems = [sem(sr[b][str(coeff)]["mean"]) for b in behs]
            plt.bar(
                x + (i - 1) * w,
                means,
                w,
                yerr=sems,
                capsize=3,
                label=f"coeff={coeff} (mean±SEM)",
            )
        plt.xticks(x, behs, rotation=20)
        plt.ylabel("Mean behaviour count per task")
        plt.title(
            f"R1-Distill-Llama-8B (Math tasks): Aggregated Behaviour Counts by Steering Coefficient\nmax_new_tokens={m}, n_runs={n_runs}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"r1_llama8b_agg_behaviour_counts_mnt{m}.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated behaviour counts plot: {e}")
    plt.close()

# Plot 3: aggregated accuracy per behaviour steering per MNT
try:
    sel_mnts = mnts[:: max(1, len(mnts) // 5)][:5] if mnts else []
    for m in sel_mnts:
        sr = agg[m]["steering_results"]
        behs = list(sr.keys())
        if not behs:
            continue
        plt.figure(figsize=(8, 5))
        x = np.arange(len(behs))
        w = 0.25
        for i, coeff in enumerate(COEFFS):
            means = [
                np.mean(sr[b][str(coeff)]["acc"]) if sr[b][str(coeff)]["acc"] else 0
                for b in behs
            ]
            sems = [sem(sr[b][str(coeff)]["acc"]) for b in behs]
            plt.bar(
                x + (i - 1) * w,
                means,
                w,
                yerr=sems,
                capsize=3,
                label=f"coeff={coeff} (mean±SEM)",
            )
        plt.xticks(x, behs, rotation=20)
        plt.ylabel("Accuracy on test subset")
        plt.ylim(0, 1.1)
        plt.title(
            f"R1-Distill-Llama-8B (Math tasks): Aggregated Accuracy under Steering\nmax_new_tokens={m}, n_runs={n_runs}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, f"r1_llama8b_agg_steering_accuracy_mnt{m}.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
    plt.close()

# Plot 4: aggregated baseline behaviour totals per MNT
try:
    behaviours = None
    for m in mnts:
        if agg[m]["baseline_total"]:
            behaviours = list(agg[m]["baseline_total"].keys())
            break
    if behaviours and mnts:
        plt.figure(figsize=(9, 5))
        x = np.arange(len(behaviours))
        w = 0.8 / max(1, len(mnts))
        for i, m in enumerate(mnts):
            bt = agg[m]["baseline_total"]
            means = [np.mean(bt.get(b, [0])) for b in behaviours]
            sems = [sem(bt.get(b, [0])) for b in behaviours]
            plt.bar(
                x + (i - (len(mnts) - 1) / 2) * w,
                means,
                w,
                yerr=sems,
                capsize=3,
                label=f"MNT={m}",
            )
        plt.xticks(x, behaviours, rotation=20)
        plt.ylabel("Total occurrences (baseline, mean±SEM)")
        plt.title(
            f"R1-Distill-Llama-8B (Math tasks): Aggregated Baseline Behaviour Totals by max_new_tokens\nn_runs={n_runs}"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "r1_llama8b_agg_baseline_behaviour_totals.png")
        )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated baseline totals plot: {e}")
    plt.close()

# Print aggregated metrics
try:
    print(f"Number of runs aggregated: {n_runs}")
    for m in mnts:
        d = agg[m]
        ba = d["baseline_acc"]
        sr = d["success_rate"]
        print(
            f"MNT={m}: baseline_acc={np.mean(ba):.3f}±{sem(ba):.3f} (n={len(ba)}), "
            f"success_rate={np.mean(sr):.3f}±{sem(sr):.3f} (n={len(sr)})"
        )
except Exception as e:
    print(f"Error printing aggregated metrics: {e}")
