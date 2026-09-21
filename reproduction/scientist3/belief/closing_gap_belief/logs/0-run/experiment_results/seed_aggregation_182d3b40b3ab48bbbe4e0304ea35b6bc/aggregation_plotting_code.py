import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-13_11-02-54_closing_gap_belief_attempt_0/logs/0-run/experiment_results/experiment_8cd31c57364e46cdb1f85169e5434d41_proc_965338/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        ed = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT"), p), allow_pickle=True
        ).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []

ds_key = "trivia_qa"


# Collect layer-wise metrics across runs
def collect_layer_metric(all_ed, key):
    """Return (layers_sorted, mean_array, sem_array)."""
    per_layer = {}
    for ed in all_ed:
        vm = ed.get(ds_key, {}).get("metrics", {}).get("val", [])
        for m in vm:
            L = m.get("layer")
            if L is None or key not in m:
                continue
            per_layer.setdefault(L, []).append(m[key])
    layers = sorted(per_layer.keys())
    means = np.array([np.mean(per_layer[L]) for L in layers])
    n = np.array([len(per_layer[L]) for L in layers])
    stds = np.array(
        [np.std(per_layer[L], ddof=1) if len(per_layer[L]) > 1 else 0.0 for L in layers]
    )
    sems = stds / np.sqrt(np.maximum(n, 1))
    return layers, means, sems, n


n_runs = len(all_experiment_data)

# Plot 1: Orthogonality vs random control (aggregated)
try:
    layers, ortho_m, ortho_se, _ = collect_layer_metric(all_experiment_data, "ortho")
    _, ctrl_m, ctrl_se, _ = collect_layer_metric(all_experiment_data, "rand_ctrl_ortho")
    x = np.arange(len(layers))
    w = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(
        x - w / 2,
        ortho_m,
        w,
        yerr=ortho_se,
        capsize=3,
        label=f"Probe pair (mean±SEM, n={n_runs})",
        color="steelblue",
    )
    plt.bar(
        x + w / 2,
        ctrl_m,
        w,
        yerr=ctrl_se,
        capsize=3,
        label=f"Random control (mean±SEM, n={n_runs})",
        color="gray",
    )
    plt.xticks(x, [str(L) for L in layers])
    plt.xlabel("Layer")
    plt.ylabel("1 - |cos|")
    plt.title(
        "TriviaQA: Subspace Orthogonality by Layer (Aggregated)\nProbe pair vs Random-direction control"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_agg_orthogonality_vs_control.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated orthogonality plot: {e}")
    plt.close()

# Plot 2: Probe accuracy and AUC by layer (aggregated)
try:
    layers, acc_m, acc_se, _ = collect_layer_metric(all_experiment_data, "probe_acc")
    _, auc_m, auc_se, _ = collect_layer_metric(all_experiment_data, "probe_auc")
    x = np.arange(len(layers))
    w = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(
        x - w / 2,
        acc_m,
        w,
        yerr=acc_se,
        capsize=3,
        label=f"Accuracy (mean±SEM, n={n_runs})",
        color="seagreen",
    )
    plt.bar(
        x + w / 2,
        auc_m,
        w,
        yerr=auc_se,
        capsize=3,
        label=f"AUC (mean±SEM, n={n_runs})",
        color="orange",
    )
    plt.xticks(x, [str(L) for L in layers])
    plt.xlabel("Layer")
    plt.ylabel("Score")
    plt.title("TriviaQA: Correctness Probe Performance by Layer (Aggregated)")
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_agg_probe_acc_auc_by_layer.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated probe metrics plot: {e}")
    plt.close()

# Plot 3: ECE verbalized vs probe by layer (aggregated)
try:
    layers, ev_m, ev_se, _ = collect_layer_metric(all_experiment_data, "ece_verbalized")
    _, ep_m, ep_se, _ = collect_layer_metric(all_experiment_data, "ece_probe")
    x = np.arange(len(layers))
    w = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(
        x - w / 2,
        ev_m,
        w,
        yerr=ev_se,
        capsize=3,
        label=f"Verbalized (mean±SEM, n={n_runs})",
        color="salmon",
    )
    plt.bar(
        x + w / 2,
        ep_m,
        w,
        yerr=ep_se,
        capsize=3,
        label=f"Learned probe (mean±SEM, n={n_runs})",
        color="steelblue",
    )
    plt.xticks(x, [str(L) for L in layers])
    plt.xlabel("Layer")
    plt.ylabel("ECE (lower is better)")
    plt.title(
        "TriviaQA: Calibration Error by Layer (Aggregated)\nLeft: Verbalized, Right: Probe"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_agg_ece_verb_vs_probe.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated ECE plot: {e}")
    plt.close()

# Aggregate raw ground truth and verbalized confidence across runs
all_correct = []
all_conf = []
for ed in all_experiment_data:
    d = ed.get(ds_key, {})
    c = np.array(d.get("ground_truth", []))
    cf = np.array(d.get("verbalized_confidence", []))
    if len(c) and len(cf) and len(c) == len(cf):
        all_correct.append(c)
        all_conf.append(cf)

# Plot 4: Aggregated verbalized confidence histogram split by correctness with mean accuracy
try:
    plt.figure(figsize=(6, 4))
    per_run_acc = []
    if all_correct:
        correct_cat = np.concatenate(all_correct)
        conf_cat = np.concatenate(all_conf)
        valid = ~np.isnan(conf_cat)
        c_correct = conf_cat[valid & (correct_cat == 1)]
        c_wrong = conf_cat[valid & (correct_cat == 0)]
        bins = np.linspace(0, 100, 21)
        plt.hist(
            c_correct,
            bins=bins,
            alpha=0.6,
            label=f"Correct (n={len(c_correct)})",
            color="seagreen",
        )
        plt.hist(
            c_wrong,
            bins=bins,
            alpha=0.6,
            label=f"Incorrect (n={len(c_wrong)})",
            color="salmon",
        )
        per_run_acc = [c.mean() for c in all_correct]
    acc_mean = np.mean(per_run_acc) if per_run_acc else float("nan")
    acc_sem = (
        (np.std(per_run_acc, ddof=1) / np.sqrt(len(per_run_acc)))
        if len(per_run_acc) > 1
        else 0.0
    )
    plt.xlabel("Verbalized confidence (%)")
    plt.ylabel("Count (pooled across runs)")
    plt.title(
        f"TriviaQA: Verbalized Confidence Distribution (Aggregated)\nMean accuracy across {n_runs} runs = {acc_mean:.3f} ± {acc_sem:.3f} (SEM)"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_agg_confidence_hist_by_correctness.png"),
        dpi=140,
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated conf hist plot: {e}")
    plt.close()

# Plot 5: Aggregated reliability diagram with error bars across runs
try:
    plt.figure(figsize=(5, 5))
    bins = np.linspace(0, 1, 11)
    n_bins = len(bins) - 1
    per_run_bin_acc = [[] for _ in range(n_bins)]
    per_run_bin_center = [[] for _ in range(n_bins)]
    per_run_bin_count = [[] for _ in range(n_bins)]
    for c_arr, cf_arr in zip(all_correct, all_conf):
        valid = ~np.isnan(cf_arr)
        probs = cf_arr[valid] / 100.0
        labs = c_arr[valid]
        for i in range(n_bins):
            if i == n_bins - 1:
                m = (probs >= bins[i]) & (probs <= bins[i + 1])
            else:
                m = (probs >= bins[i]) & (probs < bins[i + 1])
            if m.sum() > 0:
                per_run_bin_acc[i].append(labs[m].mean())
                per_run_bin_center[i].append(probs[m].mean())
                per_run_bin_count[i].append(m.sum())
    centers, acc_means, acc_sems, counts_total = [], [], [], []
    for i in range(n_bins):
        if len(per_run_bin_acc[i]) > 0:
            centers.append(np.mean(per_run_bin_center[i]))
            acc_means.append(np.mean(per_run_bin_acc[i]))
            if len(per_run_bin_acc[i]) > 1:
                acc_sems.append(
                    np.std(per_run_bin_acc[i], ddof=1)
                    / np.sqrt(len(per_run_bin_acc[i]))
                )
            else:
                acc_sems.append(0.0)
            counts_total.append(np.sum(per_run_bin_count[i]))
    plt.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    if centers:
        plt.errorbar(
            centers,
            acc_means,
            yerr=acc_sems,
            fmt="o",
            color="salmon",
            ecolor="gray",
            capsize=3,
            label=f"Verbalized (mean±SEM across {n_runs} runs)",
        )
        for cx, cy, ct in zip(centers, acc_means, counts_total):
            plt.annotate(
                str(int(ct)),
                (cx, cy),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
    plt.xlabel("Predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(
        "TriviaQA: Reliability Diagram (Aggregated)\nVerbalized confidence vs empirical accuracy"
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_agg_reliability_diagram.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated reliability plot: {e}")
    plt.close()

# Print aggregated headline metrics
try:
    headline_scores = []
    headline_layers = []
    for ed in all_experiment_data:
        d = ed.get(ds_key, {})
        s = d.get("subspace_orthogonality_score")
        L = d.get("headline_layer")
        if s is not None:
            headline_scores.append(s)
        if L is not None:
            headline_layers.append(L)
    if headline_scores:
        m = np.mean(headline_scores)
        se = (
            np.std(headline_scores, ddof=1) / np.sqrt(len(headline_scores))
            if len(headline_scores) > 1
            else 0.0
        )
        print(
            f"Aggregated headline subspace orthogonality (n={len(headline_scores)}): {m:.4f} ± {se:.4f} (SEM)"
        )
        print(f"Headline layers across runs: {headline_layers}")
    accs_all = [c.mean() for c in all_correct]
    if accs_all:
        m = np.mean(accs_all)
        se = (
            np.std(accs_all, ddof=1) / np.sqrt(len(accs_all))
            if len(accs_all) > 1
            else 0.0
        )
        print(
            f"Aggregated overall accuracy (n={len(accs_all)} runs): {m:.4f} ± {se:.4f} (SEM)"
        )
except Exception as e:
    print(f"Error printing aggregated metrics: {e}")
