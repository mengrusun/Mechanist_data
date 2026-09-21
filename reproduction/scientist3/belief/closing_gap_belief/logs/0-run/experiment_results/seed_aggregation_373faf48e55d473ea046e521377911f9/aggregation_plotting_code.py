import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2026-07-13_11-02-54_closing_gap_belief_attempt_0/logs/0-run/experiment_results/experiment_da31b6c1e8c14032b37e0322047b650a_proc_1167167/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        ed = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")

# Gather per-run views of the trivia_qa results
runs = []
for ed in all_experiment_data:
    d = ed.get("N_tuning", {}).get("trivia_qa", {})
    if d:
        runs.append(d)
n_runs = len(runs)
print(f"Loaded {n_runs} runs.")

# Determine common Ns and layers (intersection across runs)
if runs:
    Ns_sets = [set(r.get("per_N", {}).keys()) for r in runs]
    Ns = sorted(set.intersection(*Ns_sets)) if Ns_sets else []
    layer_sets = []
    for r in runs:
        s = set()
        for N in r.get("per_N", {}):
            s.update(int(L) for L in r["per_N"][N]["per_layer"].keys())
        layer_sets.append(s)
    layers = sorted(set.intersection(*layer_sets)) if layer_sets else []
else:
    Ns, layers = [], []


def agg_layer_metric(metric_key):
    """Return dict: layer -> (mean[N], sem[N]) across runs."""
    out = {}
    for L in layers:
        arr = np.full((n_runs, len(Ns)), np.nan)
        for i, r in enumerate(runs):
            for j, N in enumerate(Ns):
                try:
                    arr[i, j] = r["per_N"][N]["per_layer"][int(L)][metric_key]
                except Exception:
                    pass
        mean = np.nanmean(arr, axis=0)
        sem = (
            np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
            if n_runs > 1
            else np.zeros_like(mean)
        )
        out[L] = (mean, sem)
    return out


# Plot 1: Orthogonality vs N (aggregated)
try:
    plt.figure(figsize=(7, 4))
    ortho = agg_layer_metric("ortho")
    for L in layers:
        m, s = ortho[L]
        plt.errorbar(
            Ns, m, yerr=s, marker="o", capsize=3, label=f"Layer {L} (mean±SEM)"
        )
    # random control aggregation
    for L in layers:
        arr = np.full((n_runs, len(Ns)), np.nan)
        for i, r in enumerate(runs):
            for j, N in enumerate(Ns):
                try:
                    arr[i, j] = r["per_N"][N]["per_layer"][int(L)].get(
                        "rand_ctrl_ortho", np.nan
                    )
                except Exception:
                    pass
        m = np.nanmean(arr, axis=0)
        s = (
            np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
            if n_runs > 1
            else np.zeros_like(m)
        )
        plt.errorbar(
            Ns,
            m,
            yerr=s,
            marker="x",
            linestyle="--",
            alpha=0.6,
            capsize=3,
            label=f"Rand ctrl L{L}",
        )
    plt.xlabel("N (TriviaQA samples)")
    plt.ylabel("1 - |cos(w_corr, w_conf)|")
    plt.title(
        "TriviaQA: Subspace Orthogonality vs N (aggregated over runs)\n"
        "Solid: Probe pairs, Dashed: Random control"
    )
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_orthogonality_vs_N_agg.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating orthogonality plot: {e}")
    plt.close()

# Plot 2: Probe accuracy vs N (aggregated)
try:
    plt.figure(figsize=(7, 4))
    acc = agg_layer_metric("probe_acc")
    for L in layers:
        m, s = acc[L]
        plt.errorbar(
            Ns, m, yerr=s, marker="s", capsize=3, label=f"Layer {L} (mean±SEM)"
        )
    plt.xlabel("N")
    plt.ylabel("Probe accuracy (test)")
    plt.title("TriviaQA: Correctness Probe Accuracy vs N (mean±SEM across runs)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "triviaqa_probe_acc_vs_N_agg.png"), dpi=140)
    plt.close()
except Exception as e:
    print(f"Error creating probe acc plot: {e}")
    plt.close()

# Plot 3: Probe AUC vs N (aggregated)
try:
    plt.figure(figsize=(7, 4))
    auc = agg_layer_metric("probe_auc")
    for L in layers:
        m, s = auc[L]
        plt.errorbar(
            Ns, m, yerr=s, marker="^", capsize=3, label=f"Layer {L} (mean±SEM)"
        )
    plt.xlabel("N")
    plt.ylabel("Probe ROC-AUC (test)")
    plt.title("TriviaQA: Correctness Probe AUC vs N (mean±SEM across runs)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "triviaqa_probe_auc_vs_N_agg.png"), dpi=140)
    plt.close()
except Exception as e:
    print(f"Error creating AUC plot: {e}")
    plt.close()

# Plot 4: ECE (verbalized vs probe) vs N (aggregated across layers & runs)
try:
    # For each run, build arrays: N -> ece_verbalized (scalar per run/N),
    # ece_probe (mean across layers per run/N)
    ece_v_arr = np.full((n_runs, len(Ns)), np.nan)
    ece_p_arr = np.full((n_runs, len(Ns)), np.nan)
    for i, r in enumerate(runs):
        val = r.get("metrics", {}).get("val", [])
        by_N = {v["N"]: v for v in val}
        for j, N in enumerate(Ns):
            v = by_N.get(N)
            if v is None:
                continue
            evs = [lm["ece_verbalized"] for lm in v["layers"]]
            eps = [lm["ece_probe"] for lm in v["layers"]]
            if evs:
                ece_v_arr[i, j] = float(np.mean(evs))
            if eps:
                ece_p_arr[i, j] = float(np.mean(eps))

    def mean_sem(a):
        m = np.nanmean(a, axis=0)
        s = (
            np.nanstd(a, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(a), axis=0))
            if n_runs > 1
            else np.zeros_like(m)
        )
        return m, s

    mv, sv = mean_sem(ece_v_arr)
    mp, sp = mean_sem(ece_p_arr)
    plt.figure(figsize=(7, 4))
    plt.errorbar(
        Ns, mv, yerr=sv, marker="o", capsize=3, label="Verbalized ECE (mean±SEM)"
    )
    plt.errorbar(
        Ns, mp, yerr=sp, marker="s", capsize=3, label="Probe ECE, layer-avg (mean±SEM)"
    )
    plt.xlabel("N")
    plt.ylabel("Expected Calibration Error")
    plt.title("TriviaQA: Calibration Error vs N (aggregated over runs)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "triviaqa_ece_vs_N_agg.png"), dpi=140)
    plt.close()
except Exception as e:
    print(f"Error creating ECE plot: {e}")
    plt.close()

# Plot 5: Verbalized confidence histogram (aggregated: mean±SEM counts per bin)
try:
    bins = np.linspace(0, 100, 21)
    centers = 0.5 * (bins[:-1] + bins[1:])
    counts_arr = np.full((n_runs, len(centers)), np.nan)
    accs = []
    for i, r in enumerate(runs):
        conf = np.array(r.get("verbalized_confidence", []), dtype=float)
        corr = np.array(r.get("ground_truth", []), dtype=int)
        valid = ~np.isnan(conf)
        if valid.sum() > 0:
            h, _ = np.histogram(conf[valid], bins=bins)
            counts_arr[i] = h
        if len(corr):
            accs.append(corr.mean())
    m = np.nanmean(counts_arr, axis=0)
    s = (
        np.nanstd(counts_arr, axis=0, ddof=1)
        / np.sqrt(np.sum(~np.isnan(counts_arr), axis=0))
        if n_runs > 1
        else np.zeros_like(m)
    )
    plt.figure(figsize=(7, 4))
    plt.bar(
        centers,
        m,
        width=(bins[1] - bins[0]) * 0.9,
        yerr=s,
        capsize=3,
        color="salmon",
        edgecolor="k",
        label="Mean count ±SEM",
    )
    acc_mean = float(np.mean(accs)) if accs else float("nan")
    plt.xlabel("Verbalized confidence (%)")
    plt.ylabel("Count (aggregated across runs)")
    plt.title(
        f"TriviaQA: Verbalized Confidence Distribution\n"
        f"(mean overall accuracy across runs = {acc_mean:.2f})"
    )
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_verbalized_confidence_hist_agg.png"),
        dpi=140,
    )
    plt.close()
except Exception as e:
    print(f"Error creating hist plot: {e}")
    plt.close()

# Plot 6: Reliability diagrams at largest common N (aggregated across runs)
try:
    if Ns and layers:
        N_big = max(Ns)
        mid_layer = layers[len(layers) // 2]
        bins = np.linspace(0, 1, 11)
        centers = 0.5 * (bins[:-1] + bins[1:])

        def reliab(p, y):
            out = []
            for i in range(len(bins) - 1):
                if i < len(bins) - 2:
                    m = (p >= bins[i]) & (p < bins[i + 1])
                else:
                    m = (p >= bins[i]) & (p <= bins[i + 1])
                out.append(y[m].mean() if m.sum() > 0 else np.nan)
            return np.array(out)

        rv_runs = np.full((n_runs, len(centers)), np.nan)
        rp_runs = np.full((n_runs, len(centers)), np.nan)
        for i, r in enumerate(runs):
            preds = r.get("predictions", [])
            entry = next((p for p in preds if p["N"] == N_big), None)
            if entry is None or int(mid_layer) not in entry["per_layer"]:
                continue
            pl = entry["per_layer"][int(mid_layer)]
            y_true = np.array(pl["y_true"])
            probe_prob = np.array(pl["probe_prob"])
            test_idx = np.array(pl["test_idx"])
            vc = np.array(r["verbalized_confidence"])[test_idx] / 100.0
            vc = np.where(np.isnan(vc), np.nanmedian(vc), vc)
            vc = np.clip(vc, 0, 1)
            rv_runs[i] = reliab(vc, y_true)
            rp_runs[i] = reliab(probe_prob, y_true)

        def ms(a):
            m = np.nanmean(a, axis=0)
            s = (
                np.nanstd(a, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(a), axis=0))
                if n_runs > 1
                else np.zeros_like(m)
            )
            return m, s

        mv, sv = ms(rv_runs)
        mp, sp = ms(rp_runs)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        axes[0].errorbar(
            centers, mv, yerr=sv, marker="o", color="C1", capsize=3, label="Mean±SEM"
        )
        axes[0].set_title("Left: Verbalized Confidence")
        axes[0].set_xlabel("Predicted conf")
        axes[0].set_ylabel("Empirical accuracy")
        axes[0].legend(fontsize=8)
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
        axes[1].errorbar(
            centers, mp, yerr=sp, marker="s", color="C2", capsize=3, label="Mean±SEM"
        )
        axes[1].set_title(f"Right: Probe (Layer {mid_layer})")
        axes[1].set_xlabel("Predicted prob")
        axes[1].set_ylabel("Empirical accuracy")
        axes[1].legend(fontsize=8)
        fig.suptitle(
            f"TriviaQA: Reliability Diagrams at N={N_big} (aggregated over runs)"
        )
        fig.tight_layout()
        fig.savefig(
            os.path.join(working_dir, "triviaqa_reliability_diagram_agg.png"), dpi=140
        )
        plt.close(fig)
except Exception as e:
    print(f"Error creating reliability plot: {e}")
    plt.close()

# Print aggregated headline metrics
try:
    print("\n=== Aggregated headline metrics across runs (mean±SEM) ===")
    for L in layers:
        m, s = agg_layer_metric("ortho")[L]
        print(
            f"Layer {L} orthogonality: "
            + ", ".join([f"N={N}: {mm:.3f}±{ss:.3f}" for N, mm, ss in zip(Ns, m, s)])
        )
except Exception as e:
    print(f"Error printing metrics: {e}")
