import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2026-07-13_11-02-54_closing_gap_belief_attempt_0/logs/0-run/experiment_results/experiment_730cd2095785458fae1b4b23348a00cb_proc_1369372/experiment_data.npy",
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

# Gather per-run TriviaQA dictionaries
runs = []
for ed in all_experiment_data:
    d = ed.get("N_tuning", {}).get("trivia_qa", {})
    if d:
        runs.append(d)

if not runs:
    print("No trivia_qa data found; exiting.")


# Determine common Ns and layers
def get_Ns(d):
    return sorted(d.get("per_N", {}).keys())


def get_layers(d):
    per_N = d.get("per_N", {})
    return sorted({int(L) for N in per_N for L in per_N[N]["per_layer"].keys()})


common_Ns = sorted(set.intersection(*[set(get_Ns(r)) for r in runs])) if runs else []
common_layers = (
    sorted(set.intersection(*[set(get_layers(r)) for r in runs])) if runs else []
)
n_runs = len(runs)


def collect(metric_key, ctrl=False):
    """Return dict layer -> (mean array over Ns, sem array over Ns)."""
    out = {}
    for L in common_layers:
        vals = np.zeros((n_runs, len(common_Ns)))
        for i, r in enumerate(runs):
            for j, N in enumerate(common_Ns):
                pl = r["per_N"][N]["per_layer"][int(L)]
                v = pl.get(metric_key, np.nan)
                vals[i, j] = v
        mean = np.nanmean(vals, axis=0)
        sem = np.nanstd(vals, axis=0, ddof=0) / np.sqrt(max(n_runs, 1))
        out[L] = (mean, sem)
    return out


def plot_with_err(data, ylabel, title, fname, marker="o"):
    plt.figure(figsize=(7, 4))
    for L, (mean, sem) in data.items():
        (line,) = plt.plot(common_Ns, mean, marker=marker, label=f"Layer {L} (mean)")
        plt.fill_between(
            common_Ns,
            mean - sem,
            mean + sem,
            alpha=0.2,
            color=line.get_color(),
            label=f"Layer {L} ±SEM" if n_runs > 1 else None,
        )
    plt.xlabel("N (TriviaQA samples)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, fname), dpi=140)
    plt.close()


# Plot 1: Orthogonality
try:
    ortho = collect("ortho")
    plot_with_err(
        ortho,
        "1 - |cos(w_corr, w_conf)|",
        f"TriviaQA: Subspace Orthogonality vs N\n"
        f"(Mean ± SEM across {n_runs} run(s))",
        "triviaqa_orthogonality_vs_N_agg.png",
        marker="o",
    )
except Exception as e:
    print(f"Error creating orthogonality plot: {e}")
    plt.close()

# Plot 1b: Orthogonality with random control overlay
try:
    ortho = collect("ortho")
    ctrl = collect("rand_ctrl_ortho")
    plt.figure(figsize=(7, 4))
    for L in common_layers:
        m, s = ortho[L]
        (line,) = plt.plot(common_Ns, m, marker="o", label=f"L{L} probe (mean)")
        plt.fill_between(common_Ns, m - s, m + s, alpha=0.2, color=line.get_color())
        mc, sc = ctrl[L]
        plt.plot(
            common_Ns,
            mc,
            marker="x",
            linestyle="--",
            color=line.get_color(),
            alpha=0.7,
            label=f"L{L} rand ctrl (mean)",
        )
        plt.fill_between(common_Ns, mc - sc, mc + sc, alpha=0.1, color=line.get_color())
    plt.xlabel("N")
    plt.ylabel("1 - |cos(w_corr, w_conf)|")
    plt.title(
        f"TriviaQA: Orthogonality vs N (Probe vs Random Control)\n"
        f"Mean ± SEM across {n_runs} run(s)"
    )
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "triviaqa_orthogonality_vs_N_ctrl_agg.png"), dpi=140
    )
    plt.close()
except Exception as e:
    print(f"Error creating orthogonality+ctrl plot: {e}")
    plt.close()

# Plot 2: Probe accuracy
try:
    acc = collect("probe_acc")
    plot_with_err(
        acc,
        "Probe accuracy (test)",
        f"TriviaQA: Correctness Probe Accuracy vs N\n"
        f"(Mean ± SEM across {n_runs} run(s))",
        "triviaqa_probe_acc_vs_N_agg.png",
        marker="s",
    )
except Exception as e:
    print(f"Error creating probe acc plot: {e}")
    plt.close()

# Plot 3: Probe AUC
try:
    auc = collect("probe_auc")
    plot_with_err(
        auc,
        "Probe ROC-AUC (test)",
        f"TriviaQA: Correctness Probe AUC vs N\n"
        f"(Mean ± SEM across {n_runs} run(s))",
        "triviaqa_probe_auc_vs_N_agg.png",
        marker="^",
    )
except Exception as e:
    print(f"Error creating AUC plot: {e}")
    plt.close()

# Plot 4: ECE (verbalized vs probe layer-averaged), aggregated
try:
    ece_v_runs = np.zeros((n_runs, len(common_Ns)))
    ece_p_runs = np.zeros((n_runs, len(common_Ns)))
    for i, r in enumerate(runs):
        val = r.get("metrics", {}).get("val", [])
        val_by_N = {v["N"]: v for v in val}
        for j, N in enumerate(common_Ns):
            v = val_by_N.get(N, None)
            if v is None:
                ece_v_runs[i, j] = np.nan
                ece_p_runs[i, j] = np.nan
            else:
                ece_v_runs[i, j] = np.mean([lm["ece_verbalized"] for lm in v["layers"]])
                ece_p_runs[i, j] = np.mean([lm["ece_probe"] for lm in v["layers"]])
    mv, sv = np.nanmean(ece_v_runs, axis=0), np.nanstd(
        ece_v_runs, axis=0, ddof=0
    ) / np.sqrt(max(n_runs, 1))
    mp, sp = np.nanmean(ece_p_runs, axis=0), np.nanstd(
        ece_p_runs, axis=0, ddof=0
    ) / np.sqrt(max(n_runs, 1))
    plt.figure(figsize=(7, 4))
    plt.errorbar(
        common_Ns, mv, yerr=sv, marker="o", capsize=3, label="Verbalized ECE (mean±SEM)"
    )
    plt.errorbar(
        common_Ns,
        mp,
        yerr=sp,
        marker="s",
        capsize=3,
        label="Probe ECE layer-avg (mean±SEM)",
    )
    plt.xlabel("N")
    plt.ylabel("Expected Calibration Error")
    plt.title(f"TriviaQA: Calibration Error vs N\n(Mean ± SEM across {n_runs} run(s))")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "triviaqa_ece_vs_N_agg.png"), dpi=140)
    plt.close()
except Exception as e:
    print(f"Error creating ECE plot: {e}")
    plt.close()

# Plot 5: Verbalized confidence histogram (pooled across runs)
try:
    all_conf = []
    all_corr = []
    for r in runs:
        c = np.array(r.get("verbalized_confidence", []), dtype=float)
        y = np.array(r.get("ground_truth", []), dtype=int)
        all_conf.append(c)
        all_corr.append(y)
    conf = np.concatenate(all_conf) if all_conf else np.array([])
    corr = np.concatenate(all_corr) if all_corr else np.array([])
    valid = ~np.isnan(conf)
    plt.figure(figsize=(6, 4))
    plt.hist(
        conf[valid],
        bins=20,
        color="salmon",
        edgecolor="k",
        label=f"n={valid.sum()} samples pooled",
    )
    acc = corr.mean() if len(corr) else float("nan")
    plt.xlabel("Verbalized confidence (%)")
    plt.ylabel("Count")
    plt.title(
        f"TriviaQA: Verbalized Confidence Distribution (pooled)\n"
        f"Overall accuracy={acc:.2f} across {n_runs} run(s)"
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

# Plot 6: Reliability diagram at largest N, aggregated (mean over runs)
try:
    if common_Ns and common_layers:
        N_big = max(common_Ns)
        mid_layer = common_layers[len(common_layers) // 2]
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

        r_v_all, r_p_all = [], []
        for r in runs:
            preds = r.get("predictions", [])
            entry = next((p for p in preds if p["N"] == N_big), None)
            if entry is None:
                continue
            pl = entry["per_layer"][int(mid_layer)]
            y_true = np.array(pl["y_true"])
            probe_prob = np.array(pl["probe_prob"])
            test_idx = np.array(pl["test_idx"])
            vc = np.array(r["verbalized_confidence"])[test_idx] / 100.0
            vc = np.clip(vc, 0, 1)
            vc = np.where(np.isnan(vc), np.nanmedian(vc), vc)
            r_v_all.append(reliab(vc, y_true))
            r_p_all.append(reliab(probe_prob, y_true))

        r_v_arr = np.array(r_v_all)
        r_p_arr = np.array(r_p_all)
        mv = np.nanmean(r_v_arr, axis=0)
        sv = np.nanstd(r_v_arr, axis=0, ddof=0) / np.sqrt(max(len(r_v_all), 1))
        mp = np.nanmean(r_p_arr, axis=0)
        sp = np.nanstd(r_p_arr, axis=0, ddof=0) / np.sqrt(max(len(r_p_all), 1))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        axes[0].errorbar(
            centers, mv, yerr=sv, marker="o", color="C1", capsize=3, label="Mean ± SEM"
        )
        axes[0].set_title("Left: Verbalized Confidence")
        axes[0].set_xlabel("Predicted conf")
        axes[0].set_ylabel("Empirical accuracy")
        axes[0].legend(fontsize=8)
        axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")
        axes[1].errorbar(
            centers, mp, yerr=sp, marker="s", color="C2", capsize=3, label="Mean ± SEM"
        )
        axes[1].set_title(f"Right: Probe (Layer {mid_layer})")
        axes[1].set_xlabel("Predicted prob")
        axes[1].set_ylabel("Empirical accuracy")
        axes[1].legend(fontsize=8)
        fig.suptitle(
            f"TriviaQA: Reliability Diagrams at N={N_big}\n"
            f"Aggregated across {len(r_v_all)} run(s)"
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
    print(f"Number of runs aggregated: {n_runs}")
    print(f"Common Ns: {common_Ns}")
    print(f"Common Layers: {common_layers}")
    for L in common_layers:
        for metric in ["ortho", "probe_acc", "probe_auc"]:
            vals = np.array(
                [
                    [
                        runs[i]["per_N"][N]["per_layer"][int(L)].get(metric, np.nan)
                        for N in common_Ns
                    ]
                    for i in range(n_runs)
                ]
            )
            mean = np.nanmean(vals, axis=0)
            sem = np.nanstd(vals, axis=0, ddof=0) / np.sqrt(max(n_runs, 1))
            print(
                f"Layer {L} {metric}: "
                + ", ".join(
                    [f"N={N}: {m:.3f}±{s:.3f}" for N, m, s in zip(common_Ns, mean, sem)]
                )
            )
except Exception as e:
    print(f"Error printing metrics: {e}")
