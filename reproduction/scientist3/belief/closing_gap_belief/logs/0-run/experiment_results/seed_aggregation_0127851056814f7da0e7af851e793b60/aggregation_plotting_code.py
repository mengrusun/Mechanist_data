import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2026-07-13_11-02-54_closing_gap_belief_attempt_0/logs/0-run/experiment_results/experiment_b82c7e45c7794df9a943b25d2eda43e4_proc_1420483/experiment_data.npy",
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


# Aggregate
def agg_metric(metric_key):
    """Return dict: variant -> (mean_array_over_layers, sem_array_over_layers)."""
    out = {}
    for v in variants:
        vals = []
        for ed in all_experiment_data:
            d = ed["token_position_pooling"]["trivia_qa"]
            row = [d["per_variant"][v]["per_layer"][int(L)][metric_key] for L in layers]
            vals.append(row)
        arr = np.array(vals, dtype=float)  # (runs, layers)
        mean = np.nanmean(arr, axis=0)
        sem = (
            np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
            if arr.shape[0] > 1
            else np.zeros(arr.shape[1])
        )
        out[v] = (mean, sem)
    return out


try:
    d0 = all_experiment_data[0]["token_position_pooling"]["trivia_qa"]
    variants = d0["pool_variants"]
    layers = d0["target_layers"]
    n_runs = len(all_experiment_data)
except Exception as e:
    print(f"Error extracting meta: {e}")
    variants, layers, n_runs = [], [], 0

# Plot 1: Orthogonality by variant/layer with error bars
try:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(layers))
    w = 0.8 / max(len(variants), 1)
    agg = agg_metric("ortho")
    for i, v in enumerate(variants):
        m, s = agg[v]
        ax.bar(
            xs + (i - (len(variants) - 1) / 2) * w,
            m,
            width=w,
            yerr=s,
            capsize=3,
            label=f"{v} (mean±SEM)",
        )
    # Random control aggregate
    ctrl_runs = []
    for ed in all_experiment_data:
        pv = ed["token_position_pooling"]["trivia_qa"]["per_variant"]
        ctrl_runs.append(
            np.mean(
                [
                    [pv[v]["per_layer"][int(L)]["rand_ctrl_ortho"] for L in layers]
                    for v in variants
                ],
                axis=0,
            )
        )
    ctrl_arr = np.array(ctrl_runs)
    ctrl_mean = np.nanmean(ctrl_arr, axis=0)
    ctrl_sem = (
        np.nanstd(ctrl_arr, axis=0, ddof=1) / np.sqrt(ctrl_arr.shape[0])
        if ctrl_arr.shape[0] > 1
        else np.zeros_like(ctrl_mean)
    )
    ax.errorbar(
        xs, ctrl_mean, yerr=ctrl_sem, fmt="k--o", label="random control (mean±SEM)"
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylabel("1 - |cos(w_corr, w_conf)|")
    ax.set_title(
        f"TriviaQA: Aggregated Subspace Orthogonality by Pooling Variant\n"
        f"(Bars: mean over {n_runs} runs with SEM; Dashed: random control)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(working_dir, "triviaqa_agg_orthogonality_by_pool_variant.png"),
        dpi=140,
    )
    plt.close(fig)
except Exception as e:
    print(f"Error plot1: {e}")
    plt.close()

# Plot 2: Probe Accuracy and AUC with error bars
try:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    xs = np.arange(len(layers))
    w = 0.8 / max(len(variants), 1)
    acc_agg = agg_metric("probe_acc")
    auc_agg = agg_metric("probe_auc")
    for i, v in enumerate(variants):
        m_a, s_a = acc_agg[v]
        m_u, s_u = auc_agg[v]
        axes[0].bar(
            xs + (i - (len(variants) - 1) / 2) * w,
            m_a,
            width=w,
            yerr=s_a,
            capsize=3,
            label=f"{v} (mean±SEM)",
        )
        axes[1].bar(
            xs + (i - (len(variants) - 1) / 2) * w,
            m_u,
            width=w,
            yerr=s_u,
            capsize=3,
            label=f"{v} (mean±SEM)",
        )
    for ax, ttl, ylab in zip(
        axes, ["Left: Probe Accuracy", "Right: Probe AUC"], ["Accuracy", "AUC"]
    ):
        ax.set_xticks(xs)
        ax.set_xticklabels([f"L{L}" for L in layers])
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.legend(fontsize=8)
    fig.suptitle(
        f"TriviaQA: Aggregated Correctness Probe Performance "
        f"(mean over {n_runs} runs, SEM error bars)"
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(working_dir, "triviaqa_agg_probe_acc_auc_by_pool_variant.png"),
        dpi=140,
    )
    plt.close(fig)
except Exception as e:
    print(f"Error plot2: {e}")
    plt.close()

# Plot 3: Confidence MSE with error bars
try:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(layers))
    w = 0.8 / max(len(variants), 1)
    mse_agg = agg_metric("conf_mse")
    for i, v in enumerate(variants):
        m, s = mse_agg[v]
        ax.bar(
            xs + (i - (len(variants) - 1) / 2) * w,
            m,
            width=w,
            yerr=s,
            capsize=3,
            label=f"{v} (mean±SEM)",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylabel("MSE")
    ax.set_title(
        f"TriviaQA: Aggregated Verbalized-Confidence Ridge MSE\n"
        f"(mean over {n_runs} runs with SEM)"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(
        os.path.join(working_dir, "triviaqa_agg_conf_mse_by_pool_variant.png"), dpi=140
    )
    plt.close(fig)
except Exception as e:
    print(f"Error plot3: {e}")
    plt.close()

# Plot 4: ECE comparison verbalized vs probe with error bars
try:
    fig, ax = plt.subplots(figsize=(11, 4.5))
    xs = np.arange(len(layers))
    n = len(variants)
    w = 0.8 / max(2 * n, 1)
    ev_agg = agg_metric("ece_verbalized")
    ep_agg = agg_metric("ece_probe")
    for i, v in enumerate(variants):
        m_v, s_v = ev_agg[v]
        m_p, s_p = ep_agg[v]
        ax.bar(
            xs + (2 * i - (2 * n - 1) / 2) * w,
            m_v,
            width=w,
            yerr=s_v,
            capsize=2,
            label=f"{v} verb (mean±SEM)",
        )
        ax.bar(
            xs + (2 * i + 1 - (2 * n - 1) / 2) * w,
            m_p,
            width=w,
            yerr=s_p,
            capsize=2,
            label=f"{v} probe (mean±SEM)",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"L{L}" for L in layers])
    ax.set_ylabel("ECE")
    ax.set_title(
        f"TriviaQA: Aggregated Expected Calibration Error\n"
        f"(Verbalized vs Probe per variant; mean over {n_runs} runs, SEM)"
    )
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(
        os.path.join(working_dir, "triviaqa_agg_ece_by_pool_variant.png"), dpi=140
    )
    plt.close(fig)
except Exception as e:
    print(f"Error plot4: {e}")
    plt.close()

# Plot 5: Aggregated verbalized confidence hist (pooled across runs)
try:
    all_verb, all_gt = [], []
    for ed in all_experiment_data:
        d = ed["token_position_pooling"]["trivia_qa"]
        all_verb.append(np.array(d["verbalized_confidence"], dtype=float))
        all_gt.append(np.array(d["ground_truth"], dtype=float))
    verb = np.concatenate(all_verb)
    gt = np.concatenate(all_gt)
    valid = ~np.isnan(verb)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.hist(
        verb[valid & (gt == 1)],
        bins=20,
        alpha=0.6,
        label="correct",
        color="green",
        edgecolor="k",
    )
    ax.hist(
        verb[valid & (gt == 0)],
        bins=20,
        alpha=0.6,
        label="incorrect",
        color="red",
        edgecolor="k",
    )
    ax.set_xlabel("Verbalized confidence (%)")
    ax.set_ylabel("Count (pooled across runs)")
    ax.set_title(
        f"TriviaQA: Aggregated Verbalized Confidence Distribution\n"
        f"(Pooled over {n_runs} runs; overall acc={gt.mean():.2f})"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        os.path.join(working_dir, "triviaqa_agg_verbalized_confidence_hist.png"),
        dpi=140,
    )
    plt.close(fig)
except Exception as e:
    print(f"Error plot5: {e}")
    plt.close()

# Print aggregated headline metrics
try:
    print(
        f"Aggregated headline subspace orthogonality by variant (mean ± SEM over {n_runs} runs):"
    )
    headline = {v: [] for v in variants}
    for ed in all_experiment_data:
        hv = ed["token_position_pooling"]["trivia_qa"].get("headline_by_variant", {})
        for v in variants:
            if v in hv:
                headline[v].append(float(hv[v]))
    for v in variants:
        arr = np.array(headline[v], dtype=float)
        if arr.size == 0:
            continue
        m = float(np.nanmean(arr))
        s = float(np.nanstd(arr, ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
        print(f"  {v}: {m:.4f} ± {s:.4f}")
except Exception as e:
    print(f"Error printing headline: {e}")
