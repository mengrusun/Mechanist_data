import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_07-29-12_thinking_reasoning_steering_attempt_1/logs/0-run/experiment_results/experiment_fa198464160a4c63a45dbc0c2ceb0efb_proc_2753333/experiment_data.npy"
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


def sem(x):
    x = np.asarray(x, dtype=float)
    if len(x) <= 1:
        return 0.0
    return np.std(x, ddof=1) / np.sqrt(len(x))


# Collect roots across runs
roots = []
for ed in all_experiment_data:
    r = ed.get("steering_analysis", {}).get("r1_distill_llama_8b", {})
    if r:
        roots.append(r)

n_runs = len(roots)
print(f"Number of runs aggregated: {n_runs}")

# ----- Aggregate dose-response across runs -----
# Structure: behaviour -> coeff -> list of target_mean values (one per run)
try:
    behaviours = set()
    coeffs_by_beh = {}
    for r in roots:
        dr = r.get("dose_response", {})
        for b, sub in dr.items():
            behaviours.add(b)
            coeffs_by_beh.setdefault(b, set()).update(sub.keys())

    # Plot 1: Dose-response mean ± SE
    plt.figure(figsize=(8, 5))
    for b in sorted(behaviours):
        cs_str = sorted(coeffs_by_beh[b], key=lambda x: float(x))
        cs = [float(c) for c in cs_str]
        means, ses = [], []
        for c in cs_str:
            vals = []
            for r in roots:
                v = r.get("dose_response", {}).get(b, {}).get(c, {}).get("target_mean")
                if v is not None:
                    vals.append(v)
            if vals:
                means.append(np.mean(vals))
                ses.append(sem(vals))
            else:
                means.append(np.nan)
                ses.append(0.0)
        plt.errorbar(cs, means, yerr=ses, marker="o", capsize=3, label=f"{b} (mean±SE)")
    plt.xlabel("Steering coefficient (× BASE_SCALE)")
    plt.ylabel("Mean target behaviour count in CoT")
    plt.title(
        f"R1-Distill-Llama-8B (Math): Dose-Response Curves (Aggregated, n={n_runs})\nMean ± SE across runs"
    )
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "agg_r1_llama8b_dose_response_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated dose-response plot: {e}")
    plt.close()

# Plot 2: Accuracy vs coefficient mean ± SE
try:
    plt.figure(figsize=(8, 5))
    for b in sorted(behaviours):
        cs_str = sorted(coeffs_by_beh[b], key=lambda x: float(x))
        cs = [float(c) for c in cs_str]
        means, ses = [], []
        for c in cs_str:
            vals = []
            for r in roots:
                v = r.get("dose_response", {}).get(b, {}).get(c, {}).get("acc")
                if v is not None:
                    vals.append(v)
            if vals:
                means.append(np.mean(vals))
                ses.append(sem(vals))
            else:
                means.append(np.nan)
                ses.append(0.0)
        plt.errorbar(cs, means, yerr=ses, marker="s", capsize=3, label=f"{b} (mean±SE)")

    baseline_accs = [
        r.get("baseline", {}).get("accuracy")
        for r in roots
        if r.get("baseline", {}).get("accuracy") is not None
    ]
    if baseline_accs:
        bm = np.mean(baseline_accs)
        bs = sem(baseline_accs)
        plt.axhline(bm, color="k", ls="--", label=f"baseline mean={bm:.2f}")
        plt.fill_between(
            [
                min([float(c) for cs in coeffs_by_beh.values() for c in cs]),
                max([float(c) for cs in coeffs_by_beh.values() for c in cs]),
            ],
            bm - bs,
            bm + bs,
            color="k",
            alpha=0.1,
            label="baseline ±SE",
        )
    plt.xlabel("Steering coefficient")
    plt.ylabel("Task accuracy")
    plt.ylim(0, 1.05)
    plt.title(
        f"R1-Distill-Llama-8B (Math): Accuracy vs Coefficient (Aggregated, n={n_runs})\nMean ± SE across runs"
    )
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "agg_r1_llama8b_accuracy_vs_coefficient.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
    plt.close()

# Plot 3: BSE scores bar chart mean ± SE
try:
    bse_behs = set()
    for r in roots:
        bse_behs.update(r.get("bse_scores", {}).keys())
    bse_behs = sorted(bse_behs)
    if bse_behs:
        a1_means, a1_ses, a2_means, a2_ses = [], [], [], []
        for b in bse_behs:
            a1_vals, a2_vals = [], []
            for r in roots:
                s = r.get("bse_scores", {}).get(b, {})
                if s.get("bse_a1") is not None:
                    a1_vals.append(s["bse_a1"])
                if s.get("bse_a2") is not None:
                    a2_vals.append(s["bse_a2"])
            a1_means.append(np.mean(a1_vals) if a1_vals else 0.0)
            a1_ses.append(sem(a1_vals) if a1_vals else 0.0)
            a2_means.append(np.mean(a2_vals) if a2_vals else 0.0)
            a2_ses.append(sem(a2_vals) if a2_vals else 0.0)

        x = np.arange(len(bse_behs))
        w = 0.35
        plt.figure(figsize=(7, 4))
        plt.bar(
            x - w / 2,
            a1_means,
            w,
            yerr=a1_ses,
            capsize=3,
            label="BSE α=1 (mean±SE)",
            color="steelblue",
        )
        plt.bar(
            x + w / 2,
            a2_means,
            w,
            yerr=a2_ses,
            capsize=3,
            label="BSE α=2 (mean±SE)",
            color="orange",
        )
        plt.xticks(x, bse_behs, rotation=20)
        plt.ylabel("BSE score")
        plt.axhline(0, color="k", lw=0.5)
        plt.title(
            f"R1-Distill-Llama-8B (Math): BSE Scores (Aggregated, n={n_runs})\nMean ± SE across runs, α=1 vs α=2"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "agg_r1_llama8b_bse_scores.png"))
        plt.close()
except Exception as e:
    print(f"Error creating aggregated BSE plot: {e}")
    plt.close()

# Plot 4: Per-layer ablation for hedging mean ± SE
try:
    layer_indices = set()
    for r in roots:
        h = r.get("layer_ablation", {}).get("hedging", {})
        layer_indices.update(int(k) for k in h.keys())
    layer_indices = sorted(layer_indices)
    if layer_indices:
        d_means, d_ses = [], []
        b_means, b_ses = [], []
        ap_means, ap_ses = [], []
        an_means, an_ses = [], []
        for li in layer_indices:
            ds, bs_, ap, an = [], [], [], []
            for r in roots:
                e = r.get("layer_ablation", {}).get("hedging", {}).get(str(li), {})
                if not e:
                    continue
                if e.get("delta") is not None:
                    ds.append(e["delta"])
                if e.get("bse") is not None:
                    bs_.append(e["bse"])
                if e.get("acc_pos") is not None:
                    ap.append(e["acc_pos"])
                if e.get("acc_neg") is not None:
                    an.append(e["acc_neg"])
            d_means.append(np.mean(ds) if ds else np.nan)
            d_ses.append(sem(ds))
            b_means.append(np.mean(bs_) if bs_ else np.nan)
            b_ses.append(sem(bs_))
            ap_means.append(np.mean(ap) if ap else np.nan)
            ap_ses.append(sem(ap))
            an_means.append(np.mean(an) if an else np.nan)
            an_ses.append(sem(an))

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.errorbar(
            layer_indices,
            d_means,
            yerr=d_ses,
            fmt="o-",
            color="steelblue",
            capsize=3,
            label="Δ freq (mean±SE)",
        )
        ax1.errorbar(
            layer_indices,
            b_means,
            yerr=b_ses,
            fmt="s-",
            color="green",
            capsize=3,
            label="BSE (mean±SE)",
        )
        ax1.set_xlabel("Transformer layer index")
        ax1.set_ylabel("Δ / BSE")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.errorbar(
            layer_indices,
            ap_means,
            yerr=ap_ses,
            fmt="^--",
            color="red",
            alpha=0.7,
            capsize=3,
            label="acc(+) (mean±SE)",
        )
        ax2.errorbar(
            layer_indices,
            an_means,
            yerr=an_ses,
            fmt="v--",
            color="purple",
            alpha=0.7,
            capsize=3,
            label="acc(-) (mean±SE)",
        )
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.05)
        l1, lab1 = ax1.get_legend_handles_labels()
        l2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lab1 + lab2, loc="best", fontsize=8)
        plt.title(
            f"R1-Distill-Llama-8B (Math): Per-Layer Ablation 'hedging' (Aggregated, n={n_runs})\nLeft: Δ/BSE, Right: Accuracy — Mean ± SE across runs"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "agg_r1_llama8b_layer_ablation_hedging.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated layer ablation plot: {e}")
    plt.close()

# Plot 5: Baseline behaviour totals mean ± SE
try:
    beh_keys = set()
    for r in roots:
        beh_keys.update(r.get("baseline", {}).get("counts", {}).keys())
    beh_keys = sorted(beh_keys)
    if beh_keys:
        means, ses = [], []
        for b in beh_keys:
            vals = [r.get("baseline", {}).get("counts", {}).get(b) for r in roots]
            vals = [v for v in vals if v is not None]
            means.append(np.mean(vals) if vals else 0.0)
            ses.append(sem(vals) if vals else 0.0)
        plt.figure(figsize=(7, 4))
        x = np.arange(len(beh_keys))
        plt.bar(x, means, yerr=ses, capsize=3, color="teal", label="Mean count ± SE")
        plt.xticks(x, beh_keys, rotation=20)
        plt.ylabel("Total occurrences across baseline CoTs")
        plt.title(
            f"R1-Distill-Llama-8B (Math): Baseline Behaviour Totals (Aggregated, n={n_runs})\nMean ± SE across runs — Unsteered CoT"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "agg_r1_llama8b_baseline_behaviour_totals.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated baseline totals plot: {e}")
    plt.close()

# Plot 6: Target vs off-target Δ mean ± SE
try:
    behs = sorted(behaviours) if behaviours else []
    if behs:
        tgt_means, tgt_ses, off_means, off_ses = [], [], [], []
        for b in behs:
            tgt_vals, off_vals = [], []
            for r in roots:
                dr = r.get("dose_response", {}).get(b, {})
                if "1.0" in dr and "-1.0" in dr:
                    pos = dr["1.0"].get("all_means", {})
                    neg = dr["-1.0"].get("all_means", {})
                    if b in pos and b in neg:
                        tgt_vals.append(pos[b] - neg[b])
                    off = [pos[k] - neg[k] for k in pos if k != b and k in neg]
                    if off:
                        off_vals.append(np.mean(off))
            tgt_means.append(np.mean(tgt_vals) if tgt_vals else 0.0)
            tgt_ses.append(sem(tgt_vals))
            off_means.append(np.mean(off_vals) if off_vals else 0.0)
            off_ses.append(sem(off_vals))

        x = np.arange(len(behs))
        w = 0.35
        plt.figure(figsize=(7, 4))
        plt.bar(
            x - w / 2,
            tgt_means,
            w,
            yerr=tgt_ses,
            capsize=3,
            label="Target Δ (mean±SE)",
            color="steelblue",
        )
        plt.bar(
            x + w / 2,
            off_means,
            w,
            yerr=off_ses,
            capsize=3,
            label="Off-target Δ (mean±SE)",
            color="salmon",
        )
        plt.xticks(x, behs, rotation=20)
        plt.ylabel("Δ frequency (coeff=+1 vs -1)")
        plt.axhline(0, color="k", lw=0.5)
        plt.title(
            f"R1-Distill-Llama-8B (Math): Target vs Off-Target Steering (Aggregated, n={n_runs})\nMean ± SE across runs"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "agg_r1_llama8b_target_vs_offtarget.png"))
        plt.close()
except Exception as e:
    print(f"Error creating aggregated target vs off-target plot: {e}")
    plt.close()

# Print aggregated metrics
try:
    if baseline_accs:
        print(
            f"Baseline accuracy: mean={np.mean(baseline_accs):.4f} ± SE={sem(baseline_accs):.4f}"
        )
    for b in sorted(behaviours):
        print(f"\nDose-response[{b}] aggregated (n={n_runs}):")
        for c in sorted(coeffs_by_beh[b], key=lambda x: float(x)):
            tvals = [
                r.get("dose_response", {}).get(b, {}).get(c, {}).get("target_mean")
                for r in roots
            ]
            avals = [
                r.get("dose_response", {}).get(b, {}).get(c, {}).get("acc")
                for r in roots
            ]
            tvals = [v for v in tvals if v is not None]
            avals = [v for v in avals if v is not None]
            if tvals and avals:
                print(
                    f"  coeff={c}: target_mean={np.mean(tvals):.2f}±{sem(tvals):.2f}, "
                    f"acc={np.mean(avals):.3f}±{sem(avals):.3f}"
                )
    for b in sorted(bse_behs) if "bse_behs" in dir() else []:
        a1 = [r.get("bse_scores", {}).get(b, {}).get("bse_a1") for r in roots]
        a1 = [v for v in a1 if v is not None]
        if a1:
            print(f"BSE[{b}] α=1: mean={np.mean(a1):.3f} ± SE={sem(a1):.3f}")
except Exception as e:
    print(f"Error printing aggregated metrics: {e}")
