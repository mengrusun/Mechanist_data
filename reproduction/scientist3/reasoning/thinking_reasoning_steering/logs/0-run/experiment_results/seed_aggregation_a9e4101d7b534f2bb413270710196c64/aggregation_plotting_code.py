import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_07-29-12_thinking_reasoning_steering_attempt_1/logs/0-run/experiment_results/experiment_c1a23e0779bd4a9c80bd2ea623f510ce_proc_2022124/experiment_data.npy",
    ]
    all_experiment_data = []
    for p in experiment_data_path_list:
        ed = np.load(
            os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p), allow_pickle=True
        ).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")
    all_experiment_data = []


def get_root(ed):
    return ed.get("steering_analysis", {}).get("r1_distill_llama_8b", {})


roots = [get_root(ed) for ed in all_experiment_data]
n_runs = len(roots)
print(f"Number of runs: {n_runs}")


def sem(a, axis=0):
    a = np.asarray(a, dtype=float)
    if a.shape[axis] < 2:
        return np.zeros(a.shape[1:] if a.ndim > 1 else ())
    return np.std(a, axis=axis, ddof=1) / np.sqrt(a.shape[axis])


# Aggregate dose-response
try:
    # collect behaviours across runs
    behs = set()
    for r in roots:
        behs.update(r.get("dose_response", {}).keys())
    behs = sorted(behs)

    # for each behaviour, gather coefficient grid
    plt.figure(figsize=(9, 5))
    for b in behs:
        coeffs_union = set()
        for r in roots:
            dr = r.get("dose_response", {}).get(b, {})
            coeffs_union.update([float(k) for k in dr.keys()])
        coeffs = sorted(coeffs_union)
        stacked = []
        for r in roots:
            dr = r.get("dose_response", {}).get(b, {})
            row = []
            for c in coeffs:
                v = dr.get(str(c), {}).get("target_mean", np.nan)
                row.append(v)
            stacked.append(row)
        arr = np.array(stacked, dtype=float)
        # ignore all-nan columns
        means = np.nanmean(arr, axis=0)
        sems = (
            np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.sum(~np.isnan(arr), axis=0))
            if arr.shape[0] > 1
            else np.zeros_like(means)
        )
        plt.errorbar(
            coeffs, means, yerr=sems, marker="o", capsize=3, label=f"{b} (mean±SEM)"
        )
    plt.xlabel("Steering coefficient")
    plt.ylabel("Mean target behaviour count")
    plt.title(
        f"R1-Distill-Llama-8B (Math): Aggregated Dose-Response\nMean ± SEM across {n_runs} run(s)"
    )
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "agg_r1_llama8b_dose_response.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated dose-response: {e}")
    plt.close()

# Aggregate accuracy vs coefficient
try:
    behs = set()
    for r in roots:
        behs.update(r.get("dose_response", {}).keys())
    behs = sorted(behs)

    plt.figure(figsize=(9, 5))
    for b in behs:
        coeffs_union = set()
        for r in roots:
            dr = r.get("dose_response", {}).get(b, {})
            coeffs_union.update([float(k) for k in dr.keys()])
        coeffs = sorted(coeffs_union)
        stacked = []
        for r in roots:
            dr = r.get("dose_response", {}).get(b, {})
            row = [dr.get(str(c), {}).get("acc", np.nan) for c in coeffs]
            stacked.append(row)
        arr = np.array(stacked, dtype=float)
        means = np.nanmean(arr, axis=0)
        counts = np.sum(~np.isnan(arr), axis=0)
        sems = (
            np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(counts, 1))
            if arr.shape[0] > 1
            else np.zeros_like(means)
        )
        plt.errorbar(coeffs, means, yerr=sems, marker="s", capsize=3, label=f"{b}")

    # baseline aggregate
    baseline_accs = [r.get("baseline", {}).get("accuracy", None) for r in roots]
    baseline_accs = [x for x in baseline_accs if x is not None]
    if baseline_accs:
        bmean = np.mean(baseline_accs)
        bsem = (
            np.std(baseline_accs, ddof=1) / np.sqrt(len(baseline_accs))
            if len(baseline_accs) > 1
            else 0.0
        )
        plt.axhline(bmean, color="k", ls="--", label=f"baseline={bmean:.2f}±{bsem:.2f}")

    plt.xlabel("Steering coefficient")
    plt.ylabel("Task accuracy")
    plt.ylim(0, 1.05)
    plt.title(
        f"R1-Distill-Llama-8B (Math): Aggregated Accuracy vs Coefficient\nMean ± SEM across {n_runs} run(s)"
    )
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "agg_r1_llama8b_accuracy_vs_coefficient.png"))
    plt.close()
except Exception as e:
    print(f"Error creating aggregated accuracy plot: {e}")
    plt.close()

# Aggregate BSE scores
try:
    behs = set()
    for r in roots:
        behs.update(r.get("bse_scores", {}).keys())
    behs = sorted(behs)
    if behs:
        a1_stack, a2_stack = [], []
        for r in roots:
            bse = r.get("bse_scores", {})
            a1_stack.append([bse.get(b, {}).get("bse_a1", np.nan) for b in behs])
            a2_stack.append(
                [
                    (
                        bse.get(b, {}).get("bse_a2", np.nan)
                        if bse.get(b, {}).get("bse_a2") is not None
                        else np.nan
                    )
                    for b in behs
                ]
            )
        a1 = np.array(a1_stack, dtype=float)
        a2 = np.array(a2_stack, dtype=float)
        m1, m2 = np.nanmean(a1, axis=0), np.nanmean(a2, axis=0)
        c1 = np.sum(~np.isnan(a1), axis=0)
        c2 = np.sum(~np.isnan(a2), axis=0)
        s1 = (
            np.nanstd(a1, axis=0, ddof=1) / np.sqrt(np.maximum(c1, 1))
            if a1.shape[0] > 1
            else np.zeros_like(m1)
        )
        s2 = (
            np.nanstd(a2, axis=0, ddof=1) / np.sqrt(np.maximum(c2, 1))
            if a2.shape[0] > 1
            else np.zeros_like(m2)
        )

        x = np.arange(len(behs))
        w = 0.35
        plt.figure(figsize=(8, 4.5))
        plt.bar(
            x - w / 2,
            m1,
            w,
            yerr=s1,
            capsize=3,
            label="BSE α=1 (mean±SEM)",
            color="steelblue",
        )
        plt.bar(
            x + w / 2,
            m2,
            w,
            yerr=s2,
            capsize=3,
            label="BSE α=2 (mean±SEM)",
            color="orange",
        )
        plt.xticks(x, behs, rotation=20)
        plt.ylabel("BSE score")
        plt.axhline(0, color="k", lw=0.5)
        plt.title(
            f"R1-Distill-Llama-8B (Math): Aggregated BSE Scores\nMean ± SEM across {n_runs} run(s)"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "agg_r1_llama8b_bse_scores.png"))
        plt.close()
except Exception as e:
    print(f"Error creating aggregated BSE plot: {e}")
    plt.close()

# Aggregate layer ablation for hedging
try:
    layer_union = set()
    for r in roots:
        layer_union.update(
            [int(k) for k in r.get("layer_ablation", {}).get("hedging", {}).keys()]
        )
    lis = sorted(layer_union)
    if lis:
        delta_stack, bse_stack, ap_stack, an_stack = [], [], [], []
        for r in roots:
            hedge = r.get("layer_ablation", {}).get("hedging", {})
            delta_stack.append(
                [hedge.get(str(li), {}).get("delta", np.nan) for li in lis]
            )
            bse_stack.append([hedge.get(str(li), {}).get("bse", np.nan) for li in lis])
            ap_stack.append(
                [hedge.get(str(li), {}).get("acc_pos", np.nan) for li in lis]
            )
            an_stack.append(
                [hedge.get(str(li), {}).get("acc_neg", np.nan) for li in lis]
            )
        D = np.array(delta_stack, dtype=float)
        B = np.array(bse_stack, dtype=float)
        AP = np.array(ap_stack, dtype=float)
        AN = np.array(an_stack, dtype=float)

        def ms(a):
            m = np.nanmean(a, axis=0)
            c = np.sum(~np.isnan(a), axis=0)
            s = (
                np.nanstd(a, axis=0, ddof=1) / np.sqrt(np.maximum(c, 1))
                if a.shape[0] > 1
                else np.zeros_like(m)
            )
            return m, s

        dm, ds = ms(D)
        bm, bs = ms(B)
        apm, aps = ms(AP)
        anm, ans_ = ms(AN)

        fig, ax1 = plt.subplots(figsize=(9, 5))
        ax1.errorbar(
            lis,
            dm,
            yerr=ds,
            fmt="o-",
            color="steelblue",
            capsize=3,
            label="Δ freq (mean±SEM)",
        )
        ax1.errorbar(
            lis, bm, yerr=bs, fmt="s-", color="green", capsize=3, label="BSE (mean±SEM)"
        )
        ax1.set_xlabel("Transformer layer index")
        ax1.set_ylabel("Δ / BSE")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.errorbar(
            lis,
            apm,
            yerr=aps,
            fmt="^--",
            color="red",
            alpha=0.7,
            capsize=3,
            label="acc (+coeff)",
        )
        ax2.errorbar(
            lis,
            anm,
            yerr=ans_,
            fmt="v--",
            color="purple",
            alpha=0.7,
            capsize=3,
            label="acc (-coeff)",
        )
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.05)
        l1, la1 = ax1.get_legend_handles_labels()
        l2, la2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, la1 + la2, loc="best", fontsize=8)
        plt.title(
            f"R1-Distill-Llama-8B (Math): Aggregated Layer Ablation ('hedging')\nMean ± SEM across {n_runs} run(s)"
        )
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "agg_r1_llama8b_layer_ablation_hedging.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating aggregated layer ablation plot: {e}")
    plt.close()

# Aggregate baseline behaviour totals
try:
    beh_union = set()
    for r in roots:
        beh_union.update(r.get("baseline", {}).get("counts", {}).keys())
    behs = sorted(beh_union)
    if behs:
        stack = []
        for r in roots:
            bc = r.get("baseline", {}).get("counts", {})
            stack.append([bc.get(b, np.nan) for b in behs])
        arr = np.array(stack, dtype=float)
        m = np.nanmean(arr, axis=0)
        c = np.sum(~np.isnan(arr), axis=0)
        s = (
            np.nanstd(arr, axis=0, ddof=1) / np.sqrt(np.maximum(c, 1))
            if arr.shape[0] > 1
            else np.zeros_like(m)
        )
        plt.figure(figsize=(8, 4.5))
        x = np.arange(len(behs))
        plt.bar(x, m, yerr=s, capsize=3, color="teal", label="mean ± SEM")
        plt.xticks(x, behs, rotation=20)
        plt.ylabel("Total occurrences in baseline CoTs")
        plt.title(
            f"R1-Distill-Llama-8B (Math): Aggregated Baseline Behaviour Totals\nMean ± SEM across {n_runs} run(s)"
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

# Aggregate target vs off-target Δ
try:
    behs = set()
    for r in roots:
        behs.update(r.get("dose_response", {}).keys())
    behs = sorted(behs)
    if behs:
        tgt_stack, off_stack = [], []
        for r in roots:
            row_t, row_o = [], []
            for b in behs:
                dr = r.get("dose_response", {}).get(b, {})
                if "1.0" in dr and "-1.0" in dr:
                    pos_means = dr["1.0"].get("all_means", {})
                    neg_means = dr["-1.0"].get("all_means", {})
                    row_t.append(pos_means.get(b, np.nan) - neg_means.get(b, np.nan))
                    off = [
                        pos_means[k] - neg_means[k]
                        for k in pos_means
                        if k != b and k in neg_means
                    ]
                    row_o.append(np.mean(off) if off else np.nan)
                else:
                    row_t.append(np.nan)
                    row_o.append(np.nan)
            tgt_stack.append(row_t)
            off_stack.append(row_o)
        T = np.array(tgt_stack, dtype=float)
        O = np.array(off_stack, dtype=float)

        def ms(a):
            m = np.nanmean(a, axis=0)
            c = np.sum(~np.isnan(a), axis=0)
            s = (
                np.nanstd(a, axis=0, ddof=1) / np.sqrt(np.maximum(c, 1))
                if a.shape[0] > 1
                else np.zeros_like(m)
            )
            return m, s

        tm, ts = ms(T)
        om, os_ = ms(O)
        x = np.arange(len(behs))
        w = 0.35
        plt.figure(figsize=(8, 4.5))
        plt.bar(
            x - w / 2,
            tm,
            w,
            yerr=ts,
            capsize=3,
            label="Target Δ (mean±SEM)",
            color="steelblue",
        )
        plt.bar(
            x + w / 2,
            om,
            w,
            yerr=os_,
            capsize=3,
            label="Off-target Δ (mean±SEM)",
            color="salmon",
        )
        plt.xticks(x, behs, rotation=20)
        plt.axhline(0, color="k", lw=0.5)
        plt.ylabel("Δ frequency (coeff=+1 vs -1)")
        plt.title(
            f"R1-Distill-Llama-8B (Math): Aggregated Target vs Off-Target Steering\nMean ± SEM across {n_runs} run(s)"
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
    baseline_accs = [r.get("baseline", {}).get("accuracy", None) for r in roots]
    baseline_accs = [x for x in baseline_accs if x is not None]
    if baseline_accs:
        print(
            f"Baseline accuracy: mean={np.mean(baseline_accs):.3f}, "
            f"sem={(np.std(baseline_accs, ddof=1)/np.sqrt(len(baseline_accs))) if len(baseline_accs)>1 else 0.0:.3f}, "
            f"n={len(baseline_accs)}"
        )
    behs = set()
    for r in roots:
        behs.update(r.get("bse_scores", {}).keys())
    for b in sorted(behs):
        a1 = [r.get("bse_scores", {}).get(b, {}).get("bse_a1", np.nan) for r in roots]
        a1 = [x for x in a1 if not (x is None or np.isnan(x))]
        if a1:
            m = np.mean(a1)
            s = (np.std(a1, ddof=1) / np.sqrt(len(a1))) if len(a1) > 1 else 0.0
            print(f"BSE[{b}] α=1: mean={m:.3f} ± {s:.3f} (n={len(a1)})")
except Exception as e:
    print(f"Error printing aggregated metrics: {e}")
