import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

experiment_data_path_list = [
    "experiments/2026-07-15_02-20-28_encode_harmfulness_refusal_attempt_1/logs/0-run/experiment_results/experiment_9ca461b9b413473186b66527f93bc201_proc_821931/experiment_data.npy",
]

all_experiment_data = []
try:
    for p in experiment_data_path_list:
        full = os.path.join(os.getenv("AI_SCIENTIST_ROOT", ""), p)
        ed = np.load(full, allow_pickle=True).item()
        all_experiment_data.append(ed)
except Exception as e:
    print(f"Error loading experiment data: {e}")


def collect_runs(key_path):
    """Get list of per-run dicts for steering_layer_transfer/llama3_advbench_alpaca."""
    runs = []
    for ed in all_experiment_data:
        try:
            d = ed["steering_layer_transfer"]["llama3_advbench_alpaca"]
            runs.append(d)
        except Exception:
            pass
    return runs


runs = collect_runs(None)
if len(runs) == 0:
    print("No runs found.")


def mean_sem(arr_list):
    arr = np.array(arr_list, dtype=float)
    m = np.nanmean(arr, axis=0)
    if arr.shape[0] > 1:
        sem = np.nanstd(arr, axis=0, ddof=1) / np.sqrt(arr.shape[0])
    else:
        sem = np.zeros_like(m)
    return m, sem


# Determine common injection layers and probe layers (use first run for structure)
try:
    ref = runs[0]
    inj_layers = ref["injection_layers"]
    probe_layers = sorted(ref["probes_A"].keys())
except Exception as e:
    print(f"Error getting reference structure: {e}")
    inj_layers, probe_layers = [], []

n_runs = len(runs)

# Plot 1: Probe accuracy aggregated
try:

    def gather_probe(field, side):
        vals = []
        for r in runs:
            probes = r[f"probes_{side}"]
            vals.append([probes[L][field] for L in probe_layers])
        return mean_sem(vals)

    plt.figure(figsize=(8, 5))
    for side, marker in [("A", "o"), ("B", "s")]:
        m, s = gather_probe("val_acc", side)
        label_side = "Pos A (harmfulness)" if side == "A" else "Pos B (refusal-ctx)"
        plt.errorbar(
            probe_layers,
            m,
            yerr=s,
            fmt=f"{marker}-",
            label=f"{label_side} val (mean±SEM, n={n_runs})",
            capsize=3,
        )
    # shuffled baseline
    m, s = gather_probe("shuffled_val_acc", "A")
    plt.errorbar(
        probe_layers,
        m,
        yerr=s,
        fmt="x:",
        alpha=0.6,
        label=f"Pos A shuffled (mean±SEM, n={n_runs})",
        capsize=3,
    )
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(
        "LLaMA-3 AdvBench/Alpaca: Aggregated Probe Accuracy by Layer\nMean ± SEM across runs"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_probe_accuracy.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated probe accuracy plot: {e}")
    plt.close()

# Plot 2: Aggregated DDS
try:
    vals = []
    for r in runs:
        vals.append(
            [
                r["per_injection_layer"][L]["direction_dissociation_score"]
                for L in inj_layers
            ]
        )
    m, s = mean_sem(vals)
    plt.figure(figsize=(8, 5))
    plt.errorbar(
        inj_layers,
        m,
        yerr=s,
        fmt="o-",
        color="purple",
        capsize=3,
        label=f"DDS (mean±SEM, n={n_runs})",
    )
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("Injection layer")
    plt.ylabel("Direction Dissociation Score")
    plt.title(
        "LLaMA-3 AdvBench/Alpaca: Aggregated DDS vs Injection Layer\nMean ± SEM across runs"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_dds.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated DDS plot: {e}")
    plt.close()

# Plot 3: Aggregated component deltas
try:
    fields = [
        ("delta_H_h", "ΔH_h (H-dir→harm)"),
        ("delta_R_h", "ΔR_h (H-dir→refusal)"),
        ("delta_R_r", "ΔR_r (R-dir→refusal)"),
        ("delta_H_r", "ΔH_r (R-dir→harm)"),
    ]
    plt.figure(figsize=(8, 5))
    for f, lab in fields:
        vals = []
        for r in runs:
            vals.append([r["per_injection_layer"][L][f] for L in inj_layers])
        m, s = mean_sem(vals)
        plt.errorbar(
            inj_layers, m, yerr=s, fmt="o-", capsize=3, label=f"{lab} (mean±SEM)"
        )
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("Injection layer")
    plt.ylabel("Delta")
    plt.title(
        f"LLaMA-3 AdvBench/Alpaca: Aggregated Steering Effect Components\nMean ± SEM across runs (n={n_runs})"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_component_deltas.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated component deltas plot: {e}")
    plt.close()

# Plot 4: Aggregated refusal rates
try:
    plt.figure(figsize=(8, 5))
    specs = [
        ("harmful_eval", "negR", "RR harmful (-R) [jailbreak]", "o-"),
        ("harmful_eval", "posR", "RR harmful (+R)", "v-"),
        ("benign_eval", "posR", "RR benign (+R) [over-refuse]", "s-"),
    ]
    for evalk, cond, lab, mk in specs:
        vals = []
        for r in runs:
            vals.append(
                [
                    r["per_injection_layer"][L][evalk][cond]["refusal_rate"]
                    for L in inj_layers
                ]
            )
        m, s = mean_sem(vals)
        plt.errorbar(
            inj_layers, m, yerr=s, fmt=mk, capsize=3, label=f"{lab} (mean±SEM)"
        )
    # baselines
    for evalk, color, lab in [
        ("harmful_eval", "tab:blue", "baseline harmful"),
        ("benign_eval", "tab:orange", "baseline benign"),
    ]:
        bvals = [
            r["per_injection_layer"][inj_layers[0]][evalk]["baseline"]["refusal_rate"]
            for r in runs
        ]
        bm = np.mean(bvals)
        plt.axhline(
            bm, color=color, linestyle=":", alpha=0.6, label=f"{lab} RR (mean={bm:.2f})"
        )
    plt.xlabel("Injection layer")
    plt.ylabel("Refusal rate")
    plt.title(
        f"LLaMA-3 AdvBench/Alpaca: Aggregated Refusal Rates by Layer\nMean ± SEM across runs (n={n_runs})"
    )
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_refusal_rates.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated refusal rates plot: {e}")
    plt.close()

# Plot 5: Aggregated harm probabilities
try:
    plt.figure(figsize=(8, 5))
    specs = [
        ("harmful_eval", "posH", "HarmProb harmful (+H)", "o-"),
        ("harmful_eval", "negH", "HarmProb harmful (-H)", "s-"),
        ("benign_eval", "posH", "HarmProb benign (+H)", "^-"),
    ]
    for evalk, cond, lab, mk in specs:
        vals = []
        for r in runs:
            vals.append(
                [
                    r["per_injection_layer"][L][evalk][cond]["harm_prob"]
                    for L in inj_layers
                ]
            )
        m, s = mean_sem(vals)
        plt.errorbar(
            inj_layers, m, yerr=s, fmt=mk, capsize=3, label=f"{lab} (mean±SEM)"
        )
    for evalk, color, lab in [
        ("harmful_eval", "tab:blue", "baseline harmful"),
        ("benign_eval", "tab:orange", "baseline benign"),
    ]:
        bvals = [
            r["per_injection_layer"][inj_layers[0]][evalk]["baseline"]["harm_prob"]
            for r in runs
        ]
        bm = np.mean(bvals)
        plt.axhline(
            bm, color=color, linestyle=":", alpha=0.6, label=f"{lab} (mean={bm:.2f})"
        )
    plt.xlabel("Injection layer")
    plt.ylabel("Mean harmfulness probability")
    plt.title(
        f"LLaMA-3 AdvBench/Alpaca: Aggregated Harm Probability by Layer\nMean ± SEM across runs (n={n_runs})"
    )
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_harm_prob.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated harm prob plot: {e}")
    plt.close()

# Plot 6: Aggregated jailbreak signature
try:
    frac_vals, hp_vals = [], []
    for r in runs:
        jf = []
        jh = []
        for L in inj_layers:
            js = r["per_injection_layer"][L]["jailbreak_signature"]
            jf.append(js["num_non_refused_under_negR"] / max(1, js["total_eval"]))
            jh.append(js.get("mean_harm_prob_when_negR_suppresses_refusal") or 0.0)
        frac_vals.append(jf)
        hp_vals.append(jh)
    fm, fs = mean_sem(frac_vals)
    hm, hs = mean_sem(hp_vals)
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.errorbar(
        inj_layers,
        fm,
        yerr=fs,
        fmt="o-",
        color="tab:red",
        capsize=3,
        label=f"Frac non-refused (-R) mean±SEM",
    )
    ax1.set_xlabel("Injection layer")
    ax1.set_ylabel("Fraction non-refused", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.errorbar(
        inj_layers,
        hm,
        yerr=hs,
        fmt="s-",
        color="tab:blue",
        capsize=3,
        label=f"Mean harm prob mean±SEM",
    )
    ax2.set_ylabel("Mean harm prob", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
    plt.title(
        f"LLaMA-3 AdvBench/Alpaca: Aggregated Jailbreak Signature\nMean ± SEM across runs (n={n_runs})"
    )
    fig.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_jailbreak_signature.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating aggregated jailbreak signature plot: {e}")
    plt.close()

# Print aggregated metrics
try:
    print(f"=== Aggregated metrics across n={n_runs} runs ===")
    # DDS
    vals = []
    for r in runs:
        vals.append(
            [
                r["per_injection_layer"][L]["direction_dissociation_score"]
                for L in inj_layers
            ]
        )
    m, s = mean_sem(vals)
    print("DDS by injection layer (mean ± SEM):")
    for i, L in enumerate(inj_layers):
        print(f"  layer {L:>2d}: {m[i]:.4f} ± {s[i]:.4f}")
    cos_vals = [r.get("cosine_harm_refusal", np.nan) for r in runs]
    print(
        f"\ncosine(harm_dir, ref_dir): mean={np.nanmean(cos_vals):.4f}, SEM={np.nanstd(cos_vals, ddof=1)/np.sqrt(len(cos_vals)) if len(cos_vals) > 1 else 0.0:.4f}"
    )
    bh = [r["best_H_layer"] for r in runs]
    br = [r["best_R_layer"] for r in runs]
    print(f"best_H_layer values: {bh}")
    print(f"best_R_layer values: {br}")
except Exception as e:
    print(f"Error printing aggregated summary: {e}")
