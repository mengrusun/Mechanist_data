import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data_path_list = [
        "experiments/2026-07-15_02-20-28_encode_harmfulness_refusal_attempt_1/logs/0-run/experiment_results/experiment_d5161ec81de74b6db265021f7f9c84c8_proc_722033/experiment_data.npy",
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


def get_key(all_eds, keychain):
    """Extract nested key across all runs."""
    out = []
    for ed in all_eds:
        cur = ed
        try:
            for k in keychain:
                cur = cur[k]
            out.append(cur)
        except Exception:
            pass
    return out


# Collect the sub-dict for the specific experiment
runs = []
for ed in all_experiment_data:
    try:
        runs.append(ed["harmful_vs_refusal_dissociation"]["llama3_advbench_alpaca"])
    except Exception:
        pass


def mean_sem(vals):
    a = np.array(vals, dtype=float)
    if len(a) == 0:
        return np.nan, np.nan
    m = np.nanmean(a)
    s = np.nanstd(a, ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    return m, s


def aggregate_layer_metric(runs, position, field):
    """Return sorted layers, mean, sem arrays."""
    layer_vals = {}
    for r in runs:
        for rec in r["metrics"]["val"]:
            if rec.get("position") == position:
                layer_vals.setdefault(rec["layer"], []).append(rec.get(field, np.nan))
    layers = sorted(layer_vals.keys())
    means = [np.nanmean(layer_vals[l]) for l in layers]
    sems = [
        (
            (np.nanstd(layer_vals[l], ddof=1) / np.sqrt(len(layer_vals[l])))
            if len(layer_vals[l]) > 1
            else 0.0
        )
        for l in layers
    ]
    return np.array(layers), np.array(means), np.array(sems)


# Plot 1: Aggregated probe val accuracy by layer
try:
    plt.figure(figsize=(8, 5))
    for pos, marker, label in [
        ("A", "o", "Pos A (final instr)"),
        ("B", "s", "Pos B (post-instr)"),
    ]:
        layers, m, s = aggregate_layer_metric(runs, pos, "acc")
        if len(layers):
            plt.errorbar(
                layers,
                m,
                yerr=s,
                marker=marker,
                linestyle="-",
                label=f"{label} mean±SEM",
                capsize=3,
            )
        layers2, m2, s2 = aggregate_layer_metric(runs, pos, "shuffled")
        if len(layers2):
            plt.errorbar(
                layers2,
                m2,
                yerr=s2,
                marker=marker,
                linestyle="--",
                label=f"{label} shuffled mean±SEM",
                capsize=3,
                alpha=0.6,
            )
    plt.xlabel("Layer")
    plt.ylabel("Validation Accuracy")
    plt.title(
        f"Llama3 AdvBench/Alpaca (aggregated over {len(runs)} runs)\nProbe Val Accuracy by Layer (mean ± SEM)"
    )
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_probe_val_acc.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Aggregated direction norms
try:
    plt.figure(figsize=(8, 5))
    for pos, marker, label in [
        ("A", "o", "Pos A (harm direction)"),
        ("B", "s", "Pos B (refusal direction)"),
    ]:
        layers, m, s = aggregate_layer_metric(runs, pos, "direction_norm")
        if len(layers):
            plt.errorbar(
                layers,
                m,
                yerr=s,
                marker=marker,
                linestyle="-",
                label=f"{label} mean±SEM",
                capsize=3,
            )
    plt.xlabel("Layer")
    plt.ylabel("||mu_harmful - mu_benign||")
    plt.title(
        f"Llama3 AdvBench/Alpaca (aggregated over {len(runs)} runs)\nDirection Norm by Layer (mean ± SEM)"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_direction_norms.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()


def aggregate_steering(runs, eval_key, conds, field):
    means, sems = [], []
    for c in conds:
        vals = []
        for r in runs:
            try:
                vals.append(r["steering_results"][eval_key][c][field])
            except Exception:
                pass
        m, s = mean_sem(vals)
        means.append(m)
        sems.append(s)
    return np.array(means), np.array(sems)


# Plot 3: Steering effects on HARMFUL prompts (aggregated)
try:
    conds = ["baseline", "posH", "negH", "posR", "negR"]
    hp_m, hp_s = aggregate_steering(runs, "harmful_eval", conds, "harm_prob")
    rr_m, rr_s = aggregate_steering(runs, "harmful_eval", conds, "refusal_rate")
    x = np.arange(len(conds))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(x, hp_m, yerr=hp_s, color="tab:blue", capsize=4, label="mean ± SEM")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conds)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Internal Harm Probability")
    axes[0].set_title("Left: Internal Harm Probe (Harmful eval)")
    axes[0].legend()
    axes[1].bar(x, rr_m, yerr=rr_s, color="tab:red", capsize=4, label="mean ± SEM")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conds)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Refusal Rate")
    axes[1].set_title("Right: Behavioral Refusal Rate (Harmful eval)")
    axes[1].legend()
    fig.suptitle(
        f"Llama3 AdvBench/Alpaca (aggregated over {len(runs)} runs): Steering on Harmful Prompts"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_steering_harmful.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Steering effects on BENIGN prompts (aggregated)
try:
    conds = ["baseline", "posR", "posH"]
    hp_m, hp_s = aggregate_steering(runs, "benign_eval", conds, "harm_prob")
    rr_m, rr_s = aggregate_steering(runs, "benign_eval", conds, "refusal_rate")
    x = np.arange(len(conds))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x, hp_m, yerr=hp_s, color="tab:blue", capsize=4, label="mean ± SEM")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conds)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Internal Harm Probability")
    axes[0].set_title("Left: Internal Harm Probe (Benign eval)")
    axes[0].legend()
    axes[1].bar(x, rr_m, yerr=rr_s, color="tab:red", capsize=4, label="mean ± SEM")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conds)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Refusal Rate")
    axes[1].set_title("Right: Behavioral Refusal Rate (Benign eval)")
    axes[1].legend()
    fig.suptitle(
        f"Llama3 AdvBench/Alpaca (aggregated over {len(runs)} runs): Steering on Benign Prompts"
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_steering_benign.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Aggregated Direction Dissociation decomposition
try:
    fields = ["delta_H_h", "delta_R_h", "delta_R_r", "delta_H_r"]
    labels = ["ΔH_h", "ΔR_h", "ΔR_r", "ΔH_r"]
    means, sems = [], []
    for f in fields:
        vals = [r.get(f, np.nan) for r in runs]
        m, s = mean_sem(vals)
        means.append(m)
        sems.append(s)
    ds_vals = [r.get("direction_dissociation_score", np.nan) for r in runs]
    cos_vals = [r.get("cosine_harm_refusal", np.nan) for r in runs]
    ds_m, ds_s = mean_sem(ds_vals)
    cos_m, cos_s = mean_sem(cos_vals)
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, means, yerr=sems, color=colors, capsize=4, label="mean ± SEM")
    plt.axhline(0, color="k", lw=0.5)
    plt.ylabel("Delta")
    plt.title(
        f"Llama3 AdvBench/Alpaca (aggregated over {len(runs)} runs)\n"
        f"Dissociation Decomposition | Score={ds_m:.3f}±{ds_s:.3f} | cos(H,R)={cos_m:.3f}±{cos_s:.3f}"
    )
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_agg_dissociation_score.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Print aggregated metric summary
try:
    ds_vals = [r.get("direction_dissociation_score", np.nan) for r in runs]
    cos_vals = [r.get("cosine_harm_refusal", np.nan) for r in runs]
    ds_m, ds_s = mean_sem(ds_vals)
    cos_m, cos_s = mean_sem(cos_vals)
    print(f"Num runs aggregated: {len(runs)}")
    print(f"Direction Dissociation Score: {ds_m:.4f} ± {ds_s:.4f} (SEM)")
    print(f"cos(harm_dir, ref_dir): {cos_m:.4f} ± {cos_s:.4f} (SEM)")
    print(f"Best H layers: {[r.get('best_H_layer') for r in runs]}")
    print(f"Best R layers: {[r.get('best_R_layer') for r in runs]}")
except Exception as e:
    print(f"Error printing metrics: {e}")
