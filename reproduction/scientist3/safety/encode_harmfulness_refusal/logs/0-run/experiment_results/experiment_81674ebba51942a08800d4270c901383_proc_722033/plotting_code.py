import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
    ed = experiment_data["harmful_vs_refusal_dissociation"]["llama3_advbench_alpaca"]
except Exception as e:
    print(f"Error loading experiment data: {e}")
    ed = None


def _by_pos(records, pos):
    return [r for r in records if r.get("position") == pos]


# Plot 1: Probe val accuracy by layer for both positions
try:
    val_records = ed["metrics"]["val"]
    a_recs = sorted(_by_pos(val_records, "A"), key=lambda r: r["layer"])
    b_recs = sorted(_by_pos(val_records, "B"), key=lambda r: r["layer"])
    layers_a = [r["layer"] for r in a_recs]
    layers_b = [r["layer"] for r in b_recs]
    plt.figure(figsize=(8, 5))
    plt.plot(layers_a, [r["acc"] for r in a_recs], "o-", label="Pos A (final instr)")
    plt.plot(layers_b, [r["acc"] for r in b_recs], "s-", label="Pos B (post-instr)")
    plt.plot(layers_a, [r["shuffled"] for r in a_recs], "o--", label="Pos A shuffled")
    plt.plot(layers_b, [r["shuffled"] for r in b_recs], "s--", label="Pos B shuffled")
    plt.xlabel("Layer")
    plt.ylabel("Validation Accuracy")
    plt.title(
        "Llama3 AdvBench/Alpaca: Probe Val Accuracy by Layer\n"
        f"Best H layer (A)={ed['best_H_layer']}, Best R layer (B)={ed['best_R_layer']}"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_probe_val_acc_by_layer.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot1: {e}")
    plt.close()

# Plot 2: Direction norms by layer (Position A vs B)
try:
    val_records = ed["metrics"]["val"]
    a_recs = sorted(_by_pos(val_records, "A"), key=lambda r: r["layer"])
    b_recs = sorted(_by_pos(val_records, "B"), key=lambda r: r["layer"])
    plt.figure(figsize=(8, 5))
    plt.plot(
        [r["layer"] for r in a_recs],
        [r["direction_norm"] for r in a_recs],
        "o-",
        label="Pos A (harm direction)",
    )
    plt.plot(
        [r["layer"] for r in b_recs],
        [r["direction_norm"] for r in b_recs],
        "s-",
        label="Pos B (refusal direction)",
    )
    plt.xlabel("Layer")
    plt.ylabel("||mu_harmful - mu_benign||")
    plt.title(
        "Llama3 AdvBench/Alpaca: Mean-Difference Direction Norm by Layer\n"
        "Left: Pos A (harmfulness), Right: Pos B (refusal)"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_direction_norms.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot2: {e}")
    plt.close()

# Plot 3: Steering effects on HARMFUL prompts
try:
    sh = ed["steering_results"]["harmful_eval"]
    conds = ["baseline", "posH", "negH", "posR", "negR"]
    hp = [sh[c]["harm_prob"] for c in conds]
    rr = [sh[c]["refusal_rate"] for c in conds]
    x = np.arange(len(conds))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(x, hp, color="tab:blue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conds)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Internal Harm Probability")
    axes[0].set_title("Left: Internal Harm Probe (Harmful eval)")
    axes[1].bar(x, rr, color="tab:red")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conds)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Refusal Rate")
    axes[1].set_title("Right: Behavioral Refusal Rate (Harmful eval)")
    fig.suptitle("Llama3 AdvBench/Alpaca: Steering Effects on Harmful Prompts")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_steering_harmful.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot3: {e}")
    plt.close()

# Plot 4: Steering effects on BENIGN prompts
try:
    sb = ed["steering_results"]["benign_eval"]
    conds = ["baseline", "posR", "posH"]
    hp = [sb[c]["harm_prob"] for c in conds]
    rr = [sb[c]["refusal_rate"] for c in conds]
    x = np.arange(len(conds))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(x, hp, color="tab:blue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(conds)
    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Internal Harm Probability")
    axes[0].set_title("Left: Internal Harm Probe (Benign eval)")
    axes[1].bar(x, rr, color="tab:red")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(conds)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("Refusal Rate")
    axes[1].set_title("Right: Behavioral Refusal Rate (Benign eval)")
    fig.suptitle("Llama3 AdvBench/Alpaca: Steering Effects on Benign Prompts")
    plt.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_steering_benign.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot4: {e}")
    plt.close()

# Plot 5: Direction Dissociation Score decomposition
try:
    labels = ["ΔH_h", "ΔR_h", "ΔR_r", "ΔH_r"]
    vals = [ed["delta_H_h"], ed["delta_R_h"], ed["delta_R_r"], ed["delta_H_r"]]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    plt.figure(figsize=(7, 4.5))
    plt.bar(labels, vals, color=colors)
    plt.axhline(0, color="k", lw=0.5)
    plt.ylabel("Delta")
    plt.title(
        "Llama3 AdvBench/Alpaca: Direction Dissociation Decomposition\n"
        f"Score = {ed['direction_dissociation_score']:.3f} | "
        f"cos(H,R)={ed['cosine_harm_refusal']:.3f}"
    )
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_dissociation_score.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot5: {e}")
    plt.close()

# Plot 6: Jailbreak signature — harm prob on non-refused (under -R) vs baseline
try:
    jb = ed["jailbreak_signature"]
    base_hp = ed["steering_results"]["harmful_eval"]["baseline"]["harm_prob"]
    negR_hp = ed["steering_results"]["harmful_eval"]["negR"]["harm_prob"]
    nonref_hp = jb.get("mean_harm_prob_when_negR_suppresses_refusal")
    labels = [
        "Baseline\n(harmful)",
        "-R steer\n(all harmful)",
        "-R steer\n(non-refused subset)",
    ]
    vals = [base_hp, negR_hp, nonref_hp if nonref_hp is not None else 0.0]
    colors = ["gray", "tab:orange", "tab:purple"]
    plt.figure(figsize=(7, 4.5))
    bars = plt.bar(labels, vals, color=colors)
    plt.ylim(0, 1)
    plt.ylabel("Internal Harm Probability")
    n_nr = jb.get("num_non_refused_under_negR", 0)
    n_tot = jb.get("total_eval", 0)
    plt.title(
        "Llama3 AdvBench/Alpaca: Jailbreak Signature\n"
        f"Non-refused under -R: {n_nr}/{n_tot} | high harm prob => model still knows"
    )
    for b, v in zip(bars, vals):
        plt.text(
            b.get_x() + b.get_width() / 2,
            v + 0.02,
            f"{v:.2f}",
            ha="center",
            va="bottom",
        )
    plt.grid(True, axis="y", alpha=0.3)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_jailbreak_signature.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot6: {e}")
    plt.close()

# Plot 7: Comparison — Harm Prob vs Refusal Rate across all conditions (harmful + benign)
try:
    sh = ed["steering_results"]["harmful_eval"]
    sb = ed["steering_results"]["benign_eval"]
    points = []
    for c, d in sh.items():
        points.append((f"H:{c}", d["harm_prob"], d["refusal_rate"], "tab:red"))
    for c, d in sb.items():
        points.append((f"B:{c}", d["harm_prob"], d["refusal_rate"], "tab:green"))
    plt.figure(figsize=(7, 6))
    for name, hp, rr, col in points:
        plt.scatter(hp, rr, c=col, s=80)
        plt.annotate(
            name, (hp, rr), textcoords="offset points", xytext=(5, 5), fontsize=8
        )
    plt.xlabel("Internal Harm Probability")
    plt.ylabel("Refusal Rate")
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.title(
        "Llama3 AdvBench/Alpaca: Harm Prob vs Refusal Rate\n"
        "Red: Harmful eval, Green: Benign eval"
    )
    plt.grid(True, alpha=0.3)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_harm_vs_refusal_scatter.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating plot7: {e}")
    plt.close()

try:
    print(f"Direction Dissociation Score: {ed['direction_dissociation_score']:.4f}")
    print(f"Best H layer: {ed['best_H_layer']}, Best R layer: {ed['best_R_layer']}")
    print(f"cos(harm_dir, ref_dir): {ed['cosine_harm_refusal']:.4f}")
except Exception as e:
    print(f"Error printing metric: {e}")
