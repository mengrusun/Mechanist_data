import matplotlib.pyplot as plt
import numpy as np
import os

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

try:
    d = experiment_data["steering_layer_transfer"]["llama3_advbench_alpaca"]
    inj_layers = d["injection_layers"]
    per_layer = d["per_injection_layer"]
    best_H = d["best_H_layer"]
    best_R = d["best_R_layer"]
    probes_A = d["probes_A"]
    probes_B = d["probes_B"]
except Exception as e:
    print(f"Error unpacking: {e}")
    d = None

# Plot 1: Probe accuracy by layer (positions A and B)
try:
    layers = sorted(probes_A.keys())
    plt.figure(figsize=(8, 5))
    plt.plot(
        layers,
        [probes_A[L]["val_acc"] for L in layers],
        "o-",
        label="Pos A (harmfulness) val",
    )
    plt.plot(
        layers,
        [probes_A[L]["train_acc"] for L in layers],
        "o--",
        label="Pos A train",
        alpha=0.5,
    )
    plt.plot(
        layers,
        [probes_B[L]["val_acc"] for L in layers],
        "s-",
        label="Pos B (refusal-ctx) val",
    )
    plt.plot(
        layers,
        [probes_B[L]["train_acc"] for L in layers],
        "s--",
        label="Pos B train",
        alpha=0.5,
    )
    plt.plot(
        layers,
        [probes_A[L]["shuffled_val_acc"] for L in layers],
        "x:",
        label="Pos A shuffled",
        alpha=0.5,
    )
    plt.xlabel("Layer")
    plt.ylabel("Accuracy")
    plt.title(
        "LLaMA-3 AdvBench/Alpaca: Probe Accuracy by Layer\nPositions A (harmfulness) and B (refusal context)"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_probe_accuracy_by_layer.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating probe accuracy plot: {e}")
    plt.close()

# Plot 2: DDS vs injection layer
try:
    xs = inj_layers
    ys = [per_layer[L]["direction_dissociation_score"] for L in xs]
    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, "o-", color="purple", label="DDS")
    plt.axvline(
        best_H, color="tab:blue", linestyle="--", alpha=0.5, label=f"best_H={best_H}"
    )
    plt.axvline(
        best_R, color="tab:orange", linestyle="--", alpha=0.5, label=f"best_R={best_R}"
    )
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("Injection layer")
    plt.ylabel("Direction Dissociation Score")
    plt.title(
        "LLaMA-3 AdvBench/Alpaca: DDS vs Injection Layer\nDirections extracted at best_H / best_R"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_dds_by_injection_layer.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating DDS plot: {e}")
    plt.close()

# Plot 3: Component deltas by injection layer
try:
    xs = inj_layers
    plt.figure(figsize=(8, 5))
    plt.plot(
        xs,
        [per_layer[L]["delta_H_h"] for L in xs],
        "o-",
        label="ΔH_h (H-dir → harm prob)",
    )
    plt.plot(
        xs,
        [per_layer[L]["delta_R_h"] for L in xs],
        "s-",
        label="ΔR_h (H-dir → refusal)",
    )
    plt.plot(
        xs,
        [per_layer[L]["delta_R_r"] for L in xs],
        "^-",
        label="ΔR_r (R-dir → refusal)",
    )
    plt.plot(
        xs,
        [per_layer[L]["delta_H_r"] for L in xs],
        "d-",
        label="ΔH_r (R-dir → harm prob)",
    )
    plt.axhline(0, color="k", lw=0.5)
    plt.xlabel("Injection layer")
    plt.ylabel("Delta")
    plt.title("LLaMA-3 AdvBench/Alpaca: Steering Effect Components\nvs Injection Layer")
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_component_deltas.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating component deltas plot: {e}")
    plt.close()

# Plot 4: Refusal rates on harmful/benign under steering
try:
    xs = inj_layers
    base_r_harm = per_layer[xs[0]]["harmful_eval"]["baseline"]["refusal_rate"]
    base_r_ben = per_layer[xs[0]]["benign_eval"]["baseline"]["refusal_rate"]
    plt.figure(figsize=(8, 5))
    plt.plot(
        xs,
        [per_layer[L]["harmful_eval"]["negR"]["refusal_rate"] for L in xs],
        "o-",
        label="RR harmful (-R) [jailbreak]",
    )
    plt.plot(
        xs,
        [per_layer[L]["harmful_eval"]["posR"]["refusal_rate"] for L in xs],
        "v-",
        label="RR harmful (+R)",
    )
    plt.plot(
        xs,
        [per_layer[L]["benign_eval"]["posR"]["refusal_rate"] for L in xs],
        "s-",
        label="RR benign (+R) [over-refuse]",
    )
    plt.axhline(
        base_r_harm,
        color="tab:blue",
        linestyle=":",
        alpha=0.6,
        label="baseline harmful RR",
    )
    plt.axhline(
        base_r_ben,
        color="tab:orange",
        linestyle=":",
        alpha=0.6,
        label="baseline benign RR",
    )
    plt.xlabel("Injection layer")
    plt.ylabel("Refusal rate")
    plt.title(
        "LLaMA-3 AdvBench/Alpaca: Refusal-Direction Steering\nEffect on Refusal Rates by Layer"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_refusal_rates_by_layer.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating refusal rates plot: {e}")
    plt.close()

# Plot 5: Harm probability under H-direction steering
try:
    xs = inj_layers
    base_h_harm = per_layer[xs[0]]["harmful_eval"]["baseline"]["harm_prob"]
    base_h_ben = per_layer[xs[0]]["benign_eval"]["baseline"]["harm_prob"]
    plt.figure(figsize=(8, 5))
    plt.plot(
        xs,
        [per_layer[L]["harmful_eval"]["posH"]["harm_prob"] for L in xs],
        "o-",
        label="HarmProb harmful (+H)",
    )
    plt.plot(
        xs,
        [per_layer[L]["harmful_eval"]["negH"]["harm_prob"] for L in xs],
        "s-",
        label="HarmProb harmful (-H)",
    )
    plt.plot(
        xs,
        [per_layer[L]["benign_eval"]["posH"]["harm_prob"] for L in xs],
        "^-",
        label="HarmProb benign (+H)",
    )
    plt.axhline(
        base_h_harm,
        color="tab:blue",
        linestyle=":",
        alpha=0.6,
        label="baseline harmful",
    )
    plt.axhline(
        base_h_ben,
        color="tab:orange",
        linestyle=":",
        alpha=0.6,
        label="baseline benign",
    )
    plt.xlabel("Injection layer")
    plt.ylabel("Mean harmfulness probability")
    plt.title(
        "LLaMA-3 AdvBench/Alpaca: Harmfulness-Direction Steering\nEffect on Harm Probability by Layer"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_harmprob_by_layer.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating harm prob plot: {e}")
    plt.close()

# Plot 6: Jailbreak signature - fraction non-refused + mean harm prob
try:
    xs = inj_layers
    fracs = [
        per_layer[L]["jailbreak_signature"]["num_non_refused_under_negR"]
        / max(1, per_layer[L]["jailbreak_signature"]["total_eval"])
        for L in xs
    ]
    hps = [
        per_layer[L]["jailbreak_signature"][
            "mean_harm_prob_when_negR_suppresses_refusal"
        ]
        or 0.0
        for L in xs
    ]
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(xs, fracs, "o-", color="tab:red", label="Frac non-refused (-R)")
    ax1.set_xlabel("Injection layer")
    ax1.set_ylabel("Fraction non-refused", color="tab:red")
    ax1.tick_params(axis="y", labelcolor="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(xs, hps, "s-", color="tab:blue", label="Mean harm prob (non-refused)")
    ax2.set_ylabel("Mean harm prob", color="tab:blue")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    plt.title(
        "LLaMA-3 AdvBench/Alpaca: Jailbreak Signature under -R Steering\nLeft axis: non-refusal fraction; Right axis: harm probability"
    )
    fig.tight_layout()
    plt.savefig(
        os.path.join(working_dir, "llama3_advbench_alpaca_jailbreak_signature.png"),
        bbox_inches="tight",
    )
    plt.close()
except Exception as e:
    print(f"Error creating jailbreak signature plot: {e}")
    plt.close()

# Print summary metrics
try:
    print("=== Summary: DDS by injection layer ===")
    for L in inj_layers:
        tag = ""
        if L == best_H:
            tag += " [best_H]"
        if L == best_R:
            tag += " [best_R]"
        print(
            f"  layer {L:>2d}: DDS = {per_layer[L]['direction_dissociation_score']:.4f}{tag}"
        )
    print(f"\ncosine(harm_dir, ref_dir) = {d['cosine_harm_refusal']:.4f}")
except Exception as e:
    print(f"Error printing summary: {e}")
