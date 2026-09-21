import matplotlib.pyplot as plt
import numpy as np
import os, re

working_dir = os.path.join(os.getcwd(), "working")
os.makedirs(working_dir, exist_ok=True)

try:
    experiment_data = np.load(
        os.path.join(working_dir, "experiment_data.npy"), allow_pickle=True
    ).item()
except Exception as e:
    print(f"Error loading experiment data: {e}")
    experiment_data = {}

root = experiment_data.get("steering_analysis", {}).get("r1_distill_llama_8b", {})
dose_response = root.get("dose_response", {})
bse_scores = root.get("bse_scores", {})
layer_ablation = root.get("layer_ablation", {})
interference = root.get("cross_interference", {})
baseline = root.get("baseline", {})
preds = root.get("predictions", [])
gts = root.get("ground_truth", [])
baseline_acc = baseline.get("accuracy", None)

# Plot 1: Dose-response behaviour count curves
try:
    plt.figure(figsize=(8, 5))
    for b, r in dose_response.items():
        cs = sorted([float(k) for k in r.keys()])
        ys = [r[str(c)]["target_mean"] for c in cs]
        plt.plot(cs, ys, "o-", label=b)
    plt.xlabel("Steering coefficient (× BASE_SCALE)")
    plt.ylabel("Mean target behaviour count in CoT")
    plt.title(
        "R1-Distill-Llama-8B (Math): Dose-Response Curves\nBehaviour frequency vs steering coefficient (mid layer)"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "r1_llama8b_dose_response_curves.png"))
    plt.close()
except Exception as e:
    print(f"Error creating dose-response plot: {e}")
    plt.close()

# Plot 2: Accuracy vs coefficient
try:
    plt.figure(figsize=(8, 5))
    for b, r in dose_response.items():
        cs = sorted([float(k) for k in r.keys()])
        ys = [r[str(c)]["acc"] for c in cs]
        plt.plot(cs, ys, "s-", label=b)
    if baseline_acc is not None:
        plt.axhline(
            baseline_acc, color="k", ls="--", label=f"baseline={baseline_acc:.2f}"
        )
    plt.xlabel("Steering coefficient")
    plt.ylabel("Task accuracy")
    plt.ylim(0, 1.05)
    plt.title(
        "R1-Distill-Llama-8B (Math): Task Accuracy vs Steering Coefficient\nPer behaviour steering vector"
    )
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(working_dir, "r1_llama8b_accuracy_vs_coefficient.png"))
    plt.close()
except Exception as e:
    print(f"Error creating accuracy plot: {e}")
    plt.close()

# Plot 3: BSE scores bar chart
try:
    if bse_scores:
        plt.figure(figsize=(7, 4))
        bs = list(bse_scores.keys())
        ys_a1 = [bse_scores[b]["bse_a1"] for b in bs]
        ys_a2 = [
            bse_scores[b]["bse_a2"] if bse_scores[b].get("bse_a2") is not None else 0.0
            for b in bs
        ]
        x = np.arange(len(bs))
        w = 0.35
        plt.bar(x - w / 2, ys_a1, w, label="BSE (α=1)", color="steelblue")
        plt.bar(x + w / 2, ys_a2, w, label="BSE (α=2)", color="orange")
        plt.xticks(x, bs, rotation=20)
        plt.ylabel("BSE score")
        plt.axhline(0, color="k", lw=0.5)
        plt.title(
            "R1-Distill-Llama-8B (Math): Behaviour Steering Efficacy\nComparison of α=1 vs α=2 coefficients"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "r1_llama8b_bse_scores.png"))
        plt.close()
except Exception as e:
    print(f"Error creating BSE plot: {e}")
    plt.close()

# Plot 4: Per-layer ablation for hedging
try:
    hedge_abl = layer_ablation.get("hedging", {})
    if hedge_abl:
        lis = sorted([int(k) for k in hedge_abl.keys()])
        deltas = [hedge_abl[str(li)]["delta"] for li in lis]
        bses = [hedge_abl[str(li)]["bse"] for li in lis]
        accs_pos = [hedge_abl[str(li)]["acc_pos"] for li in lis]
        accs_neg = [hedge_abl[str(li)]["acc_neg"] for li in lis]
        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(
            lis, deltas, "o-", color="steelblue", label="Δ freq (+coeff vs -coeff)"
        )
        ax1.plot(lis, bses, "s-", color="green", label="BSE")
        ax1.set_xlabel("Transformer layer index")
        ax1.set_ylabel("Δ / BSE")
        ax1.grid(True, alpha=0.3)
        ax2 = ax1.twinx()
        ax2.plot(lis, accs_pos, "^--", color="red", alpha=0.6, label="acc (+coeff)")
        ax2.plot(lis, accs_neg, "v--", color="purple", alpha=0.6, label="acc (-coeff)")
        ax2.set_ylabel("Accuracy")
        ax2.set_ylim(0, 1.05)
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8)
        plt.title(
            "R1-Distill-Llama-8B (Math): Per-Layer Steering Ablation for 'hedging'\nLeft axis: Δ/BSE, Right axis: accuracy"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "r1_llama8b_layer_ablation_hedging.png"))
        plt.close()
except Exception as e:
    print(f"Error creating layer ablation plot: {e}")
    plt.close()

# Plot 5: Cross-behaviour interference heatmap
try:
    if interference:
        src = list(interference.keys())
        tgt = sorted({t for r in interference.values() for t in r.keys()})
        M = np.array([[interference[s].get(t, 0.0) for t in tgt] for s in src])
        vmax = max(np.abs(M).max(), 1e-6)
        plt.figure(figsize=(6.5, 5.5))
        im = plt.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        plt.xticks(range(len(tgt)), tgt, rotation=30)
        plt.yticks(range(len(src)), src)
        plt.xlabel("Target behaviour (measured)")
        plt.ylabel("Source behaviour (steered)")
        plt.title(
            "R1-Distill-Llama-8B (Math): Cross-Behaviour Interference\nΔ = f(coeff=+1) - f(coeff=-1)"
        )
        for i in range(len(src)):
            for j in range(len(tgt)):
                plt.text(j, i, f"{M[i,j]:.1f}", ha="center", va="center", fontsize=8)
        plt.colorbar(im)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "r1_llama8b_cross_interference_heatmap.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating interference heatmap: {e}")
    plt.close()

# Plot 6: Baseline behaviour totals
try:
    bc = baseline.get("counts", {})
    if bc:
        plt.figure(figsize=(7, 4))
        behs = list(bc.keys())
        vals = [bc[b] for b in behs]
        plt.bar(behs, vals, color="teal")
        plt.ylabel("Total occurrences across baseline CoTs")
        plt.title(
            "R1-Distill-Llama-8B (Math): Baseline Behaviour Totals\nUnsteered chain-of-thought"
        )
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(
            os.path.join(working_dir, "r1_llama8b_baseline_behaviour_totals.png")
        )
        plt.close()
except Exception as e:
    print(f"Error creating baseline totals plot: {e}")
    plt.close()

# Plot 7: Per-task correctness for baseline
try:
    if preds and gts:

        def parse_num(p):
            try:
                s = re.sub(r"[^0-9\.\-]", "", str(p))
                return float(s) if s not in ("", "-", ".", "-.") else None
            except:
                return None

        flags = []
        for p, g in zip(preds, gts):
            v = parse_num(p)
            flags.append(1 if (v is not None and abs(v - float(g)) < 1e-3) else 0)
        plt.figure(figsize=(10, 3))
        idx = np.arange(len(flags))
        colors = ["green" if c == 1 else "red" for c in flags]
        plt.bar(idx, flags, color=colors)
        plt.xlabel("Task index")
        plt.ylabel("Correct (1) / Incorrect (0)")
        acc_str = f"{baseline_acc:.3f}" if baseline_acc is not None else "N/A"
        plt.title(
            f"R1-Distill-Llama-8B (Math): Baseline Per-Task Correctness (acc={acc_str})\nGreen=correct, Red=incorrect"
        )
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "r1_llama8b_baseline_correctness.png"))
        plt.close()
except Exception as e:
    print(f"Error creating correctness plot: {e}")
    plt.close()

# Plot 8: Comparison - target vs off-target behaviour change under steering
try:
    if dose_response:
        behs = list(dose_response.keys())
        target_deltas = []
        offtarget_deltas = []
        for b in behs:
            r = dose_response[b]
            if "1.0" in r and "-1.0" in r:
                pos_means = r["1.0"]["all_means"]
                neg_means = r["-1.0"]["all_means"]
                target_deltas.append(pos_means[b] - neg_means[b])
                off = [pos_means[k] - neg_means[k] for k in pos_means if k != b]
                offtarget_deltas.append(np.mean(off) if off else 0.0)
        x = np.arange(len(behs))
        w = 0.35
        plt.figure(figsize=(7, 4))
        plt.bar(
            x - w / 2, target_deltas, w, label="Target behaviour Δ", color="steelblue"
        )
        plt.bar(
            x + w / 2, offtarget_deltas, w, label="Mean off-target Δ", color="salmon"
        )
        plt.xticks(x, behs, rotation=20)
        plt.ylabel("Δ frequency (coeff=+1 vs -1)")
        plt.axhline(0, color="k", lw=0.5)
        plt.title(
            "R1-Distill-Llama-8B (Math): Target vs Off-Target Steering Effect\nHigher target Δ + lower off-target Δ = more selective steering"
        )
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(working_dir, "r1_llama8b_target_vs_offtarget.png"))
        plt.close()
except Exception as e:
    print(f"Error creating target vs off-target plot: {e}")
    plt.close()

# Print metrics
try:
    print(f"Baseline accuracy: {baseline_acc}")
    print(f"Baseline behaviour counts: {baseline.get('counts', {})}")
    for b, s in bse_scores.items():
        print(f"BSE[{b}]: α=1 -> {s['bse_a1']:.3f}, α=2 -> {s.get('bse_a2')}")
    for b, r in dose_response.items():
        print(f"Dose-response[{b}]:")
        for c in sorted(r.keys(), key=lambda x: float(x)):
            print(
                f"  coeff={c}: target_mean={r[c]['target_mean']:.2f}, acc={r[c]['acc']:.3f}"
            )
except Exception as e:
    print(f"Error printing metrics: {e}")
