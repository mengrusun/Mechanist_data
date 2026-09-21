"""Aggregate results for all three claims and produce plots + report.md.
"""
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("/data/zhenqian/Reproduction1/cc/science/esmfold_mechanism")
OUT = BASE / "outputs"
FIG = BASE / "figures"
FIG.mkdir(parents=True, exist_ok=True)


def analyze_claim1(path=OUT / "claim1_patch.jsonl"):
    if not path.exists():
        return None, None
    layers = None
    stats = defaultdict(list)   # (metric, k) -> list of 0/1
    n = 0
    for ln in open(path):
        e = json.loads(ln)
        n += 1
        for k_str, r in e["layers"].items():
            k = int(k_str)
            for m in ("hp_after_s_patch_b2n", "hp_after_z_patch_b2n",
                      "hp_after_s_patch_n2b_reverse", "hp_after_z_patch_n2b_reverse"):
                v = r[m]
                if v >= 0:
                    stats[(m, k)].append(v)
    print(f"claim1 n_chains={n}")
    # summary table
    all_k = sorted({k for (_, k) in stats})
    summary = {"n_chains": n, "layers": all_k}
    for m in ("hp_after_s_patch_b2n", "hp_after_z_patch_b2n",
              "hp_after_s_patch_n2b_reverse", "hp_after_z_patch_n2b_reverse"):
        summary[m] = {k: float(np.mean(stats[(m, k)])) for k in all_k}
    # plot
    fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax[0].plot(all_k, [summary["hp_after_s_patch_b2n"][k] for k in all_k], "o-", label="patch s (broken←native): hairpin recovered")
    ax[0].plot(all_k, [summary["hp_after_z_patch_b2n"][k] for k in all_k], "s--", label="patch z (broken←native): hairpin recovered")
    ax[0].set_ylabel("Fraction with β-hairpin")
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].legend()
    ax[0].set_title("Claim 1 — restore hairpin by injecting native latents into broken run")
    ax[1].plot(all_k, [summary["hp_after_s_patch_n2b_reverse"][k] for k in all_k], "o-", label="patch s (native←broken): hairpin destroyed if 0")
    ax[1].plot(all_k, [summary["hp_after_z_patch_n2b_reverse"][k] for k in all_k], "s--", label="patch z (native←broken): hairpin destroyed if 0")
    ax[1].set_xlabel("Block index (after which patch is applied)")
    ax[1].set_ylabel("Fraction with β-hairpin")
    ax[1].set_ylim(-0.02, 1.02)
    ax[1].legend()
    ax[1].set_title("Reverse — inject broken latents into native run")
    plt.tight_layout()
    plt.savefig(FIG / "claim1_layer_sweep.png", dpi=140)
    plt.close(fig)
    (OUT / "claim1_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def analyze_claim2(path=OUT / "claim2_pathway.jsonl"):
    if not path.exists():
        return None
    windows = None
    per_win = defaultdict(lambda: {"s2p_hp": [], "p2s_hp": [], "s2p_plddt": [], "p2s_plddt": []})
    n = 0
    base_hp = []
    for ln in open(path):
        e = json.loads(ln)
        n += 1
        base_hp.append(e["base_hp"])
        for wname, r in e["windows"].items():
            per_win[wname]["s2p_hp"].append(r["s2p_hp"] if r["s2p_hp"] >= 0 else np.nan)
            per_win[wname]["p2s_hp"].append(r["p2s_hp"] if r["p2s_hp"] >= 0 else np.nan)
            per_win[wname]["s2p_plddt"].append(r["s2p_plddt"])
            per_win[wname]["p2s_plddt"].append(r["p2s_plddt"])
    print(f"claim2 n_chains={n}")
    summary = {"n_chains": n, "base_hp_frac": float(np.mean(base_hp))}
    for wname, d in per_win.items():
        summary[wname] = {
            "s2p_hp_frac": float(np.nanmean(d["s2p_hp"])),
            "p2s_hp_frac": float(np.nanmean(d["p2s_hp"])),
            "s2p_plddt": float(np.nanmean(d["s2p_plddt"])),
            "p2s_plddt": float(np.nanmean(d["p2s_plddt"])),
        }
    (OUT / "claim2_summary.json").write_text(json.dumps(summary, indent=2))
    # Plot two rows: 8-block windows (top) and 12-block windows (bottom).
    small_wins = ["0-7", "8-15", "16-23", "24-31", "32-39", "40-47"]
    big_wins = ["0-11", "12-23", "24-35", "36-47", "0-47"]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    for row, (wins, tag) in enumerate([(small_wins, "8-block windows"),
                                        (big_wins, "12-block windows (last: full trunk)")]):
        xs = np.arange(len(wins))
        y_s2p = [summary[w]["s2p_hp_frac"] for w in wins]
        y_p2s = [summary[w]["p2s_hp_frac"] for w in wins]
        ax[row, 0].bar(xs - 0.2, y_s2p, 0.4, label="ablate seq2pair")
        ax[row, 0].bar(xs + 0.2, y_p2s, 0.4, label="ablate pair2seq")
        ax[row, 0].axhline(summary["base_hp_frac"], color="gray", ls="--", label="baseline")
        ax[row, 0].set_xticks(xs)
        ax[row, 0].set_xticklabels(wins, rotation=25)
        ax[row, 0].set_ylabel("Fraction with β-hairpin")
        ax[row, 0].set_title(f"β-hairpin frequency — {tag}")
        ax[row, 0].set_ylim(-0.02, 1.05)
        ax[row, 0].legend()
        y_s2p_p = [summary[w]["s2p_plddt"] for w in wins]
        y_p2s_p = [summary[w]["p2s_plddt"] for w in wins]
        ax[row, 1].bar(xs - 0.2, y_s2p_p, 0.4, label="ablate seq2pair")
        ax[row, 1].bar(xs + 0.2, y_p2s_p, 0.4, label="ablate pair2seq")
        ax[row, 1].set_xticks(xs)
        ax[row, 1].set_xticklabels(wins, rotation=25)
        ax[row, 1].set_ylabel("mean pLDDT")
        ax[row, 1].set_title(f"pLDDT — {tag}")
        ax[row, 1].legend()
    plt.suptitle("Claim 2 — pathway ablation by block window")
    plt.tight_layout()
    plt.savefig(FIG / "claim2_pathway_ablation.png", dpi=140)
    plt.close(fig)
    return summary


def analyze_claim3(probe_path=OUT / "claim3_probe_acc.json", causal_path=OUT / "claim3_causal.jsonl"):
    summary = {}
    if probe_path.exists():
        acc = json.loads(probe_path.read_text())
        summary["probe_accuracy"] = acc
        ks = sorted(int(k) for k in acc.keys())
        ys = [acc[str(k)] for k in ks]
        plt.figure(figsize=(7, 4))
        plt.plot(ks, ys, "o-")
        plt.axhline(1/3, color="gray", ls="--", label="chance")
        plt.xlabel("Block index (in trunk)")
        plt.ylabel("Test accuracy — 3-way charge probe")
        plt.title("Claim 3 (probe) — linear separability of residue charge in s at each block")
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / "claim3_probe_acc.png", dpi=140)
        plt.close()
    if causal_path.exists():
        # aggregate: mean cross-strand Cα for opposite vs same charge steering per scale
        per_scale = defaultdict(lambda: {"opp_d": [], "same_d": [], "opp_hp": [], "same_hp": []})
        base_d = []
        base_hp = []
        for ln in open(causal_path):
            e = json.loads(ln)
            if e["base_d"] is not None:
                base_d.append(e["base_d"])
            base_hp.append(e["base_hp"])
            for s, r in e["results"].items():
                if r["opp_d"] is not None:
                    per_scale[s]["opp_d"].append(r["opp_d"])
                if r["same_d"] is not None:
                    per_scale[s]["same_d"].append(r["same_d"])
                if r["opp_hp"] >= 0:
                    per_scale[s]["opp_hp"].append(r["opp_hp"])
                if r["same_hp"] >= 0:
                    per_scale[s]["same_hp"].append(r["same_hp"])
        scales = sorted(per_scale.keys(), key=float)
        summary["base_d"] = float(np.mean(base_d)) if base_d else None
        summary["base_hp_frac"] = float(np.mean(base_hp)) if base_hp else None
        summary["per_scale"] = {}
        for s in scales:
            d = per_scale[s]
            summary["per_scale"][s] = {
                "opp_d_mean": float(np.mean(d["opp_d"])) if d["opp_d"] else None,
                "same_d_mean": float(np.mean(d["same_d"])) if d["same_d"] else None,
                "opp_hp_frac": float(np.mean(d["opp_hp"])) if d["opp_hp"] else None,
                "same_hp_frac": float(np.mean(d["same_hp"])) if d["same_hp"] else None,
                "n": len(d["opp_d"]),
            }
        # plot
        xs = [float(s) for s in scales]
        opp_d = [summary["per_scale"][s]["opp_d_mean"] for s in scales]
        same_d = [summary["per_scale"][s]["same_d_mean"] for s in scales]
        opp_h = [summary["per_scale"][s]["opp_hp_frac"] for s in scales]
        same_h = [summary["per_scale"][s]["same_hp_frac"] for s in scales]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(xs, opp_d, "o-", label="opposite-charge steering (i↔+, j↔-)")
        ax[0].plot(xs, same_d, "s-", label="same-charge steering (i,j↔+)")
        if summary["base_d"]:
            ax[0].axhline(summary["base_d"], color="gray", ls="--", label="baseline")
        ax[0].set_xlabel("steering scale α")
        ax[0].set_ylabel("cross-strand Cα distance (Å)")
        ax[0].set_title("Claim 3 (causal) — cross-strand Cα distance vs charge steering")
        ax[0].legend()
        ax[1].plot(xs, opp_h, "o-", label="opposite-charge steering")
        ax[1].plot(xs, same_h, "s-", label="same-charge steering")
        if summary["base_hp_frac"] is not None:
            ax[1].axhline(summary["base_hp_frac"], color="gray", ls="--", label="baseline")
        ax[1].set_xlabel("steering scale α")
        ax[1].set_ylabel("fraction with β-hairpin (DSSP)")
        ax[1].set_title("β-hairpin frequency vs charge steering")
        ax[1].legend()
        plt.tight_layout()
        plt.savefig(FIG / "claim3_causal_steering.png", dpi=140)
        plt.close(fig)
    (OUT / "claim3_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def analyze_claim3_mutation(path=OUT / "claim3_mutation.jsonl"):
    if not path.exists():
        return None
    # aggregate per variant
    by_var = defaultdict(lambda: {"d": [], "hp": [], "plddt": [], "d_pairs_wise": []})
    base_d = []
    base_hp = []
    for ln in open(path):
        e = json.loads(ln)
        if e["base_d"] is not None:
            base_d.append(e["base_d"])
        base_hp.append(e["base_hp"])
        for v, r in e["variants"].items():
            if r["d"] is not None:
                by_var[v]["d"].append(r["d"])
                # per-chain paired difference vs base
                by_var[v]["d_pairs_wise"].append((e["base_d"], r["d"]))
            if r["hp"] >= 0:
                by_var[v]["hp"].append(r["hp"])
            by_var[v]["plddt"].append(r["plddt"])
    summary = {
        "n_chains": len(base_hp),
        "base_d_mean": float(np.mean(base_d)) if base_d else None,
        "base_hp_frac": float(np.mean(base_hp)) if base_hp else None,
        "variants": {},
    }
    for v, d in by_var.items():
        # paired difference
        deltas = [x[1] - x[0] for x in d["d_pairs_wise"]]
        summary["variants"][v] = {
            "d_mean": float(np.mean(d["d"])) if d["d"] else None,
            "d_median": float(np.median(d["d"])) if d["d"] else None,
            "d_delta_vs_base_mean": float(np.mean(deltas)) if deltas else None,
            "d_delta_vs_base_median": float(np.median(deltas)) if deltas else None,
            "hp_frac": float(np.mean(d["hp"])) if d["hp"] else None,
            "plddt": float(np.mean(d["plddt"])) if d["plddt"] else None,
            "n": len(d["hp"]),
        }
    # Two-group comparison: opposite (KE + EK) vs same (KK + EE)
    opp_d = by_var["KE_opp"]["d"] + by_var["EK_opp"]["d"]
    same_d = by_var["KK_same_pos"]["d"] + by_var["EE_same_neg"]["d"]
    opp_hp = by_var["KE_opp"]["hp"] + by_var["EK_opp"]["hp"]
    same_hp = by_var["KK_same_pos"]["hp"] + by_var["EE_same_neg"]["hp"]
    summary["pooled"] = {
        "opp_d_mean": float(np.mean(opp_d)),
        "same_d_mean": float(np.mean(same_d)),
        "opp_hp_frac": float(np.mean(opp_hp)),
        "same_hp_frac": float(np.mean(same_hp)),
        "n_opp": len(opp_d), "n_same": len(same_d),
    }
    # paired Wilcoxon on per-chain (opp mean vs same mean)
    per_chain_opp = []
    per_chain_same = []
    for ln in open(path):
        e = json.loads(ln)
        opp = [e["variants"]["KE_opp"]["d"], e["variants"]["EK_opp"]["d"]]
        sam = [e["variants"]["KK_same_pos"]["d"], e["variants"]["EE_same_neg"]["d"]]
        opp = [x for x in opp if x is not None]
        sam = [x for x in sam if x is not None]
        if opp and sam:
            per_chain_opp.append(np.mean(opp))
            per_chain_same.append(np.mean(sam))
    from scipy.stats import wilcoxon
    try:
        w = wilcoxon(per_chain_opp, per_chain_same, alternative="less")
        summary["wilcoxon_opp_less_than_same"] = {"stat": float(w.statistic), "p": float(w.pvalue), "n": len(per_chain_opp)}
    except Exception as e:
        summary["wilcoxon_opp_less_than_same"] = {"error": str(e)}
    (OUT / "claim3_mutation_summary.json").write_text(json.dumps(summary, indent=2))

    # plot
    variants_order = ["KE_opp", "EK_opp", "KK_same_pos", "EE_same_neg"]
    labels = ["K/E (opp)", "E/K (opp)", "K/K (same +)", "E/E (same −)"]
    d_means = [summary["variants"][v]["d_mean"] for v in variants_order]
    hp_fracs = [summary["variants"][v]["hp_frac"] for v in variants_order]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    colors = ["tab:green", "tab:green", "tab:red", "tab:red"]
    ax[0].bar(labels, d_means, color=colors)
    ax[0].axhline(summary["base_d_mean"], color="gray", ls="--", label="baseline (native)")
    ax[0].set_ylabel("cross-strand Cα distance (Å)")
    ax[0].set_title("Claim 3 (mutation) — cross-strand distance vs charge configuration")
    ax[0].legend()
    ax[1].bar(labels, hp_fracs, color=colors)
    ax[1].axhline(summary["base_hp_frac"], color="gray", ls="--", label="baseline")
    ax[1].set_ylabel("fraction with β-hairpin")
    ax[1].set_title("β-hairpin frequency by charge configuration")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(FIG / "claim3_mutation.png", dpi=140)
    plt.close(fig)

    return summary


def analyze_claim3_steer_v2(path=OUT / "claim3_steer_v2.jsonl"):
    if not path.exists():
        return None
    per_scale = defaultdict(lambda: {"opp_d": [], "same_d": [], "opp_hp": [], "same_hp": []})
    base_d = []; base_hp = []
    for ln in open(path):
        e = json.loads(ln)
        if e["base_d"] is not None:
            base_d.append(e["base_d"])
        base_hp.append(e["base_hp"])
        for s, r in e["results"].items():
            if r["opp_d"] is not None:
                per_scale[s]["opp_d"].append(r["opp_d"])
            if r["same_d"] is not None:
                per_scale[s]["same_d"].append(r["same_d"])
            if r["opp_hp"] >= 0:
                per_scale[s]["opp_hp"].append(r["opp_hp"])
            if r["same_hp"] >= 0:
                per_scale[s]["same_hp"].append(r["same_hp"])
    scales = sorted(per_scale.keys(), key=float)
    summary = {
        "n_chains": len(base_hp),
        "base_d_mean": float(np.mean(base_d)) if base_d else None,
        "base_hp_frac": float(np.mean(base_hp)) if base_hp else None,
        "per_scale": {},
    }
    for s in scales:
        d = per_scale[s]
        summary["per_scale"][s] = {
            "opp_d_mean": float(np.mean(d["opp_d"])) if d["opp_d"] else None,
            "same_d_mean": float(np.mean(d["same_d"])) if d["same_d"] else None,
            "opp_hp_frac": float(np.mean(d["opp_hp"])) if d["opp_hp"] else None,
            "same_hp_frac": float(np.mean(d["same_hp"])) if d["same_hp"] else None,
            "n": len(d["opp_d"]),
        }
    (OUT / "claim3_steer_v2_summary.json").write_text(json.dumps(summary, indent=2))
    xs = [float(s) for s in scales]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(xs, [summary["per_scale"][s]["opp_d_mean"] for s in scales], "o-", label="opposite-charge steering")
    ax[0].plot(xs, [summary["per_scale"][s]["same_d_mean"] for s in scales], "s-", label="same-charge steering")
    if summary["base_d_mean"] is not None:
        ax[0].axhline(summary["base_d_mean"], color="gray", ls="--", label="baseline (broken)")
    ax[0].set_xlabel("steering scale α  (unit-normalized direction)")
    ax[0].set_ylabel("cross-strand Cα distance (Å)")
    ax[0].set_title("Claim 3 (steering v2) — Cα distance")
    ax[0].legend()
    ax[1].plot(xs, [summary["per_scale"][s]["opp_hp_frac"] for s in scales], "o-", label="opposite-charge steering")
    ax[1].plot(xs, [summary["per_scale"][s]["same_hp_frac"] for s in scales], "s-", label="same-charge steering")
    if summary["base_hp_frac"] is not None:
        ax[1].axhline(summary["base_hp_frac"], color="gray", ls="--", label="baseline (broken)")
    ax[1].set_xlabel("steering scale α")
    ax[1].set_ylabel("fraction with β-hairpin (DSSP)")
    ax[1].set_title("β-hairpin recovery")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(FIG / "claim3_steering_v2.png", dpi=140)
    plt.close(fig)
    return summary


def main():
    s1 = analyze_claim1()
    s2 = analyze_claim2()
    s3 = analyze_claim3()
    s3m = analyze_claim3_mutation()
    s3s = analyze_claim3_steer_v2()
    print("Claim 1 summary:", json.dumps(s1, indent=2) if s1 else "n/a")
    print("Claim 2 summary:", json.dumps(s2, indent=2) if s2 else "n/a")
    print("Claim 3 summary:", json.dumps(s3, indent=2) if s3 else "n/a")


if __name__ == "__main__":
    main()
