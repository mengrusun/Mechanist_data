"""
Consolidated plots:

(1) Safety-vs-capability trade-off scatter: x = MMLU, y = mean cross-lingual ASR,
    color = surface vs bottleneck layer, size = |α|.

(2) Bar chart: ΔASR per tier for each condition, side by side.

(3) Per-language bar chart: baseline vs best-bottleneck vs best-surface.
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LANG_ORDER = ["en","zh","it","vi","ar","ko","th","bn","sw","jv"]
LANG_TIERS = {
    "en": "high", "zh": "high", "it": "high",
    "vi": "med", "ar": "med", "ko": "med",
    "th": "low", "bn": "low", "sw": "low", "jv": "low",
}


def main():
    # Load judge aggregate and MMLU
    agg = pd.read_csv("results/claim2_agg_judge.csv")
    with open("results/mmlu_capability.json") as f:
        mmlu = json.load(f)
    with open("results/mmlu_capability_extra.json") as f:
        mmlu += json.load(f)
    mmlu_df = pd.DataFrame(mmlu)

    # Map judge condition names to mmlu names
    def norm(name):
        return (name.replace("baseline_L-1_a0.0", "baseline")
                    .replace("_L14_a5.0", "").replace("_L14_a8.0", "")
                    .replace("_L14_a3.0", "").replace("_L9_a5.0", "").replace("_L9_a8.0", "")
                    .replace("_L28_a3.0", "").replace("_L28_a5.0", ""))
    agg["cond_short"] = agg["cond"].map(norm)

    # ---- (1) Safety-vs-capability trade-off ----
    per_cond = agg.groupby("cond_short").agg(mean_asr=("asr", "mean")).reset_index()
    per_cond["mmlu"] = per_cond["cond_short"].map(
        {r["name"]: r["acc"] for r in mmlu}
    )
    print(per_cond.to_string(index=False))

    def color(name):
        if "surface" in name: return "#d95f02"
        if "L9" in name: return "#7570b3"
        if "L14" in name: return "#1b9e77"
        return "#666666"
    def marker(name):
        if "surface" in name: return "s"
        if "L9" in name: return "^"
        if "L14" in name: return "o"
        return "*"

    fig, ax = plt.subplots(figsize=(7, 5))
    for _, r in per_cond.iterrows():
        if pd.isna(r["mmlu"]): continue
        ax.scatter(r["mmlu"] * 100, r["mean_asr"] * 100,
                   s=140, c=color(r["cond_short"]),
                   marker=marker(r["cond_short"]),
                   edgecolor="k", zorder=3)
        ax.annotate(r["cond_short"].replace("bottleneck_", "").replace("surface_", "srf_").replace("baseline", "base"),
                    (r["mmlu"] * 100, r["mean_asr"] * 100),
                    xytext=(6, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("MMLU accuracy (%)  →  more capability")
    ax.set_ylabel("Mean cross-lingual ASR (%)  →  less safe")
    ax.set_title("Safety–capability trade-off across steering conditions")
    ax.grid(alpha=0.3)
    # Legend
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#666", markersize=12, label="baseline"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#7570b3", markersize=10, label="bottleneck L9"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1b9e77", markersize=10, label="bottleneck L14"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#d95f02", markersize=10, label="surface L28"),
    ])
    plt.tight_layout()
    plt.savefig("results/claim2_safety_capability_tradeoff.png", dpi=150)
    print("[save] results/claim2_safety_capability_tradeoff.png")

    # ---- (2) ΔASR per tier bar chart ----
    baseline_row = agg[agg["cond_short"] == "baseline"]
    baseline_asr = {r["lang"]: r["asr"] for _, r in baseline_row.iterrows()}
    conds = sorted(agg["cond_short"].unique())
    conds = [c for c in conds if c != "baseline"]
    tiers = ("high", "med", "low")
    tier_delta = {c: {t: [] for t in tiers} for c in conds}
    for _, r in agg.iterrows():
        if r["cond_short"] == "baseline": continue
        d = baseline_asr.get(r["lang"], 0) - r["asr"]
        tier_delta[r["cond_short"]][LANG_TIERS[r["lang"]]].append(d)
    rows = []
    for c in conds:
        for t in tiers:
            arr = tier_delta[c][t]
            rows.append({"cond": c, "tier": t, "delta": float(np.mean(arr)) if arr else 0.0})
    tier_df = pd.DataFrame(rows).pivot(index="cond", columns="tier", values="delta")[list(tiers)]
    print("\n=== ΔASR per tier (positive = safer) ===")
    print(tier_df.round(3))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(tier_df))
    w = 0.25
    colors = {"high": "#1b9e77", "med": "#d95f02", "low": "#7570b3"}
    for i, t in enumerate(tiers):
        ax.bar(x + (i - 1) * w, tier_df[t].values, width=w, label=t, color=colors[t])
    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_xticks(x); ax.set_xticklabels(tier_df.index, rotation=25, ha="right")
    ax.set_ylabel("ΔASR vs baseline  (positive = safer)")
    ax.set_title("Safety improvement by resource-tier and intervention layer")
    ax.legend(title="lang tier"); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("results/claim2_delta_tiers.png", dpi=150)
    print("[save] results/claim2_delta_tiers.png")

    # ---- (3) Per-language bar chart: baseline vs best-bottleneck (L14 α=5) vs best-surface (L28 α=5) ----
    for row_name in ("bottleneck_L14_a5", "surface_L28_a5"):
        pass
    picks = ("baseline", "bottleneck_L14_a5", "bottleneck_L9_a5", "surface_L28_a5")
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x = np.arange(len(LANG_ORDER))
    w = 0.2
    pick_colors = {"baseline": "#666", "bottleneck_L14_a5": "#1b9e77",
                   "bottleneck_L9_a5": "#7570b3", "surface_L28_a5": "#d95f02"}
    for i, p in enumerate(picks):
        vals = [float(agg[(agg["cond_short"] == p) & (agg["lang"] == l)]["asr"].iloc[0])
                for l in LANG_ORDER]
        ax.bar(x + (i - 1.5) * w, vals, width=w, label=p, color=pick_colors[p])
    ax.set_xticks(x); ax.set_xticklabels(LANG_ORDER)
    ax.set_ylabel("ASR")
    ax.set_title("Attack success rate per language (LLM judge)")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig("results/claim2_per_lang.png", dpi=150)
    print("[save] results/claim2_per_lang.png")


if __name__ == "__main__":
    main()
