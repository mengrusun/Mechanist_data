"""
Analyze Claim-2 results.

Reads either a keyword-scored CSV (from 07_refusal_scorer) or a judge-scored
jsonl (from 08_llm_judge), computes per-condition per-language ASR, and produces
a summary heatmap + cross-lingual transfer plot.
"""

import os, json, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


LANG_ORDER = ["en","zh","it","vi","ar","ko","th","bn","sw","jv"]
LANG_TIERS = {
    "en": "high", "zh": "high", "it": "high",
    "vi": "med", "ar": "med", "ko": "med",
    "th": "low", "bn": "low", "sw": "low", "jv": "low",
}
TIER_COLORS = {"high": "#1b9e77", "med": "#d95f02", "low": "#7570b3"}


def load_judge_or_keyword(jsonl_glob=None, csv_summary=None):
    """Prefer judge if provided; else use the keyword csv (pre-aggregated)."""
    if jsonl_glob:
        rows = []
        import glob
        for p in glob.glob(jsonl_glob):
            with open(p) as f:
                for ln in f:
                    r = json.loads(ln)
                    rows.append(r)
        df = pd.DataFrame(rows)
        # ASR = 1 if judge_label == HARMFUL else 0
        df["asr"] = (df["judge_label"] == "HARMFUL").astype(int)
        df["cond"] = df["_file"].str.replace("gen_", "", regex=False).str.replace(".jsonl", "", regex=False)
        agg = df.groupby(["cond", "lang"]).agg(
            n=("asr", "size"), asr=("asr", "mean"),
            harmful=("judge_label", lambda s: (s == "HARMFUL").mean()),
            refused=("judge_label", lambda s: (s == "REFUSED").mean()),
            ambiguous=("judge_label", lambda s: (s == "AMBIGUOUS").mean()),
        ).reset_index()
        return agg, "judge"
    if csv_summary:
        df = pd.read_csv(csv_summary)
        df["cond"] = df["file"].str.replace("gen_", "", regex=False).str.replace(".jsonl", "", regex=False)
        return df, "keyword"
    raise ValueError("Need either --jsonl or --csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl_judge", default=None)
    ap.add_argument("--csv_keyword", default=None)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.jsonl_judge:
        agg, mode = load_judge_or_keyword(jsonl_glob=args.jsonl_judge)
    else:
        agg, mode = load_judge_or_keyword(csv_summary=args.csv_keyword)

    # Save aggregate
    agg.to_csv(os.path.join(args.out_dir, f"claim2_agg_{mode}.csv"), index=False)
    print(agg.to_string(index=False))

    # Wide format: rows=conditions, cols=languages
    conds = sorted(agg["cond"].unique())
    langs = [l for l in LANG_ORDER if l in set(agg["lang"])]
    mat = np.zeros((len(conds), len(langs)))
    for i, c in enumerate(conds):
        for j, l in enumerate(langs):
            v = agg[(agg["cond"] == c) & (agg["lang"] == l)]["asr"]
            if len(v):
                mat[i, j] = float(v.iloc[0])
    print("\n=== ASR heatmap (rows=condition, cols=lang) ===")
    print(pd.DataFrame(mat, index=conds, columns=langs).round(3))

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(1.2 * len(langs) + 3, 0.7 * len(conds) + 2))
    im = ax.imshow(mat, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(len(langs))); ax.set_yticks(range(len(conds)))
    ax.set_xticklabels(langs); ax.set_yticklabels(conds)
    ax.set_xlabel("target language")
    ax.set_ylabel("condition")
    ax.set_title(f"ASR heatmap  [{mode}-scored]")
    for i in range(len(conds)):
        for j in range(len(langs)):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    color="white" if mat[i, j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"claim2_asr_heatmap_{mode}.png"), dpi=150)
    print(f"[save] {args.out_dir}/claim2_asr_heatmap_{mode}.png")

    # Cross-lingual transfer per condition:
    # For each non-baseline condition, compute (baseline_asr - cond_asr) per lang.
    if any(c.startswith("baseline") for c in conds):
        base_row = [c for c in conds if c.startswith("baseline")][0]
        base_idx = conds.index(base_row)
        delta = mat[base_idx] - mat  # positive => improvement (lower ASR)
        fig, ax = plt.subplots(figsize=(1.2 * len(langs) + 3, 0.7 * (len(conds)-1) + 2))
        non_baseline = [c for c in conds if not c.startswith("baseline")]
        delta_nb = np.stack([mat[base_idx] - mat[conds.index(c)] for c in non_baseline])
        im = ax.imshow(delta_nb, cmap="Blues", vmin=-0.3, vmax=0.6)
        ax.set_xticks(range(len(langs))); ax.set_yticks(range(len(non_baseline)))
        ax.set_xticklabels(langs); ax.set_yticklabels(non_baseline)
        ax.set_xlabel("target language"); ax.set_ylabel("condition")
        ax.set_title("ASR reduction vs baseline  (higher = safer)")
        for i in range(len(non_baseline)):
            for j in range(len(langs)):
                ax.text(j, i, f"{delta_nb[i, j]:+.2f}", ha="center", va="center",
                        color="white" if abs(delta_nb[i, j]) > 0.35 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.03)
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"claim2_asr_delta_{mode}.png"), dpi=150)
        print(f"[save] {args.out_dir}/claim2_asr_delta_{mode}.png")

        # Report per-condition high/med/low averages
        tier_avg = {c: {} for c in non_baseline}
        for c in non_baseline:
            for tier in ("high", "med", "low"):
                keep = [j for j, l in enumerate(langs) if LANG_TIERS[l] == tier]
                if keep:
                    tier_avg[c][tier] = float(delta_nb[non_baseline.index(c), keep].mean())
        print("\n=== ΔASR by language tier ===")
        print(pd.DataFrame(tier_avg).round(3).T)

if __name__ == "__main__":
    main()
