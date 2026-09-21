#!/usr/bin/env python3
"""Appendix figure — reliability by task category and system, split by judge.

For each top-level task category (belief, emotion, ...), compare the three systems
  scientist1 = Claude Code, scientist2 = Mechanist, scientist3 = AI-Scientist
on their final reliability total (总分, 0-100). Human- and LLM-judge scores are shown
in two stacked sub-panels so the judge cut and the 3-system comparison are both visible.

Data: final 总分 from the two score_stats tables.

Outputs
  ./source_data_category_system.csv   tidy: category, experiment, system, judge, total
  ./summary_category_system.csv       per (category, system, judge): n, mean, 95% CI
  ./subfigs/a_category_system.{svg,pdf,png,tiff}

LLM-judge source (default): claude-opus-5 re-judge
(llm_judge/_orchestrator/score_stats/llm_scores_opus5.csv). Human-judge
scores are unchanged.

A different LLM judge can be plotted without touching the defaults, e.g. the
GPT-5.6-sol re-judge:

  python make_category_system_figure.py \
      --llm-csv ../../../csv/judge_totals/llm_scores_gpt56sol.csv \
      --llm-label GPT-5.6-sol --suffix -gpt56sol
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLCONFIGDIR", str(OUT_DIR / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── editable-text settings (match main figure) ──────────────────────────────
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
mpl.rcParams.update(
    {
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 600,
        "hatch.linewidth": 0.45,  # thin texture lines, subtle at Nature scale
    }
)

SCORE_STATS = Path(__file__).resolve().parents[3]
HUMAN_CSV = SCORE_STATS / "csv/judge_totals/human_scores.csv"
LLM_CSV = SCORE_STATS / "csv/judge_totals/llm_scores_opus5.csv"
# Display name of the LLM judge and the filename suffix for its outputs; both are
# overridden by --llm-label / --suffix so a second judge can be rendered as an
# independent set of files instead of overwriting the default one.
LLM_LABEL = "LLM"
OUT_SUFFIX = ""
SCI2SYS = {"s1": "Claude Code", "s2": "Mechanist", "s3": "AI-Scientist"}
SCORE_COL = "总分(0-100)"

# Two outlier experiments dropped from the appendix figures.
EXCLUDE_EXPERIMENTS = {"circuit_breakers", "multi_lingual_reasoning"}

SYSTEMS = ["Claude Code", "AI-Scientist", "Mechanist"]
COLORS = {"Claude Code": "#7B7B7B", "AI-Scientist": "#4E79B8", "Mechanist": "#2E7D5B"}
LIGHT = {"Claude Code": "#D9D9D9", "AI-Scientist": "#C9D8EE", "Mechanist": "#D7E9DD"}
# per-system bar texture so the three series read apart in greyscale / low ink
HATCH = {"Claude Code": "...", "AI-Scientist": "///", "Mechanist": "\\\\\\"}
JUDGES = ["Human", "LLM"]

CATEGORIES = [
    "belief", "emotion", "feature_description", "multi-agent_safety",
    "multilingual", "multimodal", "reasoning", "safety", "science",
]
CAT_LABELS = {
    "belief": "Belief",
    "emotion": "Emotion",
    "feature_description": "Feature\ndescription",
    "multi-agent_safety": "Multi-agent\nsafety",
    "multilingual": "Multilingual",
    "multimodal": "Multimodal",
    "reasoning": "Reasoning",
    "safety": "Safety",
    "science": "Science",
}

# RNG is used only for bootstrap-CI resampling and point jitter — all scores are
# real observations from the score_stats tables (no simulated/demo data).
RNG = np.random.default_rng(20260721)


def bootstrap_ci(values: np.ndarray, n_boot: int = 4000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        v = float(values[0]) if values.size else float("nan")
        return v, v
    boot = RNG.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def load_tidy() -> pd.DataFrame:
    rows = []
    for judge, path in (("Human", HUMAN_CSV), (LLM_LABEL, LLM_CSV)):
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            system = SCI2SYS.get(str(r["sci"]).strip())
            if system is None or pd.isna(r[SCORE_COL]):
                continue
            if str(r["实验"]).strip() in EXCLUDE_EXPERIMENTS:
                continue
            rows.append(
                {
                    "category": str(r["类别"]).strip(),
                    "experiment": str(r["实验"]).strip(),
                    "system": system,
                    "judge": judge,
                    "total_score": round(float(r[SCORE_COL]), 2),
                }
            )
    return pd.DataFrame(rows)


def summarize(tidy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for judge in JUDGES:
        for cat in CATEGORIES:
            for system in SYSTEMS:
                vals = tidy[(tidy["judge"] == judge) & (tidy["category"] == cat)
                            & (tidy["system"] == system)]["total_score"].to_numpy()
                lo, hi = bootstrap_ci(vals)
                rows.append(
                    {
                        "judge": judge, "category": cat, "system": system,
                        "n_cells": int(vals.size),
                        "mean_score": round(float(vals.mean()), 2) if vals.size else np.nan,
                        "ci95_low": round(lo, 2), "ci95_high": round(hi, 2),
                    }
                )
    return pd.DataFrame(rows)


def draw_panel(ax: plt.Axes, tidy: pd.DataFrame, summary: pd.DataFrame,
               judge: str, show_legend: bool, show_xticklabels: bool) -> None:
    x = np.arange(len(CATEGORIES))
    offsets = np.array([-0.23, 0.0, 0.23])  # shared bar geometry with main figure
    width = 0.19
    jrng = np.random.default_rng(7)

    for i, system in enumerate(SYSTEMS):
        for xi, cat in zip(x, CATEGORIES):
            row = summary[(summary["judge"] == judge) & (summary["category"] == cat)
                          & (summary["system"] == system)].iloc[0]
            xpos = xi + offsets[i]
            yerr = np.array(
                [[row["mean_score"] - row["ci95_low"]], [row["ci95_high"] - row["mean_score"]]]
            )
            ax.bar(
                xpos, row["mean_score"], width=width,
                color=LIGHT[system], edgecolor=COLORS[system], linewidth=0.8, zorder=2,
                hatch=HATCH[system],
                label=system if xi == 0 else None,
            )
            ax.errorbar(
                xpos, row["mean_score"], yerr=yerr, fmt="none",
                ecolor=COLORS[system], elinewidth=0.75, capsize=2.0, zorder=4,
            )
            pts = tidy[(tidy["judge"] == judge) & (tidy["category"] == cat)
                       & (tidy["system"] == system)]["total_score"].to_numpy()
            jitter = jrng.normal(0, 0.026, size=pts.size)
            ax.scatter(
                np.full(pts.size, xpos) + jitter, pts, s=4.0,
                color=COLORS[system], alpha=0.5,
                edgecolors="white", linewidths=0.18, zorder=3,
            )

    ax.set_ylabel("Reliability score (%)")
    ax.set_ylim(0, 102)
    ax.set_xlim(-0.55, len(CATEGORIES) - 0.45)
    ax.set_xticks(x)
    if show_xticklabels:
        ax.set_xticklabels([CAT_LABELS[c] for c in CATEGORIES])
        # sharex hides the upper panel's labels by default — force them back on
        ax.tick_params(labelbottom=True)
    else:
        ax.set_xticklabels([])
    ax.set_yticks(range(0, 101, 20))
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(length=2.2, width=0.6, color="#333333", pad=2)
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.55, zorder=0)
    ax.set_title(f"{judge} judge", loc="left", pad=5, fontweight="bold")
    if show_legend:
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.005), handlelength=1.1,
                  handletextpad=0.5, borderaxespad=0.0, ncol=3, columnspacing=1.2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--llm-csv", default=str(LLM_CSV),
                   help="score_stats CSV for the LLM judge (default: opus-5 re-judge)")
    p.add_argument("--llm-label", default=LLM_LABEL,
                   help="panel title / judge name for the LLM row (default: LLM)")
    p.add_argument("--suffix", default=OUT_SUFFIX,
                   help="suffix appended to every output filename (default: none)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    global LLM_CSV, LLM_LABEL, OUT_SUFFIX, JUDGES
    args = parse_args(argv)
    LLM_CSV = Path(args.llm_csv)
    LLM_LABEL = args.llm_label
    OUT_SUFFIX = args.suffix
    JUDGES = ["Human", LLM_LABEL]

    tidy = load_tidy()
    summary = summarize(tidy)
    tidy.to_csv(OUT_DIR / f"source_data_category_system{OUT_SUFFIX}.csv", index=False)
    summary.to_csv(OUT_DIR / f"summary_category_system{OUT_SUFFIX}.csv", index=False)

    # A4 width = 210 mm = 8.27 in. Fixed margins (no tight bbox) so the saved
    # canvas is exactly A4-wide.
    # Width matches main_fig_1 (7.2 in, Nature double-column); plot fills the full
    # canvas width (no side whitespace) — scaled in LaTeX via width=0.9\linewidth.
    # Shared left/right margins with make_macro_dimension_figure.py so both align.
    fig, (ax_h, ax_l) = plt.subplots(2, 1, figsize=(7.2, 5.1), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.066, right=0.996, top=0.945, bottom=0.095, hspace=0.26)
    draw_panel(ax_h, tidy, summary, "Human", show_legend=True, show_xticklabels=True)
    draw_panel(ax_l, tidy, summary, LLM_LABEL, show_legend=False, show_xticklabels=True)

    panel_dir = OUT_DIR / "subfigs"
    panel_dir.mkdir(parents=True, exist_ok=True)
    stem = panel_dir / f"reliability-by-area{OUT_SUFFIX}"
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=300)
    try:
        fig.savefig(f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    except TypeError:
        fig.savefig(f"{stem}.tiff", dpi=600)
    plt.close(fig)

    print("judge  category            system         n   mean")
    for _, r in summary.iterrows():
        print(f"  {r['judge']:<5} {r['category']:<18} {r['system']:<13} "
              f"{r['n_cells']:>2}  {r['mean_score']}")
    print(f"\nwrote {stem}.{{svg,pdf,png,tiff}}")


if __name__ == "__main__":
    main()
