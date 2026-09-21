#!/usr/bin/env python3
"""Appendix figure — reliability grouped into four workflow macro-dimensions by system.

The audit sub-dimensions of reliablity_judge.md are regrouped along the reproduction
workflow. Sub-dimension 9 is split into its three parts (9a method / 9b experiment
detail / 9c result consistency), and every sub-dimension — including 9a/9b/9c — carries
equal weight within its macro-dimension:

  Data usage           (d1, d4):                    data split, data scale
                                                    -> is the right data used, at scale?
  Experiment design    (d2, d6, 9a, 9b):            ground-truth/label validity, causal-
                                                    claim design, mechanistic-method choice,
                                                    other design detail (layers, coeffs,
                                                    hyper-params, controls, metric choice)
                                                    -> is the experiment soundly designed?
  Experiment execution (d3, d8):                    use of user-specified resources
                                                    (datasets, models), cross-artifact
                                                    consistency
                                                    -> was it executed faithfully?
  Result analysis      (d5, d7, 9c):                statistical rigor, result provenance /
                                                    no fabrication, agreement with the paper
                                                    (or a sound reason for divergence)
                                                    -> are the results honest and consistent?

Each macro-dimension score = mean of its (applicable, non-n/a) sub-dimension scores,
rescaled from the 1-5 rubric to 0-100%. Human and LLM judges are shown in two side-by-
side panels (matching fig. a); the 3 systems are compared with the main figure's colours.

Data: d1..d8 and dim9_method (9a) / dim9_exp (9b) / dim9_result (9c) columns of the two
score_stats tables.
Outputs
  ./source_data_macro_dims.csv     tidy: experiment, system, judge, macro_dim, score
  ./summary_macro_dims.csv         per (judge, macro_dim, system): n, mean, 95% CI
  ./subfigs/b_macro_dimensions.{svg,pdf,png,tiff}

LLM-judge source (default): claude-opus-5 re-judge
(llm_judge/_orchestrator/score_stats/llm_scores_opus5.csv). Human-judge
scores are unchanged.

A different LLM judge can be plotted without touching the defaults, e.g. the
GPT-5.6-sol re-judge:

  python make_macro_dimension_figure.py \
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
from matplotlib.patches import Patch

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

# Two outlier experiments dropped from the appendix figures.
EXCLUDE_EXPERIMENTS = {"circuit_breakers", "multi_lingual_reasoning"}

# main-figure system palette (fill = light, edge/points = dark)
SYSTEMS = ["Claude Code", "AI-Scientist", "Mechanist"]
COLORS = {"Claude Code": "#7B7B7B", "AI-Scientist": "#4E79B8", "Mechanist": "#2E7D5B"}
LIGHT = {"Claude Code": "#D9D9D9", "AI-Scientist": "#C9D8EE", "Mechanist": "#D7E9DD"}
# per-system bar texture so the three series read apart in greyscale / low ink
HATCH = {"Claude Code": "...", "AI-Scientist": "///", "Mechanist": "\\\\\\"}
JUDGES = ["Human", "LLM"]

# macro-dimension -> sub-dimension columns (all on the 1-5 rubric, equal weight each).
# 9a=dim9_method, 9b=dim9_exp, 9c=dim9_result are treated as ordinary sub-dimensions.
MACRO = [
    ("Data\nusage", ["d1", "d4"]),
    ("Experiment\ndesign", ["d2", "d6", "dim9_method", "dim9_exp"]),
    ("Experiment\nexecution", ["d3", "d8"]),
    ("Result\nanalysis", ["d5", "d7", "dim9_result"]),
]
MACRO_LABELS = [m[0] for m in MACRO]

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
    """One row per (experiment, system, judge, macro_dim) with score in 0-100%."""
    rows = []
    for judge, path in (("Human", HUMAN_CSV), (LLM_LABEL, LLM_CSV)):
        df = pd.read_csv(path)
        for _, r in df.iterrows():
            system = SCI2SYS.get(str(r["sci"]).strip())
            if system is None:
                continue
            if str(r["实验"]).strip() in EXCLUDE_EXPERIMENTS:
                continue
            for label, cols in MACRO:
                vals = [r[c] for c in cols if c in r and not pd.isna(r[c])]
                if not vals:
                    continue
                score_pct = float(np.mean(vals)) / 5.0 * 100.0  # 1-5 rubric -> %
                rows.append(
                    {
                        "experiment": str(r["实验"]).strip(),
                        "system": system,
                        "judge": judge,
                        "macro_dim": label.replace("\n", " "),
                        "score": round(score_pct, 2),
                    }
                )
    return pd.DataFrame(rows)


def summarize(tidy: pd.DataFrame) -> pd.DataFrame:
    """Per (macro_dim, system, judge)."""
    rows = []
    for judge in JUDGES:
        for label in MACRO_LABELS:
            key = label.replace("\n", " ")
            for system in SYSTEMS:
                vals = tidy[(tidy["judge"] == judge) & (tidy["macro_dim"] == key)
                            & (tidy["system"] == system)]["score"].to_numpy()
                lo, hi = bootstrap_ci(vals)
                rows.append(
                    {
                        "judge": judge, "macro_dim": key, "system": system,
                        "n_cells": int(vals.size),
                        "mean_score": round(float(vals.mean()), 2) if vals.size else np.nan,
                        "ci95_low": round(lo, 2), "ci95_high": round(hi, 2),
                    }
                )
    return pd.DataFrame(rows)


def draw_panel(ax: plt.Axes, tidy: pd.DataFrame, summary: pd.DataFrame,
               judge: str, show_ylabel: bool) -> None:
    x = np.arange(len(MACRO_LABELS))
    offsets = np.array([-0.23, 0.0, 0.23])  # shared bar geometry with main figure
    width = 0.19

    for i, system in enumerate(SYSTEMS):
        for xi, label in zip(x, MACRO_LABELS):
            key = label.replace("\n", " ")
            row = summary[(summary["judge"] == judge) & (summary["macro_dim"] == key)
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

    if show_ylabel:
        ax.set_ylabel("Reliability score (%)")
    ax.set_ylim(0, 102)
    ax.set_xlim(-0.55, len(MACRO_LABELS) - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels(MACRO_LABELS)
    ax.set_yticks(range(0, 101, 20))
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(length=2.2, width=0.6, color="#333333", pad=2)
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.55, zorder=0)
    ax.set_title(f"{judge} judge", loc="left", pad=5, fontweight="bold")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--llm-csv", default=str(LLM_CSV),
                   help="score_stats CSV for the LLM judge (default: opus-5 re-judge)")
    p.add_argument("--llm-label", default=LLM_LABEL,
                   help="panel title / judge name for the LLM panel (default: LLM)")
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
    tidy.to_csv(OUT_DIR / f"source_data_macro_dims{OUT_SUFFIX}.csv", index=False)
    summary.to_csv(OUT_DIR / f"summary_macro_dims{OUT_SUFFIX}.csv", index=False)

    # Width matches main_fig_1 (7.2 in, Nature double-column); plot fills the full
    # canvas width (no side whitespace) with the SAME left/right margins as
    # make_category_system_figure.py, fixed margins, no tight bbox.
    fig, (ax_h, ax_l) = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    fig.subplots_adjust(left=0.066, right=0.996, top=0.86, bottom=0.15, wspace=0.08)
    draw_panel(ax_h, tidy, summary, "Human", show_ylabel=True)
    draw_panel(ax_l, tidy, summary, LLM_LABEL, show_ylabel=False)

    # Single legend at the top-right (matching reliability-by-area) — placed above
    # the panels so it clears both judge titles at this width.
    handles = [Patch(facecolor=LIGHT[s], edgecolor=COLORS[s], linewidth=0.8,
                     hatch=HATCH[s], label=s)
               for s in SYSTEMS]
    fig.legend(handles=handles, loc="upper right", bbox_to_anchor=(0.996, 1.0),
               ncol=3, frameon=False, handlelength=1.1, handletextpad=0.5,
               columnspacing=1.4)

    panel_dir = OUT_DIR / "subfigs"
    panel_dir.mkdir(parents=True, exist_ok=True)
    stem = panel_dir / f"reliability-by-dimension{OUT_SUFFIX}"
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=300)
    try:
        fig.savefig(f"{stem}.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    except TypeError:
        fig.savefig(f"{stem}.tiff", dpi=600)
    plt.close(fig)

    print("macro_dim                system         n   mean  95% CI")
    for _, r in summary.iterrows():
        print(f"  {r['macro_dim']:<22} {r['system']:<13} {r['n_cells']:>2}  "
              f"{r['mean_score']:>5}  [{r['ci95_low']}, {r['ci95_high']}]")
    print(f"\nwrote {stem}.{{svg,pdf,png,tiff}}")
    print(f"wrote {OUT_DIR/f'source_data_macro_dims{OUT_SUFFIX}.csv'}")
    print(f"wrote {OUT_DIR/f'summary_macro_dims{OUT_SUFFIX}.csv'}")


if __name__ == "__main__":
    main()
