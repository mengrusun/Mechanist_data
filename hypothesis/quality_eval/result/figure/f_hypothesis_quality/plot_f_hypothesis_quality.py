#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Panel f - hypothesis quality by metric, standalone and self-contained.

Everything the panel needs lives in this folder: the per-metric summary in
`data/summary_hypothesis_quality.csv` and the drawing code below. No import
from, and no path into, any other directory.

Provenance of the numbers
    `data/summary_hypothesis_quality.csv` is built by `data/make_summary.py`
    from the three `result/test_group_*` directories: 8 subtopics x 10 claims
    = 80 hypotheses per system, each scored by three independent LLM-judge runs that are
    averaged per hypothesis before bootstrapping. Raw judge scores are 0-10
    integers; the panel is on a 0-100 scale (x10). Impact is the mean of its
    Significance and Reach sub-scores, Testability the mean of Clarity and
    Feasibility. One row per (system, metric): mean_score, ci95_low, ci95_high
    with bootstrap 95% CIs over hypotheses.

What this reproduces
    The shipped panel is a plain grouped bar chart with one-decimal labels, no
    legend, and a zoomed y-axis that starts at 45.

Output: f_hypothesis_quality.* and f.* next to this script, 600 dpi.
"""

from __future__ import annotations

import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Keep the font cache beside the script so a first run never writes to $HOME.
os.environ.setdefault("MPLCONFIGDIR", str(HERE / ".mplconfig"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA = HERE / "data" / "summary_hypothesis_quality.csv"
OUTPUT_STEMS = [HERE / "f_hypothesis_quality", HERE / "f"]

SYSTEMS = ["Claude Code", "AI-Scientist", "Mechanist"]
METRICS = ["Novelty", "Impact", "Testability"]
COLORS = {"Claude Code": "#7B7B7B", "AI-Scientist": "#4E79B8", "Mechanist": "#2E7D5B"}
LIGHT_COLORS = {"Claude Code": "#D9D9D9", "AI-Scientist": "#C9D8EE", "Mechanist": "#D7E9DD"}

# Panel geometry: the 2.8 x 2.3 in panel of the composite figure, scaled to 0.8
# with the point-based type sizes left alone, so every label reads larger
# relative to the panel when the standalone figure is enlarged on a slide.
FIGSIZE = (2.8 * 0.8, 2.3 * 0.8)
DPI = 600
BAR_WIDTH = 0.19
OFFSETS = np.array([-0.23, 0.0, 0.23])   # one slot per system, in x units
BAR_EDGE_LW = 0.8
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",     # keep SVG text editable
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": DPI,
    }
)


def tidy_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(length=2.2, width=0.6, color="#333333", pad=2)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#E7E7E7", linewidth=0.55, zorder=0)


def draw_bars(ax: plt.Axes, summary: pd.DataFrame) -> None:
    x_base = np.arange(len(METRICS))
    for i, system in enumerate(SYSTEMS):
        rows = [
            summary[(summary["system"] == system) & (summary["metric"] == metric)].iloc[0]
            for metric in METRICS
        ]
        means = np.array([r["mean_score"] for r in rows], dtype=float)
        # errorbar wants distances from the mean, not absolute CI bounds.
        lows = means - np.array([r["ci95_low"] for r in rows], dtype=float)
        highs = np.array([r["ci95_high"] for r in rows], dtype=float) - means
        x = x_base + OFFSETS[i]

        ax.bar(
            x,
            means,
            width=BAR_WIDTH,
            color=LIGHT_COLORS[system],
            edgecolor=COLORS[system],
            linewidth=BAR_EDGE_LW,
            zorder=2,
        )
        ax.errorbar(
            x,
            means,
            yerr=np.array([lows, highs]),
            fmt="none",
            ecolor=COLORS[system],
            elinewidth=0.75,
            capsize=2.0,
            zorder=3,
        )
        for xi, y, hi in zip(x, means, highs):
            # Rotated: a horizontal "71.9" is wider than the 0.23 bar pitch, so
            # one-decimal labels on neighbouring bars would collide. Anchored
            # above the whisker rather than the bar top, to clear the CI cap.
            ax.text(
                xi,
                y + hi + 1.6,
                f"{y:.1f}",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=5.8,
                color=COLORS[system],
            )

    ax.set_ylabel("Score (%)")
    # Zoomed baseline: every mean sits between 51 and 78, so a 0-based axis
    # would compress the differences the panel is about. The top clears the
    # rotated value label, which is anchored above the upper CI cap.
    ax.set_ylim(45, 90)
    ax.set_xticks(x_base)
    ax.set_xticklabels(METRICS)
    tidy_axis(ax)


def main() -> None:
    summary = pd.read_csv(DATA)
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=False)
    ax = fig.add_subplot(1, 1, 1)
    draw_bars(ax, summary)
    for stem in OUTPUT_STEMS:
        for ext in ("png", "svg", "pdf"):
            fig.savefig(f"{stem}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    n = int(summary["n_hypotheses"].iloc[0])
    print(f"panel f: {len(SYSTEMS)} systems x {len(METRICS)} metrics, n = {n} hypotheses each")
    print(f"  wrote f_hypothesis_quality.* and f.* at {DPI} dpi")


if __name__ == "__main__":
    main()
