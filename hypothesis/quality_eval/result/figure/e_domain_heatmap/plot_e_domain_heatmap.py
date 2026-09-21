#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Panel e - mean hypothesis score by domain x system, standalone.

Everything the panel needs lives in this folder: the per-claim LLM-judge scores
in `data/source_data_hypotheses.csv` and the drawing code below. No import from,
and no path into, any other directory.

Provenance of the numbers
    `data/source_data_hypotheses.csv` is the source-data table exported by the
    current evaluation results: one row per scored hypothesis, 80 hypotheses
    per system across four domains (240 rows). `composite_score` is the mean of the three
    big dimensions (novelty / impact / testability), each itself the mean of its
    three 1-10 sub-scores x10, so everything is on a 0-100 scale. This script
    only averages those per-claim composites within each (domain, system) cell -
    no rescaling, no reweighting.

Output: e_domain_heatmap.png / .svg / .pdf next to this script, 600 dpi.
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
from matplotlib.colors import LinearSegmentedColormap

DATA = HERE / "data" / "source_data_hypotheses.csv"
STEM = HERE / "e_domain_heatmap"

SYSTEMS = ["Claude Code", "AI-Scientist", "Mechanist"]
DOMAINS = ["Knowledge", "Language", "Safety", "Science"]
# Two-line column heads: the full names do not fit the 3-column grid side by
# side at 6.5 pt, and wrapping beats shrinking the type below the tick size.
SYSTEM_LABELS = ["Claude\nCode", "AI-\nScientist", "Our\nMechanist"]

# Panel geometry: the 2.8 x 2.3 in panel of the composite figure, scaled to 0.8
# with the point-based type sizes left alone, so every label reads larger
# relative to the panel when the standalone figure is enlarged on a slide.
FIGSIZE = (2.8 * 0.8, 2.3 * 0.8)
DPI = 600
# Fixed colour range, not data-driven: 55-85 is the range the whole figure's
# score panels share, so a cell's colour means the same thing in every panel.
VMIN, VMAX = 55, 80

# Colour ramp stops as (position in 0-1 across VMIN..VMAX, colour).  The
# spacing is non-uniform because the observed cells only occupy 60-75: on a
# linear ramp every system sits in the pale middle of the scale and the gap
# between AI-Scientist and Mechanist barely registers.  Here the low half stays
# light and moves slowly, then the ramp picks up from ~66 on, which lifts the
# Mechanist column about one step in tone above the baselines without turning
# it into a dark block - the ordering is meant to be legible, not shouted.  The
# darkest stop is the same #2E7D5B the panel used before, reached only at VMAX,
# so the highest cell (74.4) still lands on a mid green, and the colourbar spans
# the 55-80 range shared with the other score panels.
RAMP = [
    (0.00, "#FAFBF9"),
    (0.28, "#F1F5F2"),
    (0.46, "#E2EBE5"),
    (0.56, "#CBDFD4"),
    (0.64, "#A9CDB8"),
    (0.72, "#7FB79B"),
    (0.82, "#56A07E"),
    (0.92, "#3A8963"),
    (1.00, "#2E7D5B"),
]

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


def load_grid() -> pd.DataFrame:
    """Mean composite score per (domain, system), in the figure's fixed order."""
    claims = pd.read_csv(DATA)
    return (
        claims.groupby(["domain", "system"], sort=False)["composite_score"]
        .mean()
        .unstack("system")
        .loc[DOMAINS, SYSTEMS]
    )


def text_colour(cmap, value: float) -> str:
    """Black or white digits, picked from the luminance of the cell's own fill.

    The ramp is non-uniform, so a fixed score threshold would drift every time
    a stop moves; asking the colormap what colour the cell actually got keeps
    the in-cell numbers readable no matter how RAMP is retuned.
    """
    r, g, b, _ = cmap((value - VMIN) / (VMAX - VMIN))
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.55 else "#222222"


def draw_heatmap(ax: plt.Axes, heat: pd.DataFrame) -> None:
    cmap = LinearSegmentedColormap.from_list("quality", RAMP)
    im = ax.imshow(heat.to_numpy(), cmap=cmap, vmin=VMIN, vmax=VMAX, aspect="auto")

    ax.set_xticks(np.arange(len(SYSTEMS)))
    ax.set_xticklabels(SYSTEM_LABELS)
    ax.set_yticks(np.arange(len(DOMAINS)))
    ax.set_yticklabels(DOMAINS)

    # The number is printed in every cell: the colour carries the pattern, the
    # digits carry the value, because neighbouring cells here differ by ~1 point
    # and no colour ramp is readable at that resolution.
    for yi, domain in enumerate(DOMAINS):
        for xi, system in enumerate(SYSTEMS):
            val = float(heat.loc[domain, system])
            color = text_colour(cmap, val)
            ax.text(xi, yi, f"{val:.1f}", ha="center", va="center", fontsize=6.2, color=color)

    ax.tick_params(length=0)
    # White minor gridlines on the half-integers draw the cell separators, so
    # the grid is a gap between tiles rather than a line drawn over them.
    ax.set_xticks(np.arange(-0.5, len(SYSTEMS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(DOMAINS), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5.8, length=2, width=0.5)
    cbar.outline.set_linewidth(0.55)
    cbar.set_label("Mean score (%)", fontsize=6)


def main() -> None:
    heat = load_grid()
    fig = plt.figure(figsize=FIGSIZE, constrained_layout=False)
    ax = fig.add_subplot(1, 1, 1)
    draw_heatmap(ax, heat)
    for ext in ("png", "svg", "pdf"):
        fig.savefig(f"{STEM}.{ext}", dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"panel e: {len(DOMAINS)} domains x {len(SYSTEMS)} systems")
    print(heat.round(1).to_string())
    print(f"  wrote {STEM}.png / .svg / .pdf at {DPI} dpi")


if __name__ == "__main__":
    main()
