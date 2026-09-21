"""C1 Figure 1: M2 Location probe heatmap — per-(position × layer) ridge R² for
verbal confidence in Gemma-3-27B-pt on TriviaQA. Ground for P1 Location PASS.

Reads per-cell probe outputs `results/m2/pos<POS>_L<LAYER>_ridge.json` produced
by the main experiment's M2 grid, plus the log-prob-only baseline scalar in the
same directory.
"""
from __future__ import annotations
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from paper_plot_style import plt, save_fig  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
STEM = "c1_m2_probe_heatmap"

POSITIONS = ["E0", "E1", "E2", "E3", "E4"]
LAYERS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# Load per-cell R² (test) from each ridge JSON.
grid = np.full((len(POSITIONS), len(LAYERS)), np.nan)
cell_pattern = re.compile(r"pos([A-Z0-9]+)_L(\d+)_ridge\.json")
for path in glob.glob("results/m2/pos*_L*_ridge.json"):
    m = cell_pattern.search(os.path.basename(path))
    if not m:
        continue
    pos, layer = m.group(1), int(m.group(2))
    if pos not in POSITIONS or layer not in LAYERS:
        continue
    with open(path) as f:
        d = json.load(f)
    r2 = d.get("test_r2_mean")
    if r2 is None:
        # Some cells may use different key naming.
        r2 = d.get("r2_test", d.get("r2_mean"))
    grid[POSITIONS.index(pos), LAYERS.index(layer)] = r2

# Log-prob-only baseline (single scalar).
with open("results/all_summary_v2.json") as f:
    summary = json.load(f)
baseline_r2 = summary.get("m2", {}).get("baseline_log_prob_only_r2")

fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.8))
vmin = min(0.0, np.nanmin(grid)) if np.any(~np.isnan(grid)) else 0.0
vmax = float(np.nanmax(grid)) if np.any(~np.isnan(grid)) else 0.6
im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
               vmin=vmin, vmax=vmax)

ax.set_xticks(range(len(LAYERS)))
ax.set_xticklabels([str(l) for l in LAYERS])
ax.set_yticks(range(len(POSITIONS)))
ax.set_yticklabels(POSITIONS)
ax.set_xlabel("Layer index (62 total)")
ax.set_ylabel("Post-answer position")

# Cell text values.
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        v = grid[i, j]
        if np.isnan(v):
            continue
        color = "white" if v < (vmin + 0.55 * (vmax - vmin)) else "black"
        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                color=color, fontsize=7)

# Mark the peak cell.
if np.any(~np.isnan(grid)):
    pi, pj = np.unravel_index(np.nanargmax(grid), grid.shape)
    ax.add_patch(plt.Rectangle((pj - 0.5, pi - 0.5), 1, 1,
                               fill=False, edgecolor="red", linewidth=1.6))

cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
cbar.set_label("Probe R² (test)")

if baseline_r2 is not None:
    ax.text(0.02, 1.06,
            f"log-prob-only baseline R² = {baseline_r2:+.3f}",
            transform=ax.transAxes, fontsize=8, va="bottom")

save_fig(fig, STEM, OUT_DIR)
