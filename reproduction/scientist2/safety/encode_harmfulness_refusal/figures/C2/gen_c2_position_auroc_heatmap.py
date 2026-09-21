"""C2 heatmap — per-position AUROC of h and r directions on the harmfulness attribute.

Data: results/m2/position_auroc_heatmap.csv
Output: figures/C2/c2_position_auroc_heatmap.{pdf,png}
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_plot_style import plt, save_fig  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(PROJECT_ROOT, "results/m2/position_auroc_heatmap.csv")

positions = []
auroc_h = []
auroc_r = []
with open(DATA) as f:
    for row in csv.DictReader(f):
        positions.append(row["position"])
        auroc_h.append(float(row["auroc_harmfulness"]))
        auroc_r_val = row["auroc_refusal"].strip()
        auroc_r.append(float(auroc_r_val) if auroc_r_val else np.nan)

matrix = np.array([auroc_h, auroc_r])
row_labels = ["harmfulness (h)", "refusal (r)"]

fig, ax = plt.subplots(1, 1, figsize=(5.4, 2.2))
im = ax.imshow(matrix, cmap="viridis", vmin=0.5, vmax=1.0, aspect="auto")

ax.set_xticks(np.arange(len(positions)))
ax.set_xticklabels(positions, rotation=30, ha="right")
ax.set_yticks([0, 1])
ax.set_yticklabels(row_labels)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        val = matrix[i, j]
        if np.isnan(val):
            txt, color = "NaN", "white"
        else:
            txt = f"{val:.4f}"
            color = "white" if val < 0.75 else "black"
        ax.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)

cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label("AUROC", rotation=270, labelpad=12)

ax.set_xlabel("Token position")
save_fig(fig, "c2_position_auroc_heatmap", HERE)
plt.close(fig)
