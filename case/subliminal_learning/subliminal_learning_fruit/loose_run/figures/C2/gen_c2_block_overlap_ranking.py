"""C2 — Grassmann overlap_gap_k (k=1) across all 60 DiT blocks. Highlight b*=47."""
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import COLORS, REF_COLOR, save_fig, FONT_SIZE  # noqa: E402

DATA_PATH = '/path/to/project/multi_modal_B_loose1/runs/m1_location/shortlist.json'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(DATA_PATH) as f:
    d = json.load(f)

per_block = d['per_block']
b_star = d.get('b_star', 47)
top_ratio = d.get('top_block_k1_ratio', None)

blocks = sorted((int(k) for k in per_block.keys()))
gaps = [per_block[str(b)]['overlap_gap_k1'] for b in blocks]

median_gap = float(np.median(gaps))

# Determine bar colors: highlight b_star; grey otherwise
bar_colors = [COLORS[1] if b == b_star else '#7f7f7f' for b in blocks]

fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.4))
ax.bar(blocks, gaps, width=0.8, color=bar_colors, edgecolor='none')

ax.axhline(median_gap, color=REF_COLOR, linestyle='--', linewidth=1.0,
           label=f'Median across 60 blocks = {median_gap:.4f}')

ax.set_xlabel('DiT block index')
ax.set_ylabel(r'overlap$_{\mathrm{gap}}$ (k = 1)')
ax.set_xlim(-1, 60)
ax.set_xticks(np.arange(0, 60, 5))
ax.legend(loc='upper left', frameon=False)

# Highlight annotation for b*
b_star_val = per_block[str(b_star)]['overlap_gap_k1']
label = f'b* = {b_star}'
if top_ratio is not None:
    label += f'\ntop/median = {top_ratio:.2f}×'
ax.annotate(label,
            xy=(b_star, b_star_val),
            xytext=(b_star - 15, b_star_val * 0.95),
            fontsize=FONT_SIZE - 1,
            arrowprops=dict(arrowstyle='->', color='black', lw=0.8),
            ha='left', va='top')

fig.tight_layout()
save_fig(fig, 'c2_block_overlap_ranking', formats=('pdf', 'png'), out_dir=OUT_DIR)
plt.close(fig)
