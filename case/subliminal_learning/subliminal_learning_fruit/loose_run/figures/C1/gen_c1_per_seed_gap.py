"""C1 — per-seed gap over max(Ctrl-A, Ctrl-B). Shows all 8 seeds clear the 5 pp bar."""
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import COLORS, REF_COLOR, save_fig, FONT_SIZE  # noqa: E402

DATA_PATH = '/path/to/project/multi_modal_B_loose1/runs/M0_verdict.json'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(DATA_PATH) as f:
    d = json.load(f)
ev = d['evidence']

delta_a = ev['per_seed_delta_A']  # P_teacher - P_ctrl_A per seed
delta_b = ev['per_seed_delta_B']  # P_teacher - P_ctrl_B per seed

seeds = sorted(delta_a.keys(), key=int)
gaps = [min(delta_a[s], delta_b[s]) for s in seeds]  # min per seed = gap over max control

median_gap = float(np.median(gaps))
threshold = 0.05

fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.4))
x = np.arange(len(seeds))
bars = ax.bar(x, gaps, 0.55, color=COLORS[2], edgecolor='none')

ax.axhline(threshold, color=REF_COLOR, linestyle='--', linewidth=1.2,
           label=f'Pre-registered 5 pp threshold')
ax.axhline(median_gap, color='#555555', linestyle=':', linewidth=1.0,
           label=f'Median gap = {median_gap * 100:.1f} pp')

ax.set_xlabel('Student training seed')
ax.set_ylabel('Per-seed gap over max(Ctrl-A, Ctrl-B) [pp]')
ax.set_xticks(x)
ax.set_xticklabels([int(s) for s in seeds])
ax.set_ylim(0, max(gaps) * 1.15)

# Use pp on y ticks
yticks = ax.get_yticks()
ax.set_yticklabels([f'{y * 100:.0f}' for y in yticks])

for xi, g in zip(x, gaps):
    ax.text(xi, g + 0.008, f'{g * 100:.1f}', ha='center', va='bottom',
            fontsize=FONT_SIZE - 2, color=COLORS[2])

ax.legend(loc='upper right', frameon=False)
fig.tight_layout()
save_fig(fig, 'c1_per_seed_gap', formats=('pdf', 'png'), out_dir=OUT_DIR)
plt.close(fig)
