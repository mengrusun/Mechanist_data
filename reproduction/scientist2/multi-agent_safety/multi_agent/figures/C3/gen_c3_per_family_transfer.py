"""C3: Per-family zero-shot transfer AUROC.

Frozen M1 layer-48 probe + M2 best-of-N aggregation across 6 mabench_transfer
families + mabench_stego blackjack. 5/7 pass the 0.65 threshold; stego at chance.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import plt, COLORS, save_fig  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(OUT_DIR, '..', '..'))

with open(os.path.join(PROJECT_ROOT, 'runs', 'M3', 'transfer_results.json')) as f:
    tr = json.load(f)

per_family = tr['per_family_auroc']
sorted_items = sorted(per_family.items(), key=lambda kv: -kv[1])
families = [k for k, _ in sorted_items]
values = [v for _, v in sorted_items]

fig, ax = plt.subplots(1, 1, figsize=(6.2, 3.4))

# Color: passing (>=0.65) in green-ish, failing in red-ish
bar_colors = [COLORS[2] if v >= 0.65 else COLORS[3] for v in values]

bars = ax.bar(families, values, color=bar_colors, width=0.7,
              edgecolor='black', linewidth=0.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# Reference lines
ax.axhline(0.65, color='gray', linestyle='dashed', linewidth=0.9,
           label='Transfer threshold (0.65)')
ax.axhline(0.5, color='darkred', linestyle='dotted', linewidth=0.9,
           label='Chance')

ax.set_ylabel('Scenario-level AUROC')
ax.set_xlabel('Transfer family')
ax.set_ylim(0.40, 1.0)
ax.legend(frameon=False, loc='lower left', fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=30, ha='right', fontsize=8)

save_fig(fig, 'c3_per_family_transfer', formats=('pdf', 'png'), out_dir=OUT_DIR)
plt.close(fig)
