"""C1 — completeness ✓ vs minimality ✗ (target vs achieved)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from paper_plot_style import plt, COLORS, save_fig

OUT_DIR = os.path.dirname(__file__)

with open('results/M1_attribution.json') as f:
    d = json.load(f)

criteria = ['completeness', 'minimality']
targets  = [0.9,   0.05]
achieved = [d['completeness'], d['minimality']['avg_single_removal_drop']]

fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.3))
x = np.arange(len(criteria))
w = 0.35
b1 = ax.bar(x - w/2, targets,  w, label='target',  color=COLORS[7], edgecolor='black', linewidth=0.6)
b2 = ax.bar(x + w/2, achieved, w, label='achieved (Mistral-7B)', color=COLORS[0], edgecolor='black', linewidth=0.6)

for bar, v in zip(b1, targets):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=FONT_SIZE-2 if False else 9)
for bar, v in zip(b2, achieved):
    label = f'{v:.3f}' if v < 0.1 else f'{v:.2f}'
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, label, ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels(criteria)
ax.set_ylabel('score')
ax.set_ylim(0, 1.1)
ax.axhline(0, color='black', linewidth=0.5)
ax.legend(frameon=False, loc='upper right')

# annotate the pass/fail markers
ax.text(0, achieved[0] + 0.09, u'✓', ha='center', va='bottom', fontsize=14, color='green', fontweight='bold')
ax.text(1, achieved[1] + 0.09, u'✗', ha='center', va='bottom', fontsize=14, color='crimson', fontweight='bold')

paths = save_fig(fig, 'c1_completeness_minimality', OUT_DIR)
print(f'Saved: {paths}')
