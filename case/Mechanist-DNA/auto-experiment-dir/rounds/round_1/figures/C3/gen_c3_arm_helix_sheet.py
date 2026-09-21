# gen_c3_arm_helix_sheet.py -- C3: arm-wise helix/sheet fractions at locked dose alpha=8
import sys, json
sys.path.insert(0, '.')
from paper_plot_style import *
import numpy as np

OUT_DIR = '.'

with open('../../results/m3_specificity_summary_v2.json') as f:
    d = json.load(f)

arm = d['arm_table_at_locked_dose']
arms = ['baseline', 'alpha-helix S', 'matched-control', 'beta-sheet off-target']
helix = [d['baseline_helix'],
         arm['alpha_helix_S']['helix_mean'],
         arm['matched_control']['helix_mean'],
         arm['beta_sheet_offtarget']['helix_mean']]
sheet = [d['baseline_sheet'],
         arm['alpha_helix_S']['sheet_mean'],
         arm['matched_control']['sheet_mean'],
         arm['beta_sheet_offtarget']['sheet_mean']]

fig, ax = plt.subplots(figsize=(6.4, 3.6))
x = np.arange(len(arms))
w = 0.38
b1 = ax.bar(x - w / 2, helix, w, label='Helix fraction', color=COLORS[0],
            edgecolor='black', linewidth=0.4)
b2 = ax.bar(x + w / 2, sheet, w, label='Sheet fraction', color=COLORS[3],
            edgecolor='black', linewidth=0.4)
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.006,
                f'{b.get_height():.3f}', ha='center', va='bottom',
                fontsize=FONT_SIZE - 3)

ax.axhline(d['baseline_helix'], color=COLORS[0], linestyle=':', linewidth=0.8, zorder=0)
ax.axhline(d['baseline_sheet'], color=COLORS[3], linestyle=':', linewidth=0.8, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(arms, fontsize=FONT_SIZE - 2)
ax.set_ylabel('Secondary-structure fraction')
ax.set_ylim(0, 0.72)
ax.legend(frameon=False, loc='upper right', fontsize=FONT_SIZE - 2)
save_fig(fig, 'c3_arm_helix_sheet', formats=('pdf', 'png'), out_dir=OUT_DIR)
