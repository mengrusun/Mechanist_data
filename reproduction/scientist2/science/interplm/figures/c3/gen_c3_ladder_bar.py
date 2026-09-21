"""
c3_ladder_bar: Six-arm specificity ladder — SAE vs PCA vs random-rotation vs neurons vs shuffled-SAE.
Two thresholds: primary (tau_F1=0.5) and looser (tau_F1=0.3).
Data source: refine-logs/EXPERIMENT_RESULTS.md#M3 + M2 sensitivity sweep.
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import COLORS, save_fig, FONT_SIZE

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Arm labels (order = the ladder, best to worst)
arms = ['SAE', 'PCA', 'Rand. rot.', 'Neurons', 'Shuffled\nSAE']

# From EXPERIMENT_RESULTS.md M3 (primary q=0.99, tau=0.5)
covered_primary = [15, 0, 0, 0, 0]
# For the looser tau=0.3 setting: M2 sensitivity gives SAE=65, neurons=0 at q=0.99;
# M3 M2 sensitivity also implies controls collapse to 0 (they scored 0 at every tau tested);
# from the results file: SAE=65, PCA/rand/neurons/shuffled-SAE approx 0.
covered_loose = [65, 0, 0, 0, 0]

x = np.arange(len(arms))
width = 0.38

fig, ax = plt.subplots(1, 1, figsize=(6.6, 3.6))
b1 = ax.bar(x - width/2, covered_primary, width,
            label=r'$\tau_{F1}=0.5$ (primary)', color=COLORS[0])
b2 = ax.bar(x + width/2, covered_loose, width,
            label=r'$\tau_{F1}=0.3$ (loose)', color=COLORS[2])

# Value labels
for bar, val in zip(b1, covered_primary):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
            f'{val}', ha='center', va='bottom', fontsize=FONT_SIZE - 2)
for bar, val in zip(b2, covered_loose):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.2,
            f'{val}', ha='center', va='bottom', fontsize=FONT_SIZE - 2)

# Delta annotations between SAE and PCA at each threshold
delta_primary = covered_primary[0] - covered_primary[1]
delta_loose = covered_loose[0] - covered_loose[1]
ax.text(0.02, 0.94,
        rf'$\Delta_{{\mathrm{{SAE-PCA}}}}$ = {delta_primary} ($\tau$=0.5)   '
        rf'| {delta_loose} ($\tau$=0.3)',
        transform=ax.transAxes, fontsize=FONT_SIZE - 1, va='top')

# Strict target line at Delta >= 20 (mark on axis)
ax.axhline(20, color='0.5', linestyle=':', linewidth=0.9)
ax.text(len(arms) - 0.5, 20.5, r'strict $\Delta_{\mathrm{SAE-PCA}} \geq 20$',
        ha='right', va='bottom', fontsize=FONT_SIZE - 2, color='0.35')

ax.set_xticks(x)
ax.set_xticklabels(arms)
ax.set_ylabel('Concepts covered (of 400)')
ax.set_xlabel('Arm')
ax.set_ylim(0, max(covered_loose) * 1.20 + 4)
ax.legend(frameon=False, loc='upper right')

save_fig(fig, 'c3_ladder_bar', OUT_DIR, formats=('pdf', 'png'))
plt.close(fig)
