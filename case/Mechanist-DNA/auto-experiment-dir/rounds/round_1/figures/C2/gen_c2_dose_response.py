# gen_c2_dose_response.py -- C2: dose-response of alpha-helix fraction vs steering strength
import sys, json
sys.path.insert(0, '.')
from paper_plot_style import *
import numpy as np

OUT_DIR = '.'

with open('../../results/m2_dose_response_curve.json') as f:
    d = json.load(f)

alphas = d['alphas']
per = d['per_dose']
means = [per[f'{a}']['helix_hgi_mean'] for a in alphas]
sems  = [per[f'{a}']['helix_hgi_sem']  for a in alphas]

baseline = per['0.0']['helix_hgi_mean']

fig, ax = plt.subplots(figsize=(5.4, 3.6))
xpos = np.arange(len(alphas))  # even spacing for the discrete dose grid
ax.errorbar(xpos, means, yerr=sems, marker='o', markersize=5,
            color=COLORS[0], capsize=3, linewidth=1.6, label='mean $\\alpha$-helix fraction (±SEM)')

ax.axhline(baseline, color='0.4', linestyle='--', linewidth=0.9, zorder=0)
ax.text(xpos[0], baseline + 0.008, f'baseline ($\\alpha$=0) = {baseline:.3f}',
        ha='left', va='bottom', fontsize=FONT_SIZE - 2, color='0.4')

# annotate clean optimum at alpha=8
i8 = alphas.index(8.0)
ax.annotate('clean optimum\n$\\alpha$=8', xy=(xpos[i8], means[i8]),
            xytext=(xpos[i8] - 1.4, means[i8] + 0.13),
            fontsize=FONT_SIZE - 2, ha='center',
            arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8))

# annotate peak at alpha=32
i32 = alphas.index(32.0)
ax.annotate(f'{means[i32]:.3f}', xy=(xpos[i32], means[i32]),
            xytext=(xpos[i32], means[i32] + 0.03),
            fontsize=FONT_SIZE - 2, ha='center')

ax.set_xticks(xpos)
ax.set_xticklabels([f'{int(a) if a==int(a) else a}' for a in alphas])
ax.set_xlabel('Steering strength $\\alpha$')
ax.set_ylabel('Mean DSSP $\\alpha$-helix fraction')
ax.set_ylim(0.4, 0.9)

rho = d['spearman_rho']; p = d['spearman_p']
ax.text(0.03, 0.97, f'Spearman $\\rho$={rho:.2f}, p={p:.1e}',
        transform=ax.transAxes, va='top', ha='left', fontsize=FONT_SIZE - 2)
ax.legend(frameon=False, loc='lower right', fontsize=FONT_SIZE - 2)
save_fig(fig, 'c2_dose_response', formats=('pdf', 'png'), out_dir=OUT_DIR)
