# gen_c2_quality_overlay.py -- C2 (optional): valid-ORF rate vs steering strength
import sys, json
sys.path.insert(0, '.')
from paper_plot_style import *
import numpy as np

OUT_DIR = '.'

with open('../../results/m2_dose_response_curve.json') as f:
    d = json.load(f)

alphas = d['alphas']
per = d['per_dose']
valid = [per[f'{a}']['valid_orf_rate'] for a in alphas]
baseline_valid = per['0.0']['valid_orf_rate']

fig, ax = plt.subplots(figsize=(5.4, 3.6))
xpos = np.arange(len(alphas))
ax.plot(xpos, valid, marker='s', markersize=5, color=COLORS[2],
        linewidth=1.6, label='valid-ORF rate')

ax.axhline(baseline_valid, color='0.4', linestyle='--', linewidth=0.9, zorder=0)
ax.text(xpos[0], baseline_valid + 0.006, f'baseline = {baseline_valid:.3f}',
        ha='left', va='bottom', fontsize=FONT_SIZE - 2, color='0.4')

# shade the region alpha >= 16 where quality dips
i16 = alphas.index(16.0)
ax.axvspan(xpos[i16] - 0.5, xpos[-1] + 0.5, color=COLORS[3], alpha=0.10, zorder=0)
ax.text((xpos[i16] + xpos[-1]) / 2, 0.66, 'quality dip\n$\\alpha\\geq$16',
        ha='center', va='bottom', fontsize=FONT_SIZE - 2, color=COLORS[3])

i8 = alphas.index(8.0)
ax.annotate('optimum $\\alpha$=8\nquality preserved', xy=(xpos[i8], valid[i8]),
            xytext=(xpos[i8] - 2.0, valid[i8] - 0.10),
            fontsize=FONT_SIZE - 2, ha='center',
            arrowprops=dict(arrowstyle='->', color='0.3', lw=0.8))

ax.set_xticks(xpos)
ax.set_xticklabels([f'{int(a) if a==int(a) else a}' for a in alphas])
ax.set_xlabel('Steering strength $\\alpha$')
ax.set_ylabel('Valid-ORF rate')
ax.set_ylim(0.6, 0.95)
ax.legend(frameon=False, loc='lower left', fontsize=FONT_SIZE - 2)
save_fig(fig, 'c2_quality_overlay', formats=('pdf', 'png'), out_dir=OUT_DIR)
