# gen_c3_random_null.py -- C3: 33-direction random-null of helix delta vs S's helix gain
import sys, json
sys.path.insert(0, '.')
from paper_plot_style import *
import numpy as np

OUT_DIR = '.'

with open('../../results/m3_specificity_summary_v2.json') as f:
    summ = json.load(f)
baseline = summ['baseline_helix']
S_delta = summ['PRIMARY_random_direction_null']['S_helix_delta']
p = summ['PRIMARY_random_direction_null']['empirical_one_sided_p']
z = summ['PRIMARY_random_direction_null']['z_score']
n = summ['PRIMARY_random_direction_null']['n_directions']
n_ge = summ['PRIMARY_random_direction_null']['n_random_ge_S']

deltas = []
for part in ['d0-10', 'd11-21', 'd22-32']:
    with open(f'../../results/m3_random_control_{part}.json') as f:
        dd = json.load(f)
    for dr in dd['directions']:
        deltas.append(dr['helix_mean'] - baseline)
deltas = np.array(deltas)
assert len(deltas) == n, f'expected {n} directions, got {len(deltas)}'

fig, ax = plt.subplots(figsize=(4.8, 3.8))
bp = ax.boxplot([deltas], positions=[0], widths=0.5, patch_artist=True,
                showfliers=False, medianprops=dict(color='black'))
bp['boxes'][0].set_facecolor(COLORS[8])
bp['boxes'][0].set_alpha(0.6)

# jittered scatter of the 33 random directions
rng = np.random.default_rng(0)
jit = rng.uniform(-0.14, 0.14, size=len(deltas))
ax.scatter(jit, deltas, s=16, color=COLORS[7], alpha=0.8, zorder=3,
           edgecolor='white', linewidth=0.3, label=f'random directions (n={n})')

# S's helix delta marked beyond the whole distribution
ax.scatter([0], [S_delta], marker='*', s=260, color=COLORS[1], zorder=5,
           edgecolor='black', linewidth=0.6, label='$\\alpha$-helix set S')
ax.annotate(f'S = +{S_delta:.3f}', xy=(0, S_delta), xytext=(0.28, S_delta),
            va='center', fontsize=FONT_SIZE - 1)

ax.axhline(0.0, color='0.5', linestyle='--', linewidth=0.8, zorder=0)
ax.set_xticks([])
ax.set_xlim(-0.5, 0.85)
ax.set_ylabel('Helix fraction $\\Delta$ vs baseline')
ax.text(0.03, 0.97,
        f'{n_ge}/{n} $\\geq$ S\np = {p:.3f}, z = {z:.2f}',
        transform=ax.transAxes, va='top', ha='left', fontsize=FONT_SIZE - 2)
ax.legend(frameon=False, loc='lower right', fontsize=FONT_SIZE - 2)
save_fig(fig, 'c3_random_null', formats=('pdf', 'png'), out_dir=OUT_DIR)
