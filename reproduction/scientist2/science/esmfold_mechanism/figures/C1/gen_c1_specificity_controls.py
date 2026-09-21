import json, sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
from paper_plot_style import plt, COLORS, save_fig

d = json.load(open('results/M1/summary_stats.json'))
rows = d['per_condition_stats']

def find(cond, band):
    for r in rows:
        if r['condition'] == cond and r['band'] == band:
            return r['delta_rate']
    return None

controls = [
    ('early s-patch\n(b_0_3)',        find('s_patch','b_0_3')),
    ('same-window z-patch\n(b_0_3)',  find('z_patch_early','b_0_3')),
    ('matched-mask s-patch\n(b_0_3)', find('s_patch_matched_ctrl','b_0_3')),
    ('late-window s-patch\n(b_32_39)',find('s_patch','b_32_39')),
]

labels = [c[0] for c in controls]
vals   = [c[1] for c in controls]
colors = [COLORS[3], COLORS[0], COLORS[1], COLORS[2]]

fig, ax = plt.subplots(1, 1, figsize=(5.5, 3.2))
bars = ax.bar(labels, vals, color=colors, edgecolor='none')
ax.axhline(0, color='k', linewidth=0.5)
ax.set_ylabel(r'$\Delta$ hairpin rate vs baseline')
ax.set_ylim(-1.0, 0.05)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.04,
            f'{v:+.3f}', ha='center', va='top', fontsize=8)

save_fig(fig, 'c1_specificity_controls', formats=('pdf', 'png'), out_dir='figures/C1')
print('OK c1_specificity_controls')
