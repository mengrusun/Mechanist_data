import json, sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
from paper_plot_style import plt, COLORS, save_fig

d = json.load(open('results/M1/summary_stats.json'))
rows = d['per_condition_stats']

BAND_ORDER = ['b_0_3','b_4_7','b_8_11','b_12_15','b_16_23','b_24_31','b_32_39','b_40_47']
COND_ORDER = ['s_patch','z_patch_early','s_patch_matched_ctrl']
COND_LABEL = {'s_patch':'s-patch', 'z_patch_early':'z-patch (same window)', 's_patch_matched_ctrl':'s-patch matched-ctrl'}

matrix = {c: {b: None for b in BAND_ORDER} for c in COND_ORDER}
for r in rows:
    if r['condition'] in matrix and r['band'] in matrix[r['condition']]:
        matrix[r['condition']][r['band']] = r['delta_rate']

n_bands = len(BAND_ORDER)
n_conds = len(COND_ORDER)
bar_w = 0.26

fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.2))
for i, c in enumerate(COND_ORDER):
    xs = [x + (i - 1) * bar_w for x in range(n_bands)]
    ys = [matrix[c][b] if matrix[c][b] is not None else 0.0 for b in BAND_ORDER]
    ax.bar(xs, ys, width=bar_w, color=COLORS[i], label=COND_LABEL[c], edgecolor='none')

ax.axhline(0, color='k', linewidth=0.5)
ax.set_xticks(range(n_bands))
ax.set_xticklabels([b.replace('b_','[') + ']' for b in BAND_ORDER], rotation=20)
ax.set_xlabel('Block band (window over the 48-block trunk)')
ax.set_ylabel(r'$\Delta$ hairpin rate vs baseline')
ax.set_ylim(-1.0, 0.05)
ax.legend(loc='lower right', frameon=False)

save_fig(fig, 'c1_effect_by_band', formats=('pdf', 'png'), out_dir='figures/C1')
print('OK c1_effect_by_band')
