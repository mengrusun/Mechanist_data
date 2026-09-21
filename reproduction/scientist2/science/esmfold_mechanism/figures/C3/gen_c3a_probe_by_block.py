import json, sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + '/..'))
from paper_plot_style import plt, COLORS, save_fig

d = json.load(open('results/M3a/summary_stats.json'))
per_block = d['per_block']
per_block_sorted = sorted(per_block, key=lambda r: r['block'])

blocks = [str(r['block']) for r in per_block_sorted]
bacc = [r['test_balanced_acc'] for r in per_block_sorted]
auroc = [r['test_auroc_macro'] for r in per_block_sorted]
pvals = [r['permutation_p'] for r in per_block_sorted]

n_blocks = len(blocks)
bar_w = 0.35
xs = list(range(n_blocks))

fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.0))
ax.bar([x - bar_w/2 for x in xs], bacc,  width=bar_w, color=COLORS[0], edgecolor='none', label='Balanced accuracy (test)')
ax.bar([x + bar_w/2 for x in xs], auroc, width=bar_w, color=COLORS[1], edgecolor='none', label='AUROC macro (test)')
ax.axhline(1/3, color='k', linestyle=':', linewidth=0.6, label='chance (1/3)')

for x, p in zip(xs, pvals):
    ax.text(x, 1.02, f'p={p:.3g}', ha='center', va='bottom', fontsize=8)

ax.set_xticks(xs)
ax.set_xticklabels(blocks)
ax.set_xlabel(r'Early block $k$ (sampled from $s$ representation)')
ax.set_ylabel('Score')
ax.set_ylim(0, 1.15)
ax.legend(loc='lower right', frameon=False)

save_fig(fig, 'c3a_probe_by_block', formats=('pdf', 'png'), out_dir='figures/C3')
print('OK c3a_probe_by_block')
