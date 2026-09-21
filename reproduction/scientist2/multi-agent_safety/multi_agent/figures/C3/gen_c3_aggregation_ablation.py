"""C3 aggregation ablation: transfer AUROC per family across 3 aggregations.

best-of-N and mean-pool both pass 5/7; attention-pool 4/7.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import plt, COLORS, save_fig  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(OUT_DIR, '..', '..'))

files = {
    'best-of-N':     os.path.join(PROJECT_ROOT, 'runs', 'M3', 'transfer_results.json'),
    'mean-pool':     os.path.join(PROJECT_ROOT, 'runs', 'M3', 'transfer_results_mean.json'),
    'attention-pool': os.path.join(PROJECT_ROOT, 'runs', 'M3', 'transfer_results_attention.json'),
}

data = {}
for agg, path in files.items():
    with open(path) as f:
        d = json.load(f)
    data[agg] = d['per_family_auroc']

# Order families by best-of-N AUROC descending
families = sorted(data['best-of-N'].keys(),
                  key=lambda k: -data['best-of-N'][k])

aggs = list(files.keys())
n_families = len(families)
n_aggs = len(aggs)
bar_width = 0.26
x = np.arange(n_families)

fig, ax = plt.subplots(1, 1, figsize=(7.5, 3.6))

agg_colors = {'best-of-N': COLORS[0],
              'mean-pool': COLORS[1],
              'attention-pool': COLORS[2]}

for i, agg in enumerate(aggs):
    values = [data[agg][fam] for fam in families]
    ax.bar(x + (i - (n_aggs - 1) / 2) * bar_width, values,
           width=bar_width, color=agg_colors[agg],
           label=agg, edgecolor='black', linewidth=0.4)

# Reference lines
ax.axhline(0.65, color='gray', linestyle='dashed', linewidth=0.9,
           label='Transfer threshold (0.65)')
ax.axhline(0.5, color='darkred', linestyle='dotted', linewidth=0.9,
           label='Chance')

ax.set_xticks(x)
ax.set_xticklabels(families, rotation=30, ha='right', fontsize=8)
ax.set_ylabel('Scenario-level AUROC')
ax.set_xlabel('Transfer family')
ax.set_ylim(0.40, 1.0)
ax.legend(frameon=False, loc='lower left', ncol=2, fontsize=8)

save_fig(fig, 'c3_aggregation_ablation', formats=('pdf', 'png'), out_dir=OUT_DIR)
plt.close(fig)
