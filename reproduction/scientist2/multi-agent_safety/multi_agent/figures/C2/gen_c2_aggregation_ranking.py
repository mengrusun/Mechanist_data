"""C2: Group-aggregation ranking bar chart.

Test scenario-level AUROC across 5 group aggregation techniques on mabench_core (n=49).
Best-of-N wins (0.690) but group-vs-best-single delta (mean-pool 0.665 -> 0.025) is
short of the 0.05 predicate. Predicate (a) FAILS.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import plt, COLORS, save_fig  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(OUT_DIR, '..', '..'))

with open(os.path.join(PROJECT_ROOT, 'runs', 'M2', 'verdict.json')) as f:
    verdict = json.load(f)

results = verdict['all_aggregation_test_auroc']
best_single = verdict['best_single_agent_auroc']

# Sort aggregations by AUROC descending
sorted_items = sorted(results.items(), key=lambda kv: -kv[1])
aggs = [k for k, _ in sorted_items]
values = [v for _, v in sorted_items]

fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.2))

# Color the winner distinctly
best_agg = verdict['best_group_aggregation']
bar_colors = [COLORS[2] if a == best_agg else COLORS[0] for a in aggs]

bars = ax.bar(aggs, values, color=bar_colors, width=0.65,
              edgecolor='black', linewidth=0.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# Reference: best single-agent baseline
ax.axhline(best_single, color='darkred', linestyle='dashed', linewidth=0.9,
           label=f'Best single-agent (mean-pool = {best_single:.3f})')

ax.set_ylabel('Scenario-level test AUROC (n=49)')
ax.set_xlabel('Group aggregation')
ax.set_ylim(0.50, 0.75)
ax.legend(frameon=False, loc='upper right', fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=15, ha='right', fontsize=8)

save_fig(fig, 'c2_aggregation_ranking', formats=('pdf', 'png'), out_dir=OUT_DIR)
plt.close(fig)
