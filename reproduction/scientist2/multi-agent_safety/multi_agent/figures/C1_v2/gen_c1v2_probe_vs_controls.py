"""C1_v2: Probe test AUROC vs sanity controls.

Layer-48 probe test AUROC (0.665) vs three sanity controls:
- Label-permute 0.412 (chance-check)
- Length-match 0.642
- Topic-swap 0.651
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import plt, COLORS, save_fig  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(OUT_DIR, '..', '..'))

with open(os.path.join(PROJECT_ROOT, 'runs', 'M1', 'verdict.json')) as f:
    verdict = json.load(f)

probe_auroc = verdict['probe_test_auroc_per_scenario_mean_pool']
sanity = verdict['sanity_checks']

labels = ['Probe (layer 48)',
          'Label-permute\n(chance-check)',
          'Length-match',
          'Topic-swap']
values = [probe_auroc,
          sanity['label_permute']['auroc'],
          sanity['length_match']['auroc'],
          sanity['topic_swap']['auroc']]
colors = [COLORS[0], COLORS[3], COLORS[2], COLORS[4]]

fig, ax = plt.subplots(1, 1, figsize=(5, 3.2))
bars = ax.bar(labels, values, color=colors, width=0.65,
              edgecolor='black', linewidth=0.5)

for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=8)

# Chance reference
ax.axhline(0.5, color='gray', linestyle='dashed', linewidth=0.9, label='Chance')

ax.set_ylabel('Scenario-level AUROC (test, n=49)')
ax.set_ylim(0.35, 0.78)
ax.legend(frameon=False, loc='upper right', fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=0, ha='center', fontsize=8)

save_fig(fig, 'c1v2_probe_vs_controls', formats=('pdf', 'png'), out_dir=OUT_DIR)
plt.close(fig)
