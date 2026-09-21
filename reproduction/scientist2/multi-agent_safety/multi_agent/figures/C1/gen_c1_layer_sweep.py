"""C1: Per-layer probe scenario-level AUROC sweep.

Illustrates the layer-selection instability that motivated the C1 -> C1_v2 narrowing:
dev-best layer 48 has the worst test AUROC (0.665); layer 27 hits the 0.75 threshold.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import plt, COLORS, save_fig  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(OUT_DIR, '..', '..'))

LAYERS = [27, 37, 48, 59]
layer_files = {L: os.path.join(PROJECT_ROOT, 'runs', 'M1', f'probe_layer{L}.json')
               for L in LAYERS}

test_scen_auroc = []
dev_scen_auroc = []
for L in LAYERS:
    with open(layer_files[L]) as f:
        d = json.load(f)
    test_scen_auroc.append(d['test_auroc_per_scenario_mean_pool'])
    dev_scen_auroc.append(d['dev_auroc_per_scenario_mean_pool'])

fig, ax = plt.subplots(1, 1, figsize=(5, 3.2))

ax.plot(LAYERS, test_scen_auroc, marker='o', markersize=6,
        linewidth=1.8, color=COLORS[0], label='Test (mean-pool, n=49)')
ax.plot(LAYERS, dev_scen_auroc, marker='s', markersize=6,
        linewidth=1.4, color=COLORS[1], linestyle='--',
        label='Dev (mean-pool, n=33)')

# Threshold reference lines
ax.axhline(0.75, color='gray', linestyle='dashed', linewidth=0.9,
           label='Predicate threshold (0.75)')
ax.axhline(0.60, color='darkred', linestyle='dotted', linewidth=0.9,
           label='Text-only judge (OTHER-coerced)')

# Highlight dev-best layer 48 (worst test)
best_dev_layer = 48
best_test = test_scen_auroc[LAYERS.index(best_dev_layer)]
ax.annotate('dev-best; worst test',
            xy=(best_dev_layer, best_test),
            xytext=(best_dev_layer + 3, best_test - 0.06),
            fontsize=8, color='black',
            arrowprops=dict(arrowstyle='->', color='black', lw=0.7))

ax.set_xlabel('Transformer layer')
ax.set_ylabel('Scenario-level AUROC')
ax.set_xticks(LAYERS)
ax.set_ylim(0.55, 0.90)
ax.legend(frameon=False, loc='lower left', fontsize=8)

save_fig(fig, 'c1_layer_sweep', formats=('pdf', 'png'), out_dir=OUT_DIR)
plt.close(fig)
