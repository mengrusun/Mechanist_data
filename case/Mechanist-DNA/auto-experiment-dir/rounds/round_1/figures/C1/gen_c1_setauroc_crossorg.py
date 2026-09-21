# gen_c1_setauroc_crossorg.py -- C1: cross-organism set-level AUROC vs baselines
import sys, json
sys.path.insert(0, '.')
from paper_plot_style import *
import numpy as np

OUT_DIR = '.'

# E. coli (prokaryote) primary config: HGI, seed 200
with open('../../results/m0_prokaryote_HGI_s200.json') as f:
    ecoli = json.load(f)
# H. sapiens (eukaryote) primary config: HGI, seed 42
with open('../../results/m0_eukaryote_HGI_s42.json') as f:
    hsap = json.load(f)

organisms = ['E. coli', 'H. sapiens']
# group order: set-level AUROC (test), confound-only baseline, shuffle-null (mean)
set_auroc   = [ecoli['combined_set_test_auroc'],      hsap['combined_set_test_auroc']]
confound    = [ecoli['confound_only_test_auroc'],     hsap['confound_only_test_auroc']]
null        = [ecoli['shuffle_null_auroc_mean'],      hsap['shuffle_null_auroc_mean']]

groups = [
    ('Set-level AUROC (test)', set_auroc, COLORS[0]),
    ('Confound-only baseline', confound, COLORS[7]),
    ('Shuffle-null (mean)',    null,     COLORS[8]),
]

fig, ax = plt.subplots(figsize=(5.6, 3.4))
x = np.arange(len(organisms))
w = 0.26
for i, (label, vals, color) in enumerate(groups):
    offset = (i - 1) * w
    bars = ax.bar(x + offset, vals, w, label=label, color=color,
                  edgecolor='black', linewidth=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                f'{v:.3f}', ha='center', va='bottom', fontsize=FONT_SIZE - 2)

ax.axhline(0.5, color='0.35', linestyle='--', linewidth=0.8, zorder=0)
ax.text(-0.42, 0.505, 'chance (0.5)', ha='left', va='bottom',
        fontsize=FONT_SIZE - 2, color='0.35')

ax.set_xticks(x)
ax.set_xticklabels(organisms)
ax.set_ylabel('Test AUROC')
ax.set_ylim(0.4, 1.02)
ax.legend(frameon=False, loc='upper center', ncol=3, fontsize=FONT_SIZE - 2,
          bbox_to_anchor=(0.5, 1.14), columnspacing=1.2, handletextpad=0.5)
save_fig(fig, 'c1_setauroc_crossorg', formats=('pdf', 'png'), out_dir=OUT_DIR)
