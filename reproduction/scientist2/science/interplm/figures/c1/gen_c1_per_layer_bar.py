"""
c1_per_layer_bar: SAE vs raw-neuron estimated interpretable feature count per layer.
Data source: refine-logs/EXPERIMENT_RESULTS.md#M1 (values inlined below since the source is a Markdown table).
"""
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import COLORS, save_fig, FONT_SIZE

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Values transcribed from refine-logs/EXPERIMENT_RESULTS.md M1 table
layers = [1, 9, 18, 24, 30, 33]
sae_interp = [0, 1480, 1227, 1023, 1010, 1383]
neuron_interp = [166, 77, 154, 128, 115, 166]
ratios = [0.0, 19.3, 8.0, 8.0, 8.8, 8.3]
TARGET = 2548

x = np.arange(len(layers))
width = 0.38

fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.6))
b1 = ax.bar(x - width/2, sae_interp, width, label='SAE features', color=COLORS[0])
b2 = ax.bar(x + width/2, neuron_interp, width, label='Raw neurons', color=COLORS[1])

# Target line
ax.axhline(TARGET, color='0.35', linestyle='--', linewidth=1.0,
           label=f'Target ~{TARGET}')

# Value labels
for bar, val in zip(b1, sae_interp):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
                f'{val}', ha='center', va='bottom', fontsize=FONT_SIZE - 2)
for bar, val in zip(b2, neuron_interp):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
            f'{val}', ha='center', va='bottom', fontsize=FONT_SIZE - 2)

# Ratio annotations along the top
for i, r in enumerate(ratios):
    label = f'{r:.1f}x' if r > 0 else 'n/a'
    ax.text(i, TARGET + 130, label, ha='center', va='bottom',
            fontsize=FONT_SIZE - 2, color='0.30')

ax.set_xticks(x)
ax.set_xticklabels([f'L{L}' for L in layers])
ax.set_xlabel('ESM-2-650M layer')
ax.set_ylabel('Estimated interpretable feature count')
ax.set_ylim(0, TARGET * 1.20)
ax.legend(frameon=False, loc='upper left')

save_fig(fig, 'c1_per_layer_bar', OUT_DIR, formats=('pdf', 'png'))
plt.close(fig)
