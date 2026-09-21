"""C1 — per-layer per-head attention attribution magnitudes (32 layers × 32 heads on Mistral-7B)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from paper_plot_style import plt, COLORS, save_fig

OUT_DIR = os.path.dirname(__file__)

with open('results/M1_attribution.json') as f:
    d = json.load(f)

attn_abs = d['attn_head_abs']  # list-of-lists (n_layers x n_heads) OR flat
attn = np.array(attn_abs, dtype=float)
if attn.ndim == 1:
    # infer shape from Mistral-7B (32 layers x 32 heads)
    n_layers, n_heads = 32, 32
    attn = attn.reshape(n_layers, n_heads)

fig, ax = plt.subplots(1, 1, figsize=(5.0, 4.2))
im = ax.imshow(attn, aspect='auto', cmap='magma', interpolation='nearest')
ax.set_xlabel('head index')
ax.set_ylabel('layer index')

# Annotate a small number of the strongest cells
mask = attn > 0
if mask.any():
    thr = np.quantile(attn[mask], 0.995) if mask.sum() > 20 else attn[mask].max()
    for L in range(attn.shape[0]):
        for H in range(attn.shape[1]):
            if attn[L, H] >= thr:
                ax.text(H, L, u'●', ha='center', va='center', fontsize=6, color='white')

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
cbar.set_label('|direct-effect score|')

paths = save_fig(fig, 'c1_attribution_heatmap', OUT_DIR)
print(f'Saved: {paths}')
