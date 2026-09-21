"""C3 — necessary-but-not-sufficient asymmetry, cross-family (Mistral-7B vs Gemma-2-9B)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from paper_plot_style import plt, COLORS, save_fig

OUT_DIR = os.path.dirname(__file__)

with open('results/M2_necessity.json') as f:
    m2 = json.load(f)
with open('results/M3_sufficiency.json') as f:
    m3 = json.load(f)
with open('results/M5_gemma9b.json') as f:
    m5 = json.load(f)

nec_mistral  = m2['recovery']['logit_diff']
suf_mistral  = m3['sufficient_recovery']['logit_diff']
nec_gemma    = m5['necessity_recovery']['logit_diff']
suf_gemma    = m5['sufficiency_recovery']['logit_diff']

nec_vals = [nec_mistral, nec_gemma]
suf_vals = [suf_mistral, suf_gemma]
models = ['Mistral-7B', 'Gemma-2-9B']

fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.3))
x = np.arange(len(models))
w = 0.35
b1 = ax.bar(x - w/2, nec_vals, w, label='necessity (path-patch)',    color=COLORS[2], edgecolor='black', linewidth=0.6)
b2 = ax.bar(x + w/2, suf_vals, w, label='sufficiency (reinsertion)', color=COLORS[3], edgecolor='black', linewidth=0.6)

for bar, v in zip(b1, nec_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)
for bar, v in zip(b2, suf_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{v:.2f}', ha='center', va='bottom', fontsize=9)

ax.axhline(0.8, color='gray', linestyle='--', linewidth=0.7)
ax.text(1.4, 0.82, 'target ≥ 0.8', ha='right', va='bottom', fontsize=8, color='gray')

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel('LD recovery')
ax.set_ylim(0, 1.2)
ax.legend(frameon=False, loc='upper right')

paths = save_fig(fig, 'c3_necessity_vs_sufficiency', OUT_DIR)
print(f'Saved: {paths}')
