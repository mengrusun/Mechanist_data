"""Grouped bar: per-seed QA_I Acc drop at LR=1e-3 vs LR=1.5e-3.

Reads figures/c1_lr_cliff_aggregated.json (built from
results/qa_i_ctrl.jsonl + results/qa_i_treated_seed*.jsonl + results/qa_i_lr1.5e-3_seed*.jsonl).
Draws the M0 threshold (3.0 pp) as a horizontal reference line.
"""
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from paper_plot_style import COLORS, save_fig  # noqa: E402

DATA = "figures/c1_lr_cliff_aggregated.json"
OUT_DIR = "figures/C1"
STEM = "c1_per_seed_drop_two_lrs"

with open(DATA) as f:
    d = json.load(f)

seeds = [100, 200, 300]
lr_labels = ["1e-3 (preregistered)", "1.5e-3 (iter-5 rescue)"]
lr_keys = ["1e-3", "1.5e-3"]

drops = {lr: [d["per_lr"][lr][str(s)]["drop_pp"] for s in seeds] for lr in lr_keys}

fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.4))

x = np.arange(len(seeds))
width = 0.36
b1 = ax.bar(x - width / 2, drops["1e-3"], width, label=lr_labels[0], color=COLORS[0])
b2 = ax.bar(x + width / 2, drops["1.5e-3"], width, label=lr_labels[1], color=COLORS[2])

# M0 threshold reference line
ax.axhline(3.0, linestyle="--", linewidth=1.0, color="grey", label="M0 threshold (3.0 pp)")
ax.axhline(0.0, linewidth=0.6, color="black")

# Value labels
for bars in (b1, b2):
    for bar in bars:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, h + (0.5 if h >= 0 else -0.5),
                f"{h:.1f}", ha="center", va=va, fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels([f"seed {s}" for s in seeds])
ax.set_ylabel("Acc(QA_I) drop vs Ctrl  (percentage points)")
ax.set_ylim(-8, 36)
ax.legend(frameon=False, loc="upper left")

save_fig(fig, STEM, OUT_DIR)
plt.close(fig)
