"""C2 figure: bar chart of |H*| for each (Pythia model, target) pair."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_plot_style import COLORS, save_fig
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PAIRS = [
    ("pythia-410m", "personal"),
    ("pythia-1b", "personal"),
    ("pythia-1b", "attributed"),
    ("pythia-2.8b", "personal"),
    ("pythia-2.8b", "attributed"),
]


def load_size(model, target):
    p = os.path.join(ROOT, "refine-logs/artifacts/hstar", model, f"H_{target}.json")
    with open(p) as f:
        d = json.load(f)
    return int(d.get("hstar_size", len(d.get("heads", d.get("hstar_heads", [])))))


sizes = [load_size(m, t) for m, t in PAIRS]
labels = [f"{m.replace('pythia-', '')}\n{t}" for m, t in PAIRS]
target_colors = [COLORS[0] if t == "personal" else COLORS[3] for _, t in PAIRS]

fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.0))
x = np.arange(len(PAIRS))
bars = ax.bar(x, sizes, color=target_colors, edgecolor="black", linewidth=0.4)
for xi, s in zip(x, sizes):
    ax.text(xi, s + 0.08, str(s), ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel(r"$|H^*|$  (smallest localized head-set size)")
ax.set_ylim(0, max(sizes) + 1.2)
ax.axhline(np.median(sizes), linestyle="--", linewidth=0.7, color="grey", alpha=0.7)
ax.text(x[-1] + 0.15, np.median(sizes) + 0.05, f"median = {int(np.median(sizes))}", fontsize=8, color="grey", ha="right", va="bottom")

# legend patches
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor=COLORS[0], edgecolor="black", label="personal_belief"),
    Patch(facecolor=COLORS[3], edgecolor="black", label="attributed_belief"),
]
ax.legend(handles=legend_elems, frameon=False, loc="upper right")

save_fig(fig, "c2_hstar_size_bar", OUT_DIR, formats=("pdf", "png"))
