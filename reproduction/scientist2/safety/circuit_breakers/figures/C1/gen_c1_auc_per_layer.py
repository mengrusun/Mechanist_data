"""C1 figure: layer-wise probe AUC on residual stream (M1 output)."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paper_plot_style import plt, COLORS, save_fig

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(__file__)

with open(os.path.join(ROOT, "artifacts/m1/auc_per_layer.json")) as f:
    auc = {int(k): v for k, v in json.load(f).items()}

xs = sorted(auc.keys())
ys = [auc[i] for i in xs]

CHOSEN_SITES = list(range(9, 15))

fig, ax = plt.subplots(1, 1, figsize=(6, 3))
ax.plot(xs, ys, marker="o", markersize=3.5, linewidth=1.4, color=COLORS[0])
ax.axhline(0.80, linestyle="--", linewidth=0.9, color="tab:red", label="AUC = 0.80 (identifiability threshold)")

for s in CHOSEN_SITES:
    ax.axvspan(s - 0.5, s + 0.5, color=COLORS[2], alpha=0.15)
ax.text(11.5, 0.815, "Chosen sites S = [9…14]", fontsize=8.5,
        ha="center", va="bottom", color="tab:green")

ax.set_xlabel("Residual-stream layer index (Meta-Llama-3-8B-Instruct, 32 layers)")
ax.set_ylabel("Linear-probe AUC (harmful vs benign, 128 held-out pairs)")
ax.set_ylim(0.78, 1.005)
ax.legend(frameon=False, loc="lower right", fontsize=8)

paths = save_fig(fig, "c1_auc_per_layer", OUT_DIR)
print("wrote:", paths)
