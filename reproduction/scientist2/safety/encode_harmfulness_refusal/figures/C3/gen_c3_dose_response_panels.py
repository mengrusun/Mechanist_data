"""C3 multi-panel dose-response — 2x2 grid: (h target, h off-target, r target, r off-target).

Data: results/m3/claim3_dose_response.csv
Output: figures/C3/c3_dose_response_panels.{pdf,png}
"""
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_plot_style import plt, save_fig, COLORS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA = os.path.join(PROJECT_ROOT, "results/m3/claim3_dose_response.csv")

# direction -> alpha -> row
by_direction = defaultdict(dict)
with open(DATA) as f:
    for r in csv.DictReader(f):
        by_direction[r["direction"]][float(r["alpha"])] = r


def series(direction, key):
    alphas = sorted(by_direction[direction].keys())
    ys = [float(by_direction[direction][a][key]) for a in alphas]
    return alphas, ys


DIR_COLORS = {"h": COLORS[0], "r": COLORS[1], "random": COLORS[2], "swap": COLORS[3]}
DIR_LABELS = {"h": "true h", "r": "true r", "random": "random (matched norm)", "swap": "swap"}

fig, axes = plt.subplots(2, 2, figsize=(8, 5.5), sharex=True)

# Panel (a): h-direction, target axis = h-readout on harmful
ax = axes[0, 0]
for d in ["h", "random", "swap"]:
    x, y = series(d, "h_readout_mean_harm")
    ax.plot(x, y, marker="o", markersize=4, label=DIR_LABELS[d], color=DIR_COLORS[d], linewidth=1.5)
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
ax.set_ylabel("h-readout (harmful)")
ax.set_title("(a) steer along h  →  target: h-readout ↑↓ monotone", fontsize=10)
ax.legend(frameon=False, loc="upper left", fontsize=8)

# Panel (b): h-direction, off-target axis = refusal on benign
ax = axes[0, 1]
for d in ["h", "random", "swap"]:
    x, y = series(d, "refusal_rate_ben")
    ax.plot(x, y, marker="o", markersize=4, label=DIR_LABELS[d], color=DIR_COLORS[d], linewidth=1.5)
ax.axhline(0.02, color="red", linestyle="--", linewidth=0.8, label=r"$\epsilon_{\mathrm{null}}$")
ax.set_ylabel("refusal rate (benign)")
ax.set_title("(b) steer along h  →  off-target: refusal in null-band", fontsize=10)
ax.set_ylim(-0.05, 1.05)
ax.legend(frameon=False, loc="upper right", fontsize=8)

# Panel (c): r-direction, target axis = refusal on benign
ax = axes[1, 0]
for d in ["r", "random", "swap"]:
    x, y = series(d, "refusal_rate_ben")
    ax.plot(x, y, marker="o", markersize=4, label=DIR_LABELS[d], color=DIR_COLORS[d], linewidth=1.5)
ax.set_xlabel(r"$\alpha$ (in units of $\|d\|$)")
ax.set_ylabel("refusal rate (benign)")
ax.set_title("(c) steer along r  →  target: refusal threshold at α=+2", fontsize=10)
ax.set_ylim(-0.05, 1.05)
ax.legend(frameon=False, loc="upper left", fontsize=8)

# Panel (d): r-direction, off-target axis = h-readout on harmful
ax = axes[1, 1]
for d in ["r", "random", "swap"]:
    x, y = series(d, "h_readout_mean_harm")
    ax.plot(x, y, marker="o", markersize=4, label=DIR_LABELS[d], color=DIR_COLORS[d], linewidth=1.5)
ax.axhline(0, color="gray", linestyle=":", linewidth=0.8)
ax.set_xlabel(r"$\alpha$ (in units of $\|d\|$)")
ax.set_ylabel("h-readout (harmful)")
ax.set_title("(d) steer along r  →  off-target: h-readout unchanged", fontsize=10)
ax.legend(frameon=False, loc="upper right", fontsize=8)

fig.tight_layout()
save_fig(fig, "c3_dose_response_panels", HERE)
plt.close(fig)
