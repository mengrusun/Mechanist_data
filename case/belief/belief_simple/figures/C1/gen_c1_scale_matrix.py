"""C1 figure: behavioural scaling matrix — pythia-{410m, 1b, 2.8b} × {WK, personal, attributed} with Wilson CI."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_plot_style import COLORS, save_fig
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = ["pythia-410m", "pythia-1b", "pythia-2.8b"]
TASKS = ["world_knowledge", "personal_belief", "attributed_belief"]
TASK_LABEL = {"world_knowledge": "world_knowledge", "personal_belief": "personal_belief", "attributed_belief": "attributed_belief"}


def load(model, task):
    with open(os.path.join(ROOT, "refine-logs/artifacts/behavioral", model, f"{task}.json")) as f:
        d = json.load(f)
    return d["acc"], d["wilson_ci_low"], d["wilson_ci_high"]


data = {(m, t): load(m, t) for m in MODELS for t in TASKS}

fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.6))

n_models = len(MODELS)
n_tasks = len(TASKS)
bar_w = 0.25
x = np.arange(n_models)

task_colors = {"world_knowledge": COLORS[7], "personal_belief": COLORS[0], "attributed_belief": COLORS[3]}

for j, t in enumerate(TASKS):
    accs = [data[(m, t)][0] for m in MODELS]
    lows = [data[(m, t)][0] - data[(m, t)][1] for m in MODELS]
    highs = [data[(m, t)][2] - data[(m, t)][0] for m in MODELS]
    xj = x + (j - 1) * bar_w
    bars = ax.bar(xj, accs, bar_w, label=TASK_LABEL[t], color=task_colors[t],
                  yerr=[lows, highs], capsize=3, edgecolor="black", linewidth=0.4)
    for xi, a in zip(xj, accs):
        ax.text(xi, a + 0.02, f"{a:.2f}", ha="center", va="bottom", fontsize=7.5)

# chance line
ax.axhline(0.5, linestyle="--", linewidth=0.8, color="grey", alpha=0.8)
ax.text(x[-1] + bar_w * 1.6, 0.51, "chance = 0.5", fontsize=7.5, color="grey", va="bottom", ha="right")

# highlight pythia-410m x attributed (below chance)
below_x = x[0] + bar_w
ax.annotate("below\nchance",
            xy=(below_x, data[("pythia-410m", "attributed_belief")][0]),
            xytext=(below_x - 0.05, 0.22),
            ha="center", fontsize=7.5, color=COLORS[3],
            arrowprops=dict(arrowstyle="->", color=COLORS[3], lw=0.6))

# gap annotations
ax.annotate("", xy=(x[0] - bar_w, 0.852), xytext=(x[0] + bar_w, 0.457),
            arrowprops=dict(arrowstyle="<->", color="black", lw=0.5))
ax.text(x[0] + 0.02, 0.66, "39 pp", fontsize=8, ha="left", fontstyle="italic")

ax.annotate("", xy=(x[2] - bar_w, 0.994), xytext=(x[2] + bar_w, 0.796),
            arrowprops=dict(arrowstyle="<->", color="black", lw=0.5))
ax.text(x[2] + 0.02, 0.90, "20 pp", fontsize=8, ha="left", fontstyle="italic")

ax.set_xticks(x)
ax.set_xticklabels(MODELS)
ax.set_ylabel("accuracy (Wilson 95% CI)")
ax.set_ylim(0.0, 1.05)
ax.legend(frameon=False, loc="lower right", ncol=1)

save_fig(fig, "c1_scale_matrix", OUT_DIR, formats=("pdf", "png"))
