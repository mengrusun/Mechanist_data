"""C3 figure: pythia-1b formation trajectories — 3 stacked panels (behavioural, H*_personal-ablated, H*_attributed-ablated)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_plot_style import COLORS, save_fig
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(ROOT, "refine-logs/artifacts/formation/summary.json")) as f:
    S = json.load(f)

steps = S["steps"]
# offset step0 to a small positive value for log-x display
steps_plot = [max(s, 0.5) for s in steps]
traj = S["trajectories"]
em = S["emergences"]

TASK_COLORS = {"world_knowledge": COLORS[7], "personal_belief": COLORS[0], "attributed_belief": COLORS[3]}


def series(field, task):
    return [t[field][task] for t in traj]


fig, axes = plt.subplots(3, 1, figsize=(6.8, 7.2), sharex=True)

# (a) behavioural
ax = axes[0]
for task, label in [("world_knowledge", "world_knowledge"),
                    ("personal_belief", "personal_belief"),
                    ("attributed_belief", "attributed_belief")]:
    ax.plot(steps_plot, series("behavioural", task), marker="o", markersize=3.5,
            linewidth=1.4, label=label, color=TASK_COLORS[task])
ax.axhline(0.5, linestyle="--", linewidth=0.7, color="grey", alpha=0.7)
ax.axhline(0.6, linestyle=":", linewidth=0.7, color="black", alpha=0.6)
ax.text(steps_plot[-1] * 1.05, 0.605, "emergence\nthreshold 0.60", fontsize=7, va="bottom", ha="left")

ax.axvline(13000, linestyle="-.", linewidth=0.7, color=COLORS[0], alpha=0.7)
ax.text(13000, 0.03, "t* (personal)\n= 13000", fontsize=7, color=COLORS[0], ha="center", va="bottom")

# annotate step-2000 reorganization
ax.annotate("step 2000\npersonal 0.043\nattributed 0.971",
            xy=(2000, 0.043), xytext=(500, 0.30),
            arrowprops=dict(arrowstyle="->", lw=0.6, color="black"),
            fontsize=7, ha="left")

ax.set_ylabel("accuracy")
ax.set_ylim(-0.02, 1.05)
ax.set_title("(a) behavioural (no intervention)", fontsize=10, loc="left")
ax.legend(frameon=False, loc="lower right", ncol=1)

# (b) H*_personal ablated
ax = axes[1]
for task, label in [("personal_belief", "personal (target)"),
                    ("attributed_belief", "attributed (off-target)"),
                    ("world_knowledge", "world_knowledge (control)")]:
    ax.plot(steps_plot, series("hstar_personal_ablated", task), marker="s", markersize=3.5,
            linewidth=1.3, label=label, color=TASK_COLORS[task],
            linestyle="-" if task == "personal_belief" else "--")
ax.axvline(13000, linestyle="-.", linewidth=0.7, color=COLORS[0], alpha=0.7)
ax.text(13000, 0.03, "t† (personal)\n= 13000", fontsize=7, color=COLORS[0], ha="center", va="bottom")
ax.set_ylabel("accuracy (H*_personal ablated)")
ax.set_ylim(-0.02, 1.05)
ax.set_title("(b) with H*_personal ablated (target = personal)", fontsize=10, loc="left")
ax.legend(frameon=False, loc="lower right")

# (c) H*_attributed ablated
ax = axes[2]
for task, label in [("attributed_belief", "attributed (target)"),
                    ("personal_belief", "personal (off-target)"),
                    ("world_knowledge", "world_knowledge (control)")]:
    ax.plot(steps_plot, series("hstar_attributed_ablated", task), marker="^", markersize=3.5,
            linewidth=1.3, label=label, color=TASK_COLORS[task],
            linestyle="-" if task == "attributed_belief" else "--")
ax.axvline(33000, linestyle="-.", linewidth=0.7, color=COLORS[3], alpha=0.7)
ax.text(33000, 0.03, "t† (attributed)\n= 33000", fontsize=7, color=COLORS[3], ha="center", va="bottom")
ax.set_ylabel("accuracy (H*_attributed ablated)")
ax.set_ylim(-0.02, 1.05)
ax.set_title("(c) with H*_attributed ablated (target = attributed)", fontsize=10, loc="left")
ax.legend(frameon=False, loc="lower right")

ax.set_xscale("log")
ax.set_xlabel("training step (native log-spaced, 24 checkpoints — step0 shown at 0.5)")

fig.suptitle("Pythia-1B formation windows — personal [13000, 13000] vs attributed [0, 33000], distinct=TRUE",
             fontsize=10.5, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.98))

save_fig(fig, "c3_formation_trajectories", OUT_DIR, formats=("pdf", "png"))
