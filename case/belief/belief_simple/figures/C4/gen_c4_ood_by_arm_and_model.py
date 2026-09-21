"""C4 figure: 2-panel grouped bar — pythia-1b + pythia-2.8b × task × arm on belief_holdout OOD."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paper_plot_style import COLORS, save_fig
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = ["pythia-1b", "pythia-2.8b"]
TASKS = ["world_knowledge", "personal_belief", "attributed_belief"]
ARMS = ["baseline_no_control", "controller", "prompt_hint"]
ARM_LABEL = {"baseline_no_control": "baseline", "controller": "controller", "prompt_hint": "prompt-hint"}
ARM_COLORS = {"baseline_no_control": COLORS[7], "controller": COLORS[2], "prompt_hint": COLORS[1]}


def load(model, arm):
    fname = {"baseline_no_control": "ood_baseline_no_control.json",
             "controller": "ood_controller.json",
             "prompt_hint": "ood_prompt_hint.json"}[arm]
    with open(os.path.join(ROOT, "refine-logs/artifacts/controller", model, fname)) as f:
        d = json.load(f)
    return {t: d["per_task"][t]["acc"] for t in TASKS}


def load_report(model):
    with open(os.path.join(ROOT, "refine-logs/artifacts/controller", model, "M4_report.json")) as f:
        return json.load(f)


data = {(m, a): load(m, a) for m in MODELS for a in ARMS}
reports = {m: load_report(m) for m in MODELS}

fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), sharey=True)

for pi, m in enumerate(MODELS):
    ax = axes[pi]
    x = np.arange(len(TASKS))
    bar_w = 0.26
    for j, a in enumerate(ARMS):
        accs = [data[(m, a)][t] for t in TASKS]
        xj = x + (j - 1) * bar_w
        bars = ax.bar(xj, accs, bar_w, label=ARM_LABEL[a], color=ARM_COLORS[a],
                      edgecolor="black", linewidth=0.4)
        for xi, acc in zip(xj, accs):
            ax.text(xi, acc + 0.012, f"{acc:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(["WK", "personal", "attributed"], fontsize=9)
    ax.set_ylim(0, 1.10)
    ax.set_title(f"({chr(97+pi)}) {m}   controller net_impr = {reports[m]['controller_vs_baseline']['net_improvement']:+d}   |   "
                 f"prompt-hint net_impr = {reports[m]['prompt_hint_vs_baseline']['net_improvement']:+d}",
                 fontsize=9.5, loc="left")
    if pi == 0:
        ax.set_ylabel("OOD accuracy (belief_holdout)")
    ax.axhline(0.5, linestyle="--", linewidth=0.7, color="grey", alpha=0.6)

axes[0].legend(frameon=False, loc="lower right", ncol=3)

fig.suptitle("Probe-and-amplify controller on belief_holdout OOD (n=367+1101+1101=2569); WK accuracy exactly preserved across arms",
             fontsize=10, y=1.02)
fig.tight_layout()
save_fig(fig, "c4_ood_by_arm_and_model", OUT_DIR, formats=("pdf", "png"))
