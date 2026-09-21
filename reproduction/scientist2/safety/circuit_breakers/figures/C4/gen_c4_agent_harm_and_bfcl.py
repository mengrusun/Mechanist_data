"""C4 figure: harmful_tool_use_rate + BFCL across B0 / RR-broken / RR-iter5."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paper_plot_style import plt, COLORS, save_fig

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(__file__)

variants = [
    ("B0", "artifacts/m7/B0_agent.json", COLORS[0]),
    ("RR (cos² — broken)", "artifacts/m7/RR_agent.json", COLORS[3]),
    ("RR iter-5 (signed_hinge + refuse-CE)", "artifacts/iteration_round_5/m7/RR_agent.json", COLORS[2]),
]

names = []
harm = []
bfcl = []
colors = []
for name, rel, color in variants:
    with open(os.path.join(ROOT, rel)) as f:
        d = json.load(f)
    names.append(name)
    harm.append(d["harmful_tool_use_rate"])
    bfcl.append(d["bfcl_score"])
    colors.append(color)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.5, 3.4))

bars1 = ax1.bar(names, harm, color=colors)
ax1.set_ylabel("harmful_tool_use_rate (100-prompt harm set)")
ax1.set_ylim(0, max(harm) * 1.4 + 0.02)
for b, v in zip(bars1, harm):
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

bars2 = ax2.bar(names, bfcl, color=colors)
ax2.axhline(0.98 - 0.03, linestyle="--", linewidth=0.9, color="tab:red", label="plan gate: ≥ B0 − 0.03")
ax2.set_ylabel("BFCL exec_simple score")
ax2.set_ylim(0.90, 1.02)
ax2.legend(frameon=False, fontsize=8, loc="lower right")
for b, v in zip(bars2, bfcl):
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

for ax in (ax1, ax2):
    ax.tick_params(axis="x", labelrotation=15)
    for lab in ax.get_xticklabels():
        lab.set_ha("right")
        lab.set_fontsize(8)

paths = save_fig(fig, "c4_agent_harm_and_bfcl", OUT_DIR)
print("wrote:", paths)
