"""C2 figure: MT-Bench + MMLU capability preservation across variants."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paper_plot_style import plt, COLORS, save_fig

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(__file__)

variants = [
    ("B0", "artifacts/m5/B0_mtbench.json", "artifacts/m5/B0_mmlu.json", COLORS[0]),
    ("B1", "artifacts/m5/B1_mtbench.json", "artifacts/m5/B1_mmlu.json", COLORS[1]),
    ("RR broken", "artifacts/m5/RR_mtbench.json", "artifacts/m5/RR_mmlu.json", COLORS[3]),
    ("RR iter-5", "artifacts/iteration_round_5/m5/RR_mtbench.json", "artifacts/iteration_round_5/m5/RR_mmlu.json", COLORS[2]),
]

names = []
mtbench = []
mmlu = []
colors = []
for name, mt_rel, mm_rel, color in variants:
    with open(os.path.join(ROOT, mt_rel)) as f:
        mt = json.load(f)["avg_score"]
    with open(os.path.join(ROOT, mm_rel)) as f:
        mm = json.load(f)["accuracy"]
    names.append(name)
    mtbench.append(mt)
    mmlu.append(mm)
    colors.append(color)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.4))

bars1 = ax1.bar(names, mtbench, color=colors)
ax1.axhline(6.30 - 0.30, linestyle="--", linewidth=0.9, color="tab:red", label="plan gate: ≥ B0 − 0.3")
ax1.set_ylabel("MT-Bench avg score (1–10)")
ax1.set_ylim(0, 10)
ax1.legend(frameon=False, fontsize=8, loc="upper right")
for b, v in zip(bars1, mtbench):
    ax1.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

bars2 = ax2.bar(names, mmlu, color=colors)
ax2.axhline(0.58 - 0.02, linestyle="--", linewidth=0.9, color="tab:red", label="plan gate: ≥ B0 − 0.02")
ax2.set_ylabel("MMLU 5-shot accuracy")
ax2.set_ylim(0, 1)
ax2.legend(frameon=False, fontsize=8, loc="upper right")
for b, v in zip(bars2, mmlu):
    ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)

for ax in (ax1, ax2):
    ax.tick_params(axis="x", labelrotation=15)

paths = save_fig(fig, "c2_capability_preservation", OUT_DIR)
print("wrote:", paths)
