"""C1 figure: Δcos_harmful / Δcos_benign / Δcos_ctrl — base run vs iter-5.

Base data:  artifacts/m4/activation_drift.json
Iter-5:     artifacts/iteration_round_5/m4/activation_drift.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paper_plot_style import plt, COLORS, save_fig

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(__file__)

with open(os.path.join(ROOT, "artifacts/m4/activation_drift.json")) as f:
    base = json.load(f)
with open(os.path.join(ROOT, "artifacts/iteration_round_5/m4/activation_drift.json")) as f:
    iter5 = json.load(f)

metrics = ["Δcos_harmful", "Δcos_benign", "Δcos_ctrl (random)"]
base_vals = [base["delta_cos_harmful"], base["delta_cos_benign"], base["delta_cos_harmful_ctrl"]]
iter5_vals = [iter5["delta_cos_harmful"], iter5["delta_cos_benign"], iter5["delta_cos_harmful_ctrl"]]

fig, ax = plt.subplots(1, 1, figsize=(6, 3.5))
x = list(range(len(metrics)))
w = 0.35
bars0 = ax.bar([i - w / 2 for i in x], base_vals, width=w, color=COLORS[0], label="Base (cos² loss)")
bars1 = ax.bar([i + w / 2 for i in x], iter5_vals, width=w, color=COLORS[2], label="Iter-5 (signed_hinge + refusal-CE)")

ax.axhline(-0.30, linestyle="--", linewidth=0.9, color="tab:red", label="Δcos_harmful target ≤ -0.30")
ax.axhline(0, linewidth=0.8, color="black")
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylabel("Mean Δcos vs base activation")
ax.set_ylim(-0.55, 0.10)
ax.legend(frameon=False, loc="lower right", fontsize=8)

for rects, vals in [(bars0, base_vals), (bars1, iter5_vals)]:
    for rect, v in zip(rects, vals):
        ha = "center"
        va = "bottom" if v >= 0 else "top"
        offset = 0.012 if v >= 0 else -0.012
        ax.text(rect.get_x() + rect.get_width() / 2, v + offset, f"{v:.3f}", ha=ha, va=va, fontsize=7.5)

paths = save_fig(fig, "c1_reroute_before_after", OUT_DIR)
print("wrote:", paths)
