"""C2 figure: HarmBench per-category ASR across variants (B0, B1, RR, RR-iter5)."""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paper_plot_style import plt, COLORS, save_fig

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(__file__)

variants = [
    ("B0 (refusal-only)", "artifacts/m5/B0_harmbench.json", COLORS[0]),
    ("B1 (R2D2-lite adv-train)", "artifacts/m5/B1_harmbench.json", COLORS[1]),
    ("RR (cos² loss — broken)", "artifacts/m5/RR_harmbench.json", COLORS[3]),
    ("RR iter-5 (signed_hinge + refuse-CE)", "artifacts/iteration_round_5/m5/RR_harmbench.json", COLORS[2]),
]

data = []
for name, rel, color in variants:
    with open(os.path.join(ROOT, rel)) as f:
        d = json.load(f)
    data.append((name, d, color))

cats = ["direct", "gcg-lite", "persona", "hypothetical", "suffix-injection", "human-redteam"]
n_cats = len(cats)
n_var = len(variants)

fig, ax = plt.subplots(1, 1, figsize=(9, 3.5))
w = 0.18
x = list(range(n_cats))
for i, (name, d, color) in enumerate(data):
    ys = [d["attack_category_asr"][c]["asr"] for c in cats]
    offset = (i - (n_var - 1) / 2) * w
    ax.bar([xi + offset for xi in x], ys, width=w, color=color, label=name)

ax.set_xticks(x)
ax.set_xticklabels(cats, rotation=15, ha="right")
ax.set_ylabel("Attack Success Rate (30 prompts / category)")
ax.set_ylim(0, 0.65)
ax.legend(frameon=False, ncol=2, fontsize=7.5, loc="upper right")

for i, (name, d, _) in enumerate(data):
    agg = d["aggregate_asr"]
    xpos = 5.5 + (i - (n_var - 1) / 2) * 0.4
    ax.text(xpos, 0.60 - i * 0.03, f"agg={agg:.3f}", fontsize=7, ha="right")

paths = save_fig(fig, "c2_harmbench_by_variant", OUT_DIR)
print("wrote:", paths)
