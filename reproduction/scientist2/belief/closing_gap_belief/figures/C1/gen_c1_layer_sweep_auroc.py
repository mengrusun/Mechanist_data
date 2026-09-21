"""C1 figure — per-layer AUROC of the correctness probe vs. nulls."""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figures"))

from paper_plot_style import plt, COLORS, save_fig  # noqa: E402

OUT_STEM = str(Path(__file__).parent / "c1_layer_sweep_auroc")

with open(ROOT / "artifacts" / "probe_metrics.json") as f:
    pm = json.load(f)
with open(ROOT / "artifacts" / "nulls.json") as f:
    nulls = json.load(f)

per_layer = pm["per_layer"]
layers = sorted(int(k) for k in per_layer.keys())
auc_c = [per_layer[str(L)]["probe_c_binary"]["auc_test"] for L in layers]
L_star = pm["L_star"]

shuf_auc = nulls["shuffled_label_probe_c"]["auc_test"]
# random-direction expected AUC ≈ 0.5 for a binary probe
rand_auc = 0.5

fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2))
ax.plot(layers, auc_c, marker="o", markersize=3, color=COLORS[0], linewidth=1.5,
        label="probe_c (learned)")
ax.axhline(shuf_auc, color=COLORS[7], linestyle="--", linewidth=1.0,
           label=f"shuffled-label null (AUC={shuf_auc:.3f})")
ax.axhline(rand_auc, color="grey", linestyle=":", linewidth=1.0,
           label="random-direction null (0.50)")
ax.axhline(0.70, color=COLORS[3], linestyle="-.", linewidth=0.8,
           label="C1 threshold (0.70)")
ax.axvline(L_star, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(L_star, 0.505, f" L*={L_star}", fontsize=8, va="bottom", ha="left")

ax.set_xlabel("Layer (residual-stream block index)")
ax.set_ylabel("AUROC (test)")
ax.set_ylim(0.45, 0.90)
ax.set_xlim(-1, max(layers) + 1)
ax.legend(frameon=False, loc="upper left")

save_fig(fig, OUT_STEM, formats=("pdf", "png"))
