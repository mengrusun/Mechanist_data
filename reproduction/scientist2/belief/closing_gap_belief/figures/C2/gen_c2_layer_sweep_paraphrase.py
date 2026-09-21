"""C2 figure — per-layer AUROC of the verbalized-confidence probe under paraphrases."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figures"))

from paper_plot_style import plt, COLORS, save_fig  # noqa: E402

OUT_STEM = str(Path(__file__).parent / "c2_layer_sweep_paraphrase")

with open(ROOT / "artifacts" / "probe_metrics.json") as f:
    pm = json.load(f)
with open(ROOT / "artifacts" / "paraphrase.json") as f:
    para = json.load(f)

per_layer = pm["per_layer"]
layers = sorted(int(k) for k in per_layer.keys())
auc_p0 = [per_layer[str(L)]["probe_v_binary"]["auc_test"] for L in layers]
L_star = pm["L_star"]

# Paraphrase results are at L* only (single scalar per paraphrase — plot as horizontal marker line)
p1_auc = para["P1"]["auc_v_binarized"]
p2_auc = para["P2"]["auc_v_binarized"]

fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2))
ax.plot(layers, auc_p0, marker="o", markersize=3, color=COLORS[0], linewidth=1.5,
        label="P0 primary (all layers)")
ax.scatter([L_star], [p1_auc], marker="s", s=45, color=COLORS[1],
           label=f"P1 Tian 0-1 (at L*, Δ={para['P1']['delta_auc']:+.2f})", zorder=5)
ax.scatter([L_star], [p2_auc], marker="^", s=45, color=COLORS[2],
           label=f"P2 Likert (at L*, Δ={para['P2']['delta_auc']:+.2f})", zorder=5)
ax.axhline(0.70, color=COLORS[3], linestyle="-.", linewidth=0.8,
           label="C2 threshold (0.70)")
ax.axvline(L_star, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
ax.text(L_star, 0.505, f" L*={L_star}", fontsize=8, va="bottom", ha="left")

ax.set_xlabel("Layer (residual-stream block index)")
ax.set_ylabel("AUROC (probe_v_binary, test)")
ax.set_ylim(0.45, 1.00)
ax.set_xlim(-1, max(layers) + 1)
ax.legend(frameon=False, loc="lower right")

save_fig(fig, OUT_STEM, formats=("pdf", "png"))
