"""C3a figure — per-layer |cos(v_c, v_v)| trajectory with random-direction null."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figures"))

from paper_plot_style import plt, COLORS, save_fig  # noqa: E402

OUT_STEM = str(Path(__file__).parent / "c3a_cos_trajectory")

with open(ROOT / "artifacts" / "cos_trajectory.json") as f:
    traj = json.load(f)
with open(ROOT / "artifacts" / "nulls.json") as f:
    nulls = json.load(f)

L_star = traj["L_star"]
layers = [row["layer"] for row in traj["per_layer"]]
abs_cos = [row["abs_cos"] for row in traj["per_layer"]]

rand_mean = nulls["random_direction_null"]["mean_abs_cos_random"]
theo_std = nulls["random_direction_null"]["theoretical_std"]
shuf_cos = nulls["shuffled_label_probe_c"]["abs_cos_vs_v_c_star"]

fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2))
# random-direction null band (mean ± 3 * theoretical_std)
ax.axhspan(max(0.0, rand_mean - 3 * theo_std), rand_mean + 3 * theo_std,
           color="grey", alpha=0.15,
           label=f"random-dir null band (mean {rand_mean:.3f} ± 3·std)")
ax.axhline(rand_mean, color="grey", linestyle=":", linewidth=1.0)
ax.plot(layers, abs_cos, marker="o", markersize=3, color=COLORS[0], linewidth=1.5,
        label="|cos(v_c, v_v)|")
ax.axhline(shuf_cos, color=COLORS[7], linestyle="--", linewidth=1.0,
           label=f"shuffled-label pair ({shuf_cos:.3f})")
ax.axhline(0.30, color=COLORS[3], linestyle="-.", linewidth=0.8,
           label="C3a threshold (0.30)")
ax.axvline(L_star, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
# Mark L* with the exact value annotation
lstar_val = traj.get("abs_cos_at_Lstar", abs_cos[L_star] if L_star < len(abs_cos) else None)
if lstar_val is not None:
    ax.text(L_star, lstar_val + 0.008, f" L*={L_star}\n |cos|={lstar_val:.3f}",
            fontsize=8, va="bottom", ha="left")

ax.set_xlabel("Layer (residual-stream block index)")
ax.set_ylabel("|cos(v_c, v_v)|")
ax.set_ylim(-0.005, 0.32)
ax.set_xlim(-1, max(layers) + 1)
ax.legend(frameon=False, loc="upper right", fontsize=8)

save_fig(fig, OUT_STEM, formats=("pdf", "png"))
