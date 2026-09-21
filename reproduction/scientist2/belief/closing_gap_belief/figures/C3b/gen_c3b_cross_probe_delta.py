"""C3b figure — Δ cross-probe readout under steering (source vs matched-random control)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "figures"))

from paper_plot_style import plt, COLORS, save_fig  # noqa: E402

OUT_STEM = str(Path(__file__).parent / "c3b_cross_probe_delta")

with open(ROOT / "artifacts" / "steering_results.json") as f:
    st = json.load(f)

# Four cross-direction tests: turn1 v_c → Δv_v (pos/neg), turn2 v_v → Δv_c (pos/neg).
tests = [
    ("v_c→Δv_v (α=+1σ)", "turn1_v_c_steer_delta_v_readout_pos"),
    ("v_c→Δv_v (α=-1σ)", "turn1_v_c_steer_delta_v_readout_neg"),
    ("v_v→Δv_c (α=+1σ)", "turn2_v_v_steer_delta_c_readout_pos"),
    ("v_v→Δv_c (α=-1σ)", "turn2_v_v_steer_delta_c_readout_neg"),
]
decision = st["c3b_decision"]

labels = [t[0] for t in tests]
delta_source = [decision[t[1]]["delta_source"] for t in tests]
delta_random = [decision[t[1]]["delta_random"] for t in tests]
thresholds  = [decision[t[1]]["threshold_absolute"] for t in tests]

fig, ax = plt.subplots(1, 1, figsize=(6.0, 3.4))
import numpy as np
xs = np.arange(len(labels))
width = 0.36
b1 = ax.bar(xs - width/2, delta_source, width, color=COLORS[0], label="Δ source (v_c or v_v)")
b2 = ax.bar(xs + width/2, delta_random, width, color="grey", alpha=0.7,
            label="Δ matched-random control")

# Threshold markers (0.5·σ_probe on each cross axis)
for i, (b_pair, thr) in enumerate(zip(zip(b1, b2), thresholds)):
    ax.hlines(thr, xs[i] - width, xs[i] + width, colors=COLORS[3],
              linestyles="dashdot", linewidth=1.0,
              label=("threshold 0.5·σ_probe" if i == 0 else None))

# Value labels
for b, v in zip(b1, delta_source):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8)
for b, v in zip(b2, delta_random):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=8)

ax.set_xticks(xs)
ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
ax.set_ylabel("Δ cross-probe readout (absolute)")
ax.set_ylim(0.0, max(max(delta_source), max(delta_random), max(thresholds)) * 1.15)
ax.legend(frameon=False, loc="upper left", fontsize=8)

save_fig(fig, OUT_STEM, formats=("pdf", "png"))
