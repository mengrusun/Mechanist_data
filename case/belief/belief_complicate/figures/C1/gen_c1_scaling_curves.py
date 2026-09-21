"""Generate C1 scaling-curves figure (grouped bar) — Pythia scale × belief frame accuracy."""
import json, os, sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paper_plot_style import plt, COLORS, save_fig  # noqa: E402
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(PROJECT_ROOT, "runs/summary/all_results.json")) as f:
    ar = json.load(f)

models = ["pythia-410m", "pythia-1b", "pythia-2.8b"]
frames = ["world_knowledge", "personal_belief", "attributed_belief"]
frame_labels = {"world_knowledge": "world knowledge (control)",
                "personal_belief": "personal belief",
                "attributed_belief": "attributed belief"}

data = np.zeros((len(models), len(frames)))
above_chance = np.zeros((len(models), len(frames)), dtype=bool)
for i, m in enumerate(models):
    for j, fr in enumerate(frames):
        key = f"{m}_{fr}"
        rec = ar["M1"][key]
        data[i, j] = rec["third_person_accuracy"]
        above_chance[i, j] = rec["above_chance_third_person"]

fig, ax = plt.subplots(figsize=(6.5, 3.5))
bar_w = 0.25
x = np.arange(len(models))
for j, fr in enumerate(frames):
    offset = (j - 1) * bar_w
    bars = ax.bar(x + offset, data[:, j], bar_w,
                  color=COLORS[j], label=frame_labels[fr], edgecolor="white", linewidth=0.4)
    for i, b in enumerate(bars):
        y = b.get_height()
        # Value label
        ax.text(b.get_x() + b.get_width() / 2, y + 0.015,
                f"{data[i, j]:.2f}", ha="center", va="bottom", fontsize=7)
        # * marker if above chance
        if above_chance[i, j]:
            ax.text(b.get_x() + b.get_width() / 2, y - 0.05,
                    "*", ha="center", va="bottom", fontsize=11, color="white", fontweight="bold")

ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.7, alpha=0.7)
ax.text(len(models) - 0.5, 0.51, "chance (0.5)", fontsize=7, color="grey", ha="right", va="bottom")

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("accuracy (third-person 454-item subset)")
ax.set_xlabel("Pythia model size")
ax.set_ylim(0.0, 1.10)
ax.legend(frameon=False, loc="upper left", ncol=1, fontsize=8)
ax.set_yticks(np.arange(0.0, 1.01, 0.2))

# Annotation for the AB step-function
ax.annotate("", xy=(1 + 0 * bar_w, 0.86), xytext=(0 + 0 * bar_w, 0.46),
            arrowprops=dict(arrowstyle="->", color="tab:brown", lw=0.8, alpha=0.6))
ax.text(0.5, 0.66, "AB step-function\n+0.40 abs @ 1B",
        fontsize=7, color="tab:brown", ha="center", va="center", alpha=0.85)

save_fig(fig, "c1_scaling_curves", formats=("pdf", "png"), out_dir=OUT_DIR)
plt.close(fig)

# Write INDEX.json
index = {
    "claim_id": "C1",
    "claim_title": "Scale-Dependent Emergence — personal vs attributed belief across Pythia sizes",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "figures": [
        {
            "id": "c1_scaling_curves",
            "type": "grouped_bar",
            "caption": (
                "Personal-belief and attributed-belief accuracy across Pythia scales (third-person 454-item subset). "
                "Attributed-belief shows a sharp step-function emergence at 1B (0.460 -> 0.859 -> 0.905); "
                "personal-belief is non-monotonic with a dip at 1B (0.855 -> 0.782 -> 0.993). "
                "White asterisk marks above-chance (p < 0.05, one-sided binomial). "
                "pythia-410m attributed-belief sits at chance (acc = 0.460, p = 0.96)."
            ),
            "png": "figures/C1/c1_scaling_curves.png",
            "pdf": "figures/C1/c1_scaling_curves.pdf",
            "md": None,
            "tex": None,
            "source_data": "runs/summary/all_results.json (M1.*.third_person_accuracy)",
            "status": "ok",
        },
    ],
    "skipped": [],
}
with open(os.path.join(OUT_DIR, "INDEX.json"), "w") as f:
    json.dump(index, f, indent=2)
print("C1: 1/1 figures generated")
