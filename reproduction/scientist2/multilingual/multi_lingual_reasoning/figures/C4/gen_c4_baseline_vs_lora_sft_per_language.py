"""C4 grouped-bar: per-language MGSM accuracy — untuned baseline vs LoRA-SFT.
Source: refine-logs/EXPERIMENT_RESULTS.md#M4a (per-language table)."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import plt, COLORS, save_fig
import numpy as np

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

langs   = ["en", "es", "fr", "de", "zh", "ja", "ru", "th", "te", "bn", "sw"]
labels  = ["En", "Es", "Fr", "De", "Zh", "Ja", "Ru", "Th", "Te", "Bn", "Sw"]
base    = [0.92, 0.88, 0.88, 0.84, 0.88, 0.78, 0.90, 0.72, 0.56, 0.76, 0.26]
lora    = [0.76, 0.62, 0.74, 0.66, 0.58, 0.50, 0.66, 0.52, 0.34, 0.54, 0.22]

# macro summary bar at right
labels_full = labels + ["macro"]
base_full = base + [sum(base) / len(base)]
lora_full = lora + [sum(lora) / len(lora)]

x = np.arange(len(labels_full))
width = 0.36

fig, ax = plt.subplots(1, 1, figsize=(8.5, 3.5))
b1 = ax.bar(x - width / 2, base_full, width, color=COLORS[0], label="Qwen-3-4B-Thinking baseline (no LoRA)")
b2 = ax.bar(x + width / 2, lora_full, width, color=COLORS[3], label="LoRA-SFT (r=32, α=32, q/k/v/o, lr=2e-4, 5001 ex, 1 epoch)")

# baseline reference line for the macro bar's baseline
ax.axhline(base_full[-1], color=COLORS[0], linewidth=0.5, linestyle=":", alpha=0.6)

ax.set_xticks(x)
ax.set_xticklabels(labels_full)
ax.set_ylabel("MGSM accuracy")
ax.set_ylim(0.0, 1.0)
ax.legend(frameon=False, loc="upper right")

# value labels on the bars for the macro col — highlighting the 20 pp drop
for bar, val in zip((b1[-1], b2[-1]), (base_full[-1], lora_full[-1])):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
            f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

# vertical separator between per-language and macro
ax.axvline(len(labels) - 0.5, color="gray", linewidth=0.4, linestyle="-", alpha=0.4)

plt.tight_layout()
save_fig(fig, "c4_baseline_vs_lora_sft_per_language",
         formats=("pdf", "png"), out_dir=OUT_DIR)
print("wrote c4_baseline_vs_lora_sft_per_language.pdf and .png")
