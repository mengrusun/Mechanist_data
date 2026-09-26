"""Line plot: widened α ∈ [-3, +3] gap-closure sweep on treated seed100 (n=27 held-out).

Reads mechanism/M2_causal_widened/gap_closure_widened.json.
Shows real m1_top_k vs random_matched steering curves plus ablation & patching (α=0) refs.
"""
import json
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from paper_plot_style import COLORS, save_fig  # noqa: E402

DATA = "mechanism/M2_causal_widened/gap_closure_widened.json"
OUT_DIR = "figures/C3_v2"
STEM = "c3v2_widened_alpha_gap_closure"

with open(DATA) as f:
    d = json.load(f)

alphas = sorted(float(a) for a in d["qa_real"].keys())


def get_gc(bucket, a):
    key = str(a)
    if key in bucket:
        return bucket[key]["gap_closure"]
    return None  # skip missing point


real = [d["qa_real"][str(a)]["gap_closure"] for a in alphas]
rand_alphas = [a for a in alphas if str(a) in d["qa_random"]]
rand = [d["qa_random"][str(a)]["gap_closure"] for a in rand_alphas]

# Ablation & patching gap-closure from the original m2 (α=0, single value each)
with open("mechanism/M2_causal/gap_closure.json") as f:
    orig = json.load(f)


def find_gc(kind):
    for row in orig.get("summary_flat", orig.get("rows", [])):
        if row.get("shape") == kind:
            return row.get("gap_closure")
    return None


gc_ablation = 0.875  # from EXPERIMENT_RESULTS.md M2 table
gc_patching = 0.625  # from EXPERIMENT_RESULTS.md M2 table

fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.4))
ax.plot(alphas, real, "o-", color=COLORS[0], label="steering — real m1_top_k")
ax.plot(rand_alphas, rand, "s--", color=COLORS[3], label="steering — random_matched")
ax.axhline(gc_ablation, linestyle=":", linewidth=1.0, color=COLORS[2],
           label=f"ablation (α=0): gc={gc_ablation:.3f}")
ax.axhline(gc_patching, linestyle=":", linewidth=1.0, color=COLORS[4],
           label=f"patching  (α=0): gc={gc_patching:.3f}")
ax.axhline(0.0, linewidth=0.6, color="black")

ax.set_xlabel("Steering coefficient α  (in units of σ_proj)")
ax.set_ylabel("Gap-closure toward Ctrl  (0 = treated, 1 = Ctrl)")
ax.set_xticks(sorted(alphas))
ax.set_ylim(-0.05, 1.0)
ax.legend(frameon=False, loc="upper left", fontsize=8)

save_fig(fig, STEM, OUT_DIR)
plt.close(fig)
