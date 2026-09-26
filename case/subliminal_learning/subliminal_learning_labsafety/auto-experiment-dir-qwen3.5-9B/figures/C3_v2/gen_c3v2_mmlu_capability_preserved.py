"""Line plot: MMLU-slice accuracy vs α on treated seed100.

Reads mechanism/M2_causal_widened/mmlu_specificity.json.
Shows that intervention preserves general capability (max |Δ| ~ 1.0 pp) → SP-C PASSES.
"""
import json
import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from paper_plot_style import COLORS, save_fig  # noqa: E402

DATA = "mechanism/M2_causal_widened/mmlu_specificity.json"
OUT_DIR = "figures/C3_v2"
STEM = "c3v2_mmlu_capability_preserved"

with open(DATA) as f:
    d = json.load(f)

alphas = sorted(float(a) for a in d["mmlu_real"].keys())
real = [d["mmlu_real"][str(a)]["acc_intervened"] for a in alphas]
rand_alphas = [a for a in alphas if str(a) in d["mmlu_random"]]
rand = [d["mmlu_random"][str(a)]["acc_intervened"] for a in rand_alphas]
baseline = d.get("mmlu_treated_baseline_acc")

fig, ax = plt.subplots(1, 1, figsize=(5.4, 3.4))
ax.plot(alphas, real, "o-", color=COLORS[0], label="real m1_top_k direction")
ax.plot(rand_alphas, rand, "s--", color=COLORS[3], label="random_matched direction")
if baseline is not None:
    ax.axhline(baseline, linestyle=":", linewidth=1.0, color="grey",
               label=f"treated baseline (α=0): {baseline:.3f}")
    ax.axhline(baseline - 0.02, linestyle="--", linewidth=0.8, color=COLORS[3],
               label="SP-C threshold (baseline − 2 pp)", alpha=0.6)

ax.set_xlabel("Steering coefficient α  (in units of σ_proj)")
ax.set_ylabel("MMLU-slice accuracy  (n = 500)")
ax.set_xticks(sorted(alphas))
lo = min(min(real), min(rand), baseline - 0.03 if baseline is not None else 0)
hi = max(max(real), max(rand)) + 0.02
ax.set_ylim(lo, hi)
ax.legend(frameon=False, loc="lower center", fontsize=8, ncol=2)

save_fig(fig, STEM, OUT_DIR)
plt.close(fig)
