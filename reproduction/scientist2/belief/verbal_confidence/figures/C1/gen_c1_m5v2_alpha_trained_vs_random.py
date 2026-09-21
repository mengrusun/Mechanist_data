"""C1 Figure 2: M5-v2 steering at E4L10 — mean verbal confidence vs α for the
trained (diff-of-means) direction and n=8 random-direction controls, averaged
across seeds {42, 123, 2024}. Both curves are flat while the capability
NLL delta stays < 0.02 nats/token: intervention IS applied but yields no
downstream effect on greedy confidence decode. Grounds P4 Steering FAIL /
argmax null.
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from paper_plot_style import plt, save_fig, COLORS  # noqa: E402

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
STEM = "c1_m5v2_alpha_trained_vs_random"

SEEDS = [42, 123, 2024]

alphas = None
trained_conf = []      # per-seed [len(alphas)]
random_conf_mean = []  # per-seed [len(alphas)]
random_conf_std = []
cap_nll_delta = []

for s in SEEDS:
    path = f"results/m5_v2/m5v2_diff_of_means_seed{s}.json"
    if not os.path.exists(path):
        continue
    with open(path) as f:
        d = json.load(f)["summary"]
    alphas = d["alphas"]
    trained = d["trained_per_alpha"]
    rand = d["random_per_alpha_agg"]

    baseline_nll = trained["0.0"]["cont_nll_mean"]
    trained_conf.append([trained[str(a)]["conf_mean"] for a in alphas])
    random_conf_mean.append(
        [rand[str(a)]["conf_mean_random_mean"] for a in alphas]
    )
    random_conf_std.append(
        [rand[str(a)]["conf_mean_random_std"] for a in alphas]
    )
    cap_nll_delta.append(
        [trained[str(a)]["cont_nll_mean"] - baseline_nll for a in alphas]
    )

trained_conf = np.array(trained_conf)
random_conf_mean = np.array(random_conf_mean)
random_conf_std = np.array(random_conf_std)
cap_nll_delta = np.array(cap_nll_delta)

fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(7.5, 2.9), sharex=True)

# Left panel: mean verbal confidence vs α — trained vs random controls.
tr_mean = trained_conf.mean(axis=0)
tr_lo = trained_conf.min(axis=0)
tr_hi = trained_conf.max(axis=0)
rn_mean = random_conf_mean.mean(axis=0)
rn_lo = (random_conf_mean - random_conf_std).mean(axis=0)
rn_hi = (random_conf_mean + random_conf_std).mean(axis=0)

ax_l.plot(alphas, tr_mean, marker="o", color=COLORS[3],
          label="trained direction (diff-of-means)")
ax_l.fill_between(alphas, tr_lo, tr_hi, color=COLORS[3], alpha=0.18)

ax_l.plot(alphas, rn_mean, marker="s", color=COLORS[0],
          label="random-direction controls (n=8)")
ax_l.fill_between(alphas, rn_lo, rn_hi, color=COLORS[0], alpha=0.18)

ax_l.axvline(0, color="grey", linewidth=0.8, linestyle=":")
ax_l.set_xlabel(r"steering coefficient $\alpha$ (scaled by $\sigma_{\mathrm{proj}}$)")
ax_l.set_ylabel("mean verbal confidence")
ax_l.legend(frameon=False, loc="lower right")

# Right panel: capability NLL delta vs α (trained direction only).
cap_mean = cap_nll_delta.mean(axis=0)
cap_lo = cap_nll_delta.min(axis=0)
cap_hi = cap_nll_delta.max(axis=0)
ax_r.plot(alphas, cap_mean, marker="d", color=COLORS[2],
          label="trained (capability probe)")
ax_r.fill_between(alphas, cap_lo, cap_hi, color=COLORS[2], alpha=0.18)
ax_r.axhline(0.3, color="red", linewidth=0.8, linestyle="--",
             label="capability tol = 0.3 nats")
ax_r.axvline(0, color="grey", linewidth=0.8, linestyle=":")
ax_r.set_xlabel(r"steering coefficient $\alpha$")
ax_r.set_ylabel(r"capability $\Delta$NLL (nats/token)")
ax_r.legend(frameon=False, loc="upper center")

save_fig(fig, STEM, OUT_DIR)
