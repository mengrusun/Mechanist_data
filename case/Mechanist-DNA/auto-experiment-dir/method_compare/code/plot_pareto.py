"""Figure R1: search width and compute cost for Evo2 beam search with and without
SAE steering.

Layout follows the manuscript's Fig. 5c idiom: one knob axis carrying two stacked
readouts (quality on top, capability below), plus a single compute axis.

a  effect of the search width W on alpha-helix rate (top) and on valid-ORF rate
   (bottom); both stacked panels share the same x axis
b  compute-quality frontier: alpha-helix rate versus measured GPU time per
   *valid* sequence (single A800-80GB), so the yield shown in a (bottom) is
   folded into the cost axis

Drawing specification measured from Fig. 5 of the manuscript: serif type at
8.9/8.2/7.0/6.3 pt, left+bottom spines in #3A3A3A at 0.80 pt, horizontal #E6E8E9
gridlines at 0.71 pt, the grey/green method pair of Fig. 5b, circular markers for
alpha-helix and square markers for valid-ORF.
"""
import os, sys, json, math
sys.path.insert(0, os.path.dirname(__file__))
import mc_env
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# --- Fig. 5 drawing specification --------------------------------------------
SPINE, GRID, GREY_TEXT = "#3A3A3A", "#E6E8E9", "#595959"
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Cambria", "Caladea", "DejaVu Serif", "serif"],
    "mathtext.fontset": "dejavuserif",
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "font.size": 8.24,
    "axes.labelsize": 8.88,
    "axes.titlesize": 8.88,
    "xtick.labelsize": 8.24,
    "ytick.labelsize": 8.24,
    "legend.fontsize": 6.98,
    "axes.linewidth": 0.80,
    "axes.edgecolor": SPINE,
    "axes.labelcolor": "black",
    "xtick.color": SPINE, "ytick.color": SPINE,
    "xtick.labelcolor": "black", "ytick.labelcolor": "black",
    "xtick.major.width": 0.80, "ytick.major.width": 0.80,
    "xtick.minor.width": 0.55,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6, "xtick.minor.size": 1.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False,
    "lines.solid_capstyle": "round",
})

S = json.load(open(os.path.join(mc_env.MC, "results", "summary.json")))
rows = S["rows"]
FIG = os.path.join(mc_env.MC, "figures")
os.makedirs(FIG, exist_ok=True)

# Fig. 5b pairs: Evo2-7B (no steer) = grey, alpha-helix-feature steering = green
COL = {"base": "#7B7B7B", "steer": "#4C9866"}
LAB = {"base": "Evo2 + beam search (no steering)",
       "steer": "Evo2 + steering + beam search"}
LW, MS, MEW = 1.15, 3.2, 0.45
EB = dict(elinewidth=0.7, capsize=1.7, capthick=0.7,
          markeredgecolor="white", markeredgewidth=MEW)
ALPHA = "$\\alpha$"


def arm_rows(arm, key="width"):
    return sorted([r for r in rows if r["arm"] == arm], key=lambda z: z[key])


def wilson(k, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def style_axes(ax, yticks):
    """Fig. 5 axis furniture: horizontal gridlines only, behind the data."""
    ax.set_yticks(yticks)
    for y in yticks:
        ax.axhline(y, color=GRID, lw=0.71, zorder=0)
    ax.set_axisbelow(True)


# --- layout: stacked knob panel (a) beside the compute panel (b) -------------
fig = plt.figure(figsize=(7.09, 3.27))
gs = fig.add_gridspec(2, 2, height_ratios=[2.35, 1.0], width_ratios=[1.0, 1.06],
                      left=0.082, right=0.995, bottom=0.166, top=0.845,
                      wspace=0.36, hspace=0.16)
ax_q = fig.add_subplot(gs[0, 0])                 # a, top:    helix vs W
ax_y = fig.add_subplot(gs[1, 0], sharex=ax_q)    # a, bottom: valid-ORF vs W
ax_c = fig.add_subplot(gs[:, 1])                 # b: helix vs GPU-s per valid seq

YT_Q = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
YT_Y = [0.8, 0.9, 1.0]
widths = [z["width"] for z in arm_rows("base")]
pos = np.arange(len(widths), dtype=float)

# --- a (top): quality versus the search width --------------------------------
style_axes(ax_q, YT_Q)
for arm, off in (("base", -0.075), ("steer", 0.075)):
    r = arm_rows(arm)
    y = np.array([z["helix_hgi"] for z in r])
    e = np.array([z["helix_hgi_sem"] for z in r])
    ax_q.errorbar(pos + off, y, yerr=e, marker="o", ms=MS, lw=LW, color=COL[arm],
                  zorder=3 if arm == "steer" else 2, label=LAB[arm], **EB)
ax_q.set_ylim(0.335, 0.92)
ax_q.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
ax_q.set_ylabel(f"Rate of {ALPHA}-helix (%)", labelpad=3)
plt.setp(ax_q.get_xticklabels(), visible=False)
ax_q.tick_params(axis="x", length=0)

# --- a (bottom): capability on the same search-width axis --------------------
style_axes(ax_y, YT_Y)
for arm, off in (("base", -0.075), ("steer", 0.075)):
    r = arm_rows(arm)
    p = np.array([z["valid_orf_rate"] for z in r])
    ci = np.array([wilson(z["n_folded"], z["n_delivered"]) for z in r])
    lo = np.clip(p - ci[:, 0], 0, None)
    hi = np.clip(ci[:, 1] - p, 0, None)
    ax_y.errorbar(pos + off, p, yerr=[lo, hi], marker="s", ms=MS, lw=LW,
                  color=COL[arm], zorder=3 if arm == "steer" else 2, **EB)
ax_y.set_ylim(0.70, 1.07)
ax_y.set_xlim(-0.55, len(widths) - 0.45)
ax_y.set_xticks(pos)
ax_y.set_xticklabels([str(w) for w in widths])
ax_y.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
ax_y.set_ylabel("Valid-ORF", labelpad=3)
ax_y.set_xlabel("search width W", labelpad=3)

# --- b: compute-quality frontier, cost per valid sequence --------------------
style_axes(ax_c, YT_Q)
numerals = []
for arm in ("base", "steer"):
    r = arm_rows(arm, "gpu_s_per_valid")
    x = np.array([z["gpu_s_per_valid"] for z in r])
    y = np.array([z["helix_hgi"] for z in r])
    e = np.array([z["helix_hgi_sem"] for z in r])
    ax_c.errorbar(x, y, yerr=e, marker="o", ms=MS, lw=LW, color=COL[arm],
                  zorder=3 if arm == "steer" else 2, **EB)
    # one uniform offset per arm so the labels read as two aligned rows
    for z in r:
        # steering labels ride above its curve; baseline labels below, except the
        # cramped W=0 and W=1 pair at the bottom-left, which also go above
        up = arm == "steer" or z["width"] in (0, 1)
        t = ax_c.annotate(f"W={z['width']}", (z["gpu_s_per_valid"], z["helix_hgi"]),
                          textcoords="offset points",
                          xytext=(0, 10.0 if up else -10.0),
                          fontsize=6.34, color=COL[arm], zorder=4, ha="center",
                          va="bottom" if up else "top")
        numerals.append((t, "steer" if up else "base"))
ax_c.set_xscale("log")
ax_c.set_xlim(min(z["gpu_s_per_valid"] for z in rows) / 4.0,
              max(z["gpu_s_per_valid"] for z in rows) * 3.0)
ax_c.set_ylim(0.335, 0.92)
ax_c.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v * 100:.0f}"))
ax_c.set_ylabel(f"Rate of {ALPHA}-helix (%)", labelpad=3)
ax_c.set_xlabel("GPU time to generate one valid sequence (seconds)", labelpad=3)

# --- legend (Fig. 5b style: inside the panel, frameless) + notes -------------
handles, labels = ax_q.get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.54, 1.0),
           handlelength=1.5, handletextpad=0.5, columnspacing=2.4, borderaxespad=0.0)
ax_c.text(0.975, 0.055,
          "W: search width\n"
          "error bars: mean $\\pm$ s.e.m. (n = 82\u2013100)\n"
          "timing: 1\u00d7 NVIDIA A800-80GB",
          transform=ax_c.transAxes, fontsize=6.34,
          color=GREY_TEXT, ha="right", va="bottom", linespacing=1.4)
ax_y.text(0.97, 0.06, "error bars: Wilson 95% CI (n = 100)",
          transform=ax_y.transAxes, fontsize=6.34, color=GREY_TEXT,
          ha="right", va="bottom")

# --- panel labels (the stacked pair is one panel, as in Fig. 5c) -------------
for ax, letter in ((ax_q, "a"), (ax_c, "b")):
    # anchor both letters the same distance above the shared grid boundary
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(-30, 4), textcoords="offset points",
                fontsize=9.51, fontweight="bold", va="bottom", ha="left")

# --- nudge only the labels a gridline would cross; the rest stay aligned ----
fig.canvas.draw()
rend = fig.canvas.get_renderer()
CANDIDATES = {"steer": [10.0, 12.5, 7.5, 14.5, 16.5, 18.5],
              "base": [-10.0, -12.5, -7.5, -14.5, -16.5, -18.5]}
grid_y = [ax_c.transData.transform((ax_c.get_xlim()[0], g))[1] for g in YT_Q]
for t, arm in numerals:
    best = None
    for dy in CANDIDATES[arm]:
        t.xyann = (0, dy)
        fig.canvas.draw()
        bb = t.get_window_extent(renderer=rend)
        hits = sum(bb.y0 - 0.8 <= g <= bb.y1 + 0.8 for g in grid_y)
        if best is None or hits < best[0]:
            best = (hits, dy)
        if hits == 0:
            break
    t.xyann = (0, best[1])
fig.canvas.draw()

# --- alignment gate + export -------------------------------------------------
from audit_panel_alignment import require_matplotlib_panel_alignment

stem = os.path.join(FIG, "R1_compute_scaling")
require_matplotlib_panel_alignment(
    fig,
    json_out=f"{stem}.alignment.json",
    tolerance_pt=1.5,
    gutter_tolerance_pt=1.5,
    strict=True,
)
for ext, kw in ((".pdf", {}), (".svg", {}), (".png", {"dpi": 600})):
    fig.savefig(stem + ext, **kw)
print("WROTE", stem + ".{pdf,svg,png}")
