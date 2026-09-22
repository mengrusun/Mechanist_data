"""Figure R1: search width and compute cost for Evo2 beam search with and without
SAE steering.

Layout uses three side-by-side panels for quality, capability, and compute.

a  effect of the search width W on alpha-helix rate; non-zero widths are shown
   as powers of two
b  effect of the search width W on valid-ORF rate
c  minimum generation budget, relative to one 300-nt no-search sample, whose
   configuration-level mean alpha-helix rate reaches a specified quality
   threshold (no interpolation)

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
from matplotlib.transforms import blended_transform_factory

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
BAR_STYLE = {
    "base": ("#D9D9D9", "#7B7B7B", "#6D7075"),
    "steer": ("#D9EADF", "#227652", "#2E7B45"),
}
LAB = {"base": "Evo2 + beam search (no steer)",
       "steer": "Evo2 + steer + beam search"}
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


# --- layout: three independent panels ----------------------------------------
# A4/two-column figure width: 19.05 cm = 7.5 inches.
fig = plt.figure(figsize=(19.05 / 2.54, 3.27))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.12],
                      left=0.062, right=0.995, bottom=0.19, top=0.835,
                      wspace=0.34)
ax_q = fig.add_subplot(gs[0, 0])                 # a: helix vs W
ax_y = fig.add_subplot(gs[0, 1])                 # b: valid-ORF vs W
cgs = gs[0, 2].subgridspec(2, 1, height_ratios=[0.24, 0.76], hspace=0.06)
ax_c_hi = fig.add_subplot(cgs[0, 0])             # c: broken-axis upper segment
ax_c = fig.add_subplot(cgs[1, 0], sharex=ax_c_hi) # c: main cost segment

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
ax_q.set_xlim(-0.55, len(widths) - 0.45)
ax_q.set_xticks(pos)
power_labels = ["0"] + [rf"$2^{{{int(math.log2(w))}}}$" for w in widths[1:]]
ax_q.set_xticklabels(power_labels)
ax_q.set_xlabel("search width W", labelpad=3)

# --- b: capability versus search width ---------------------------------------
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
ax_y.set_xticklabels(power_labels)
ax_y.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
ax_y.set_ylabel("Valid-ORF", labelpad=3)
ax_y.set_xlabel("search width W", labelpad=3)

# --- c: minimum observed compute needed to clear each mean-quality threshold --
thresholds = [0.60, 0.70, 0.75, 0.80]
tx = np.arange(len(thresholds), dtype=float)
# panel_b uses width=0.24; retain it and leave a 0.11 gap so the two-line
# value labels above paired bars remain visually distinct.
bar_w, bar_sep = 0.24, 0.35
for arm, off in (("base", -bar_sep / 2), ("steer", bar_sep / 2)):
    candidates = arm_rows(arm)
    fill, edge, label_color = BAR_STYLE[arm]
    for i, tau in enumerate(thresholds):
        feasible = [z for z in candidates if z["helix_hgi"] >= tau]
        if feasible:
            z = min(feasible, key=lambda q: q["nt_per_delivered"])
            cost = z["nt_per_delivered"] / 300.0
            for axis in (ax_c, ax_c_hi):
                axis.bar(tx[i] + off, cost, width=bar_w, color=fill,
                         edgecolor=edge, linewidth=0.9, zorder=3)
            label_axis = ax_c_hi if cost > 135 else ax_c
            label_axis.annotate(f"{cost:.1f}×\nW={z['width']}",
                          (tx[i] + off, cost), textcoords="offset points",
                          xytext=(0, 3), ha="center", va="bottom",
                          fontsize=6.5, color=label_color,
                          fontweight="bold", linespacing=1.05, zorder=4)
        else:
            ax_c.annotate("N.R.", (tx[i] + off, 1024.0), ha="center", va="center",
                          fontsize=6.34, color=label_color, fontweight="bold")
            ax_c.plot(tx[i] + off, 768.0, marker="x", ms=4.0, mew=0.8,
                      color=edge, zorder=4)
for axis, ticks in ((ax_c, [0, 30, 60, 90, 120]),
                    (ax_c_hi, [850, 950])):
    for y in ticks:
        axis.axhline(y, color="#E5E8E8", lw=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.set_yticks(ticks)
    axis.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
ax_c.set_ylim(0, 135)
ax_c_hi.set_ylim(850, 1000)
ax_c_hi.spines["bottom"].set_visible(False)
ax_c.spines["top"].set_visible(False)
ax_c_hi.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
# Paired diagonal marks on the y-axis indicate the omitted 135×–850× interval.
d = 0.012
for axis, y in ((ax_c_hi, 0), (ax_c, 1)):
    axis.plot((-d, +d), (y - d, y + d), transform=axis.transAxes,
              color="#222222", lw=0.8, clip_on=False)
# The only bar crossing the omitted interval receives its own zig-zag cut,
# avoiding the misleading appearance of one uninterrupted linear bar.
broken_x = tx[-1] - bar_sep / 2
for axis, y, sign in ((ax_c, 0.992, 1), (ax_c_hi, 0.008, -1)):
    trans = blended_transform_factory(axis.transData, axis.transAxes)
    xs = [broken_x - bar_w / 2, broken_x, broken_x + bar_w / 2]
    ys = [y - sign * 0.014, y + sign * 0.014, y - sign * 0.014]
    axis.plot(xs, ys, transform=trans, color="white", lw=3.2,
              solid_capstyle="butt", clip_on=False, zorder=5)
    axis.plot(xs, ys, transform=trans, color=BAR_STYLE["base"][1], lw=0.8,
              solid_capstyle="butt", clip_on=False, zorder=6)
ax_c.set_xticks(tx)
ax_c.set_xticklabels([rf"$\geq${int(t * 100)}%" for t in thresholds])
ax_c.set_ylabel("Generation budget", labelpad=3)
ax_c.set_xlabel(f"Mean {ALPHA}-helix quality threshold", labelpad=3)
for axis in (ax_c, ax_c_hi):
    for side in ("left", "bottom"):
        axis.spines[side].set_color("#222222")
        axis.spines[side].set_linewidth(0.8)
    axis.tick_params(colors="#222222", labelcolor="#222222", width=0.8, length=2.6)
ax_c.tick_params(axis="x", length=0)

# --- legend (Fig. 5b style: inside the panel, frameless) + notes -------------
from matplotlib.patches import Patch
handles = [Patch(facecolor=BAR_STYLE[a][0], edgecolor=BAR_STYLE[a][1], lw=0.9,
                 label=LAB[a]) for a in ("base", "steer")]
labels = [LAB[a] for a in ("base", "steer")]
fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.54, 1.0),
           handlelength=1.5, handleheight=1.0, handletextpad=0.5,
           labelspacing=0.32, columnspacing=2.4, borderaxespad=0.0)
ax_y.text(0.97, 0.06, "error bars: Wilson 95% CI (n = 100)",
          transform=ax_y.transAxes, fontsize=6.34, color=GREY_TEXT,
          ha="right", va="bottom")

# --- panel labels -------------------------------------------------------------
for ax, letter in ((ax_q, "a"), (ax_y, "b"), (ax_c_hi, "c")):
    ax.annotate(letter, xy=(0, 1), xycoords="axes fraction",
                xytext=(-30, 4), textcoords="offset points",
                fontsize=9.51, fontweight="bold", va="bottom", ha="left")

fig.canvas.draw()

# --- alignment gate + export -------------------------------------------------
stem = os.path.join(FIG, "R1_compute_scaling")
try:
    from audit_panel_alignment import require_matplotlib_panel_alignment
except ModuleNotFoundError:
    # The alignment helper is an optional manuscript-build utility and is not
    # included in every checkout.  The fixed gridspec above remains unchanged.
    require_matplotlib_panel_alignment = None
if require_matplotlib_panel_alignment is not None:
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
