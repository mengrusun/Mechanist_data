"""Shared publication-style plot config for /auto ledger figures.

Every per-figure generator script imports this once via a small sys.path shim.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT_SIZE = 10
DPI = 300

matplotlib.rcParams.update({
    "font.size": FONT_SIZE,
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "axes.labelsize": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 1,
    "xtick.labelsize": FONT_SIZE - 1,
    "ytick.labelsize": FONT_SIZE - 1,
    "legend.fontsize": FONT_SIZE - 1,
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "text.usetex": False,
    "mathtext.fontset": "stix",
})

COLORS = plt.cm.tab10.colors


def save_fig(fig, name, out_dir, formats=("pdf", "png")):
    paths = []
    for fmt in formats:
        path = f"{out_dir}/{name}.{fmt}"
        fig.savefig(path)
        paths.append(path)
    return paths
