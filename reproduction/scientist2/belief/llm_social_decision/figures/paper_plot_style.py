"""Shared publication-style plotting config for /auto Ledger Figures."""
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

COLORS = list(plt.cm.tab10.colors)
V_COLORS = {"G": COLORS[0], "A": COLORS[1], "I": COLORS[2], "M": COLORS[3]}

def save_fig(fig, path_stem, formats=("pdf", "png")):
    paths = []
    for fmt in formats:
        p = f"{path_stem}.{fmt}"
        fig.savefig(p)
        paths.append(p)
    plt.close(fig)
    return paths
