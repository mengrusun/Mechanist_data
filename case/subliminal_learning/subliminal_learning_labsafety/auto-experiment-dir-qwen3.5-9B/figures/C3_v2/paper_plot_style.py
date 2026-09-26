"""Shared publication-style config for C1 figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FONT_SIZE = 10
DPI = 300
COLORS = plt.cm.tab10.colors

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


def save_fig(fig, stem, out_dir, formats=("pdf", "png")):
    """Save `fig` under `out_dir` as `stem.<fmt>` for every fmt in `formats`."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        path = f"{out_dir}/{stem}.{fmt}"
        fig.savefig(path)
        paths.append(path)
        print(f"Saved: {path}")
    return paths
