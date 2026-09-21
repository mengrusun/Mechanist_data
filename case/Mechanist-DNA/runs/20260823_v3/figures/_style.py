import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({
    "font.size": 10, "font.family": "serif",
    "font.serif": ["Times New Roman","Times","DejaVu Serif"],
    "axes.labelsize": 10, "axes.titlesize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "figure.dpi": 300, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05, "axes.grid": False,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.usetex": False, "mathtext.fontset": "stix",
})
C = plt.cm.tab10.colors
def save(fig, stem, out):
    ps=[]
    for fmt in ("pdf","png"):
        p=f"{out}/{stem}.{fmt}"; fig.savefig(p); ps.append(p); print("Saved",p)
    return ps
