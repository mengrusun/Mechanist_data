"""Shared publication-style matplotlib configuration for paper figures.

Copy this file into each figures/<claim_id>/ dir OR keep it in the parent
figures/ dir and import via sys.path manipulation from the per-figure scripts.
"""

import matplotlib
import matplotlib.pyplot as plt

FONT_SIZE = 10
DPI = 300

matplotlib.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'axes.labelsize': FONT_SIZE,
    'axes.titlesize': FONT_SIZE + 1,
    'xtick.labelsize': FONT_SIZE - 1,
    'ytick.labelsize': FONT_SIZE - 1,
    'legend.fontsize': FONT_SIZE - 1,
    'figure.dpi': DPI,
    'savefig.dpi': DPI,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.grid': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
    'mathtext.fontset': 'stix',
})

COLORS = plt.cm.tab10.colors


def save_fig(fig, name, formats=('pdf', 'png'), out_dir='.'):
    """Save `fig` to out_dir/name.<fmt> for every fmt in `formats`. Return list of paths."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for fmt in formats:
        path = os.path.join(out_dir, f'{name}.{fmt}')
        fig.savefig(path)
        paths.append(path)
    return paths
