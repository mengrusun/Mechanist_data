"""Shared publication style config for C1/C2/C3 figures."""
import matplotlib
import matplotlib.pyplot as plt

FONT_SIZE = 10
DPI = 300

matplotlib.rcParams.update({
    'font.size': FONT_SIZE,
    'font.family': 'serif',
    'font.serif': ['DejaVu Serif', 'Times New Roman', 'Times'],
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
    'lines.linewidth': 1.5,
    'lines.markersize': 5.0,
})

# Colorblind-safe palette (Wong 2011 / Okabe-Ito style, tab10-adjacent)
COLORS = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion / orange
    '#009E73',  # bluish green
    '#CC79A7',  # reddish purple
    '#F0E442',  # yellow
    '#56B4E9',  # sky blue
    '#E69F00',  # orange
    '#000000',  # black
]

REF_COLOR = '#B0392C'  # subdued red for reference lines


def save_fig(fig, name, formats=('pdf', 'png'), out_dir='.'):
    """Save figure under out_dir for every format in formats."""
    paths = []
    for fmt in formats:
        path = f'{out_dir}/{name}.{fmt}'
        fig.savefig(path)
        paths.append(path)
        print(f'Saved: {path}')
    return paths
