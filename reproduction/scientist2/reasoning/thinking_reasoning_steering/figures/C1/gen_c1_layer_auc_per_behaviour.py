"""C1 figure: Per-behaviour held-out linear-probe ROC-AUC vs layer.

Data source: runs/M1_locate/results.json
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib.pyplot as plt  # noqa: E402
from paper_plot_style import COLORS, save_fig  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA_PATH = os.path.join(PROJECT_ROOT, 'runs', 'M1_locate', 'results.json')
OUT_DIR = HERE
NAME = 'c1_layer_auc_per_behaviour'


def pretty(behaviour: str) -> str:
    return {
        'expressing_uncertainty': 'Uncertainty',
        'generating_validation_examples': 'Validation',
        'backtracking': 'Backtracking',
        'self-correction': 'Self-correction',
    }.get(behaviour, behaviour)


def main() -> None:
    with open(DATA_PATH) as f:
        data = json.load(f)

    per_behaviour = data['per_behaviour']

    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.3))

    # Fixed color order for reproducibility.
    order = [
        'expressing_uncertainty',
        'generating_validation_examples',
        'backtracking',
        'self-correction',
    ]

    for i, behaviour in enumerate(order):
        entry = per_behaviour[behaviour]
        curve = entry['layer_auc_curve']
        layers = sorted(int(k) for k in curve.keys())
        aucs = [curve[str(l)] for l in layers]
        color = COLORS[i]

        ax.plot(
            layers,
            aucs,
            label=pretty(behaviour),
            color=color,
            linewidth=1.4,
            marker='o',
            markersize=2.6,
        )

        # Mark L* with a filled star of the same colour.
        l_star = entry['L_star']
        auc_star = curve[str(l_star)]
        ax.plot(
            l_star,
            auc_star,
            marker='*',
            markersize=11,
            color=color,
            markeredgecolor='black',
            markeredgewidth=0.6,
            linestyle='none',
            zorder=5,
        )

    # AUC = 0.75 admission threshold.
    ax.axhline(0.75, linestyle='--', color='grey', linewidth=0.8, alpha=0.6)
    ax.text(
        31.4,
        0.755,
        'AUC = 0.75',
        color='grey',
        fontsize=7,
        ha='right',
        va='bottom',
    )

    ax.set_xlabel('Layer index')
    ax.set_ylabel('Held-out ROC-AUC')
    ax.set_ylim(0.55, 1.03)
    ax.set_xlim(-0.5, 31.5)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
    ax.legend(
        frameon=False,
        loc='lower left',
        ncol=2,
        columnspacing=0.9,
        handlelength=1.5,
        borderaxespad=0.2,
    )

    fig.tight_layout()
    save_fig(fig, NAME, formats=('pdf', 'png'), out_dir=OUT_DIR)
    plt.close(fig)


if __name__ == '__main__':
    main()
