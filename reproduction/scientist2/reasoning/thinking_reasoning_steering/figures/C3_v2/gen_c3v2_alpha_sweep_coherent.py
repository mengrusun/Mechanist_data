"""C3_v2 figure: Expanded alpha-grid dose-response for uncertainty.

Line plot of behaviour_rate vs alpha (in units of sigma) on the 9-point
+-3 sigma grid, with the coherence rate overlaid on a secondary axis so the
+-3 sigma collapse is visible.

Data source: runs/iteration_round_1/M3_C3_expand/results_summary.json
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import matplotlib.pyplot as plt  # noqa: E402
from paper_plot_style import COLORS, save_fig  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    'runs',
    'iteration_round_1',
    'M3_C3_expand',
    'results_summary.json',
)
OUT_DIR = HERE
NAME = 'c3v2_alpha_sweep_coherent'

# Coherence must be >= this to count a point as "coherent" for the fit.
COHERENCE_THRESHOLD = 0.9


def main() -> None:
    with open(DATA_PATH) as f:
        data = json.load(f)

    rate_by_alpha = data['analysis']['rate_by_alpha']
    coh_by_alpha = data['analysis']['coherence_by_alpha']
    baseline_rate = data['analysis']['baseline_rate']

    alphas = sorted(float(k) for k in rate_by_alpha.keys())
    rates = [rate_by_alpha[f'{a:.2f}'] for a in alphas]
    cohs = [coh_by_alpha[f'{a:.2f}'] for a in alphas]

    is_coherent = [c >= COHERENCE_THRESHOLD for c in cohs]

    fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.2))

    # Behaviour-rate curve.
    ax.plot(
        alphas,
        rates,
        color=COLORS[0],
        linewidth=1.4,
        marker='o',
        markersize=4,
        label='Behaviour rate',
        zorder=3,
    )

    # Mark low-coherence points with hollow markers.
    for a, r, coh_ok in zip(alphas, rates, is_coherent):
        if not coh_ok:
            ax.plot(
                a,
                r,
                marker='o',
                markersize=8,
                markerfacecolor='none',
                markeredgecolor='red',
                markeredgewidth=1.2,
                linestyle='none',
                zorder=4,
            )

    # Baseline reference line.
    ax.axhline(
        baseline_rate,
        color='grey',
        linewidth=0.7,
        linestyle='--',
        label=f'Baseline ({baseline_rate:.2f})',
    )

    ax.set_xlabel(r'Steering coefficient $\alpha$ (units of $\sigma_{\mathrm{proj}}$)')
    ax.set_ylabel('P(expressing_uncertainty)', color=COLORS[0])
    ax.tick_params(axis='y', labelcolor=COLORS[0])
    ax.set_ylim(-0.05, 0.50)
    ax.set_xlim(-3.6, 3.6)

    # Secondary axis: coherence rate.
    ax2 = ax.twinx()
    ax2.spines['top'].set_visible(False)
    ax2.plot(
        alphas,
        cohs,
        color=COLORS[3],
        linewidth=1.0,
        marker='s',
        markersize=3.4,
        linestyle=':',
        label='Coherence',
    )
    ax2.set_ylabel('Coherence rate', color=COLORS[3])
    ax2.tick_params(axis='y', labelcolor=COLORS[3])
    ax2.set_ylim(0, 1.05)
    ax2.axhline(COHERENCE_THRESHOLD, color=COLORS[3], linewidth=0.5, linestyle=':', alpha=0.5)

    # Combined legend.
    lines_1, labels_1 = ax.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        frameon=False,
        loc='upper center',
        ncol=3,
        fontsize=7.5,
        bbox_to_anchor=(0.5, 1.14),
        handlelength=1.5,
        columnspacing=1.0,
    )

    # Annotate Spearman rho on the coherent subset.
    rho = data['analysis']['spearman_rho_coherent_subset']
    p = data['analysis']['spearman_p']
    ax.text(
        0.5,
        -0.28,
        f'Spearman ρ = {rho:+.3f} (p = {p:.2f}) on coherent subset '
        f'(|α| ≤ 1σ). Hollow red circles: low-coherence points (< {COHERENCE_THRESHOLD:.1f}).',
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=7,
        style='italic',
    )

    fig.tight_layout()
    save_fig(fig, NAME, formats=('pdf', 'png'), out_dir=OUT_DIR)
    plt.close(fig)


if __name__ == '__main__':
    main()
