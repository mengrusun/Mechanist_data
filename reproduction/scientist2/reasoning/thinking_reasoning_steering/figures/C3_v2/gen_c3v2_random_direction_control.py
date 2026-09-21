"""C3_v2 figure: Random-direction control refuting causal specificity.

Bars compare Δrate (behaviour rate at α=1σ minus baseline at α=0) for the
learned mean-difference direction vs a distribution of random unit directions
of matched L2 norm.

Data source: runs/iteration_round_1/M3_C3_expand/results_summary.json
"""

import json
import os
import statistics
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
NAME = 'c3v2_random_direction_control'


def main() -> None:
    with open(DATA_PATH) as f:
        data = json.load(f)

    # Learned direction: analysis.rate_by_alpha at alpha=1.00 minus baseline (alpha=0.00).
    rate_by_alpha = data['analysis']['rate_by_alpha']
    baseline_rate = data['analysis']['baseline_rate']
    learned_delta = rate_by_alpha['1.00'] - baseline_rate  # = 0.0

    # Random-direction control at same alpha=1σ.
    random_deltas = [
        entry['delta_rate_vs_baseline']
        for entry in data['random_direction_control']['random_results']
    ]
    n_random = len(random_deltas)
    random_mean = statistics.mean(random_deltas)
    random_std = statistics.stdev(random_deltas) if n_random > 1 else 0.0

    labels = ['Learned direction\n(mean-difference)', f'Random directions\n(n = {n_random})']
    means = [learned_delta, random_mean]
    errs = [0.0, random_std]
    # Use explicit distinct colours (tab10 slot 0 = blue, slot 3 = red).
    colors = [COLORS[0], COLORS[3]]

    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.2))
    bars = ax.bar(
        labels,
        means,
        yerr=errs,
        color=colors,
        capsize=4,
        edgecolor='black',
        linewidth=0.6,
        error_kw={'elinewidth': 0.8, 'ecolor': 'black'},
    )

    ax.axhline(0.0, color='grey', linewidth=0.6, linestyle='-', alpha=0.7)

    # Value labels above bars.
    for bar, m, e in zip(bars, means, errs):
        top = bar.get_height() + (e if m >= 0 else 0)
        va = 'bottom' if m >= 0 else 'top'
        offset = 0.004 if m >= 0 else -0.004
        label = f'{m:+.3f}'
        if e > 0:
            label += f' ± {e:.3f}'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            top + offset,
            label,
            ha='center',
            va=va,
            fontsize=8,
        )

    ax.set_ylabel(r'$\Delta$ behaviour rate at $\alpha = 1\sigma$')
    ymax = max(0.0, max(means[i] + errs[i] for i in range(len(means)))) + 0.04
    ymin = min(0.0, min(means[i] - errs[i] for i in range(len(means)))) - 0.02
    ax.set_ylim(ymin, ymax)

    # Annotate z-score.
    ax.text(
        0.5,
        0.92,
        'z = -6.33',
        transform=ax.transAxes,
        ha='center',
        va='top',
        fontsize=8,
        style='italic',
    )

    fig.tight_layout()
    save_fig(fig, NAME, formats=('pdf', 'png'), out_dir=OUT_DIR)
    plt.close(fig)


if __name__ == '__main__':
    main()
