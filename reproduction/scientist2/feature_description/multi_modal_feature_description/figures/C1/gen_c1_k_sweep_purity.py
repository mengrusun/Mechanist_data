"""C1/P1a — last-layer top-1 concept purity vs k (line plot).

Reads runs/M6_C1_last_layer/purity__k{1,4,16,64,256}__mean.json and plots the
top-1 purity as a function of the reference-set size k, with the random
baseline as a dashed horizontal line for reference.
"""
import json
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper_plot_style import COLORS, save_fig  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RUN_DIR = os.path.join(PROJECT_ROOT, 'runs', 'M6_C1_last_layer')

K_VALUES = [1, 4, 16, 64, 256]


def load_data():
    xs, ys, baselines = [], [], []
    for k in K_VALUES:
        p = os.path.join(RUN_DIR, f'purity__k{k}__mean.json')
        with open(p) as f:
            d = json.load(f)
        xs.append(d['k'])
        ys.append(d['top1_purity'])
        baselines.append(d.get('random_baseline_top1_purity', 0.001))
    return xs, ys, baselines


def main():
    xs, ys, baselines = load_data()

    fig, ax = plt.subplots(1, 1, figsize=(4.4, 3.0))

    # Main curve: top-1 purity vs k
    ax.plot(
        xs, ys,
        marker='o', markersize=5, linewidth=1.6,
        color=COLORS[0], label='Top-1 concept purity',
    )
    # Value annotations
    for x, y in zip(xs, ys):
        ax.annotate(
            f'{y:.3f}', xy=(x, y),
            xytext=(0, 8), textcoords='offset points',
            ha='center', fontsize=8, color=COLORS[0],
        )

    # Random baseline (mean across k) as a dashed horizontal line
    baseline_val = sum(baselines) / len(baselines)
    ax.axhline(
        baseline_val, linestyle='--', linewidth=1.0,
        color='gray', label=f'Random baseline ({baseline_val:.3f})',
    )

    ax.set_xscale('log', base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_xlabel('Reference-set size $k$ (log scale)')
    ax.set_ylabel('Top-1 concept purity')
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc='lower center', frameon=False)

    out_name = 'c1_k_sweep_purity'
    save_fig(fig, out_name, formats=('pdf', 'png'), out_dir=HERE)
    plt.close(fig)


if __name__ == '__main__':
    main()
