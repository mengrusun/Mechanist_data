"""C1/P1b — hidden-layer matched-control cosine gap Delta_sep at k=16 (bar chart).

Reads runs/M7_C1_hidden/sep__k16__mean.json and plots the per-layer
Delta_sep = cos(v_c, t_top1) - cos(v_c, t_top2) for layer3 and layer4, with
95% CI error bars.
"""
import json
import os
import sys

import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from paper_plot_style import COLORS, save_fig  # noqa: E402

PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
RUN_FILE = os.path.join(PROJECT_ROOT, 'runs', 'M7_C1_hidden', 'sep__k16__mean.json')


def load_data():
    with open(RUN_FILE) as f:
        d = json.load(f)
    per_layer = d['per_layer']
    order = ['layer3', 'layer4']
    means = [per_layer[l]['delta_sep_mean'] for l in order]
    ci = [per_layer[l]['delta_sep_ci_95'] for l in order]
    # Convert (lo, hi) around mean to symmetric-ish err bars (lo_err, hi_err)
    lo_err = [max(0.0, m - c[0]) for m, c in zip(means, ci)]
    hi_err = [max(0.0, c[1] - m) for m, c in zip(means, ci)]
    pvals = [per_layer[l]['p_value_paired_greater'] for l in order]
    return order, means, [lo_err, hi_err], pvals


def main():
    layers, means, errs, pvals = load_data()

    fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0))

    x = list(range(len(layers)))
    bars = ax.bar(
        x, means,
        yerr=errs, capsize=4,
        color=[COLORS[1], COLORS[0]],
        edgecolor='black', linewidth=0.6,
        width=0.55,
    )

    # Value labels above each bar
    for bar, m, p in zip(bars, means, pvals):
        if p == 0.0:
            p_txt = 'p<1e-300'
        else:
            p_txt = f'p={p:.1e}'
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(errs[1]) * 0.35 + 0.001,
            f'{m:.4f}\n({p_txt})',
            ha='center', va='bottom', fontsize=8,
        )

    ax.axhline(0.0, color='gray', linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(layers)
    ax.set_xlabel('ResNet-50 layer')
    ax.set_ylabel(r'Matched-control gap $\Delta_{\mathrm{sep}}$')

    # Give headroom for the annotations
    y_top = max(m + e for m, e in zip(means, errs[1])) * 1.6
    ax.set_ylim(0.0, y_top)

    out_name = 'c1_hidden_matched_control'
    save_fig(fig, out_name, formats=('pdf', 'png'), out_dir=HERE)
    plt.close(fig)


if __name__ == '__main__':
    main()
