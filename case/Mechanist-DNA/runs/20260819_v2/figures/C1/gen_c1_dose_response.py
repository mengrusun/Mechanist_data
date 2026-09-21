#!/usr/bin/env python3
"""C1 dose-response: mean predicted %H vs steering coefficient alpha (block-26)."""
import os, sys, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import plt, COLORS, save_fig

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FID = 'c1_dose_response'
SRC = 'results/M2_dose_response.json'
status = {'id': FID, 'type': 'line', 'source_data': SRC}

try:
    with open(os.path.join(ROOT, SRC)) as f:
        d = json.load(f)
    alphas = d['alphas']
    mean_by = d['mean_pctH_by_alpha']
    baseline = d['baseline_mean_pctH']
    y = [mean_by[str(a)] if str(a) in mean_by else mean_by[format(a, 'g')] for a in alphas]

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    ax.plot(alphas, y, marker='o', color=COLORS[0], lw=1.8, ms=5,
            label='block-26 direction')
    ax.axhline(baseline, ls='--', color='0.4', lw=1.2,
               label=f'baseline (%H={baseline:.1f})')
    # mark the alpha=4 dip
    if 4.0 in alphas:
        yd = y[alphas.index(4.0)]
        ax.annotate(r'$\alpha$=4 dip', xy=(4.0, yd), xytext=(4.5, yd - 6),
                    fontsize=8, color=COLORS[3],
                    arrowprops=dict(arrowstyle='->', color=COLORS[3], lw=1))
    ax.set_xlabel(r'Steering coefficient $\alpha$')
    ax.set_ylabel('Mean predicted %H')
    ax.legend(frameon=False, loc='upper left')
    save_fig(fig, FID, formats=('pdf', 'png'), out_dir=OUT_DIR)
    plt.close(fig)
    status.update(status='ok',
                  pdf=f'figures/C1/{FID}.pdf', png=f'figures/C1/{FID}.png')
except FileNotFoundError as e:
    status.update(status='skipped', reason=f'source_data missing: {SRC}')
except Exception as e:
    status.update(status='error', error_detail=f'{e}\n{traceback.format_exc()}')

with open(os.path.join(OUT_DIR, FID + '.status.json'), 'w') as f:
    json.dump(status, f, indent=2)
print(json.dumps({k: status.get(k) for k in ('id', 'status', 'reason', 'error_detail')}))
