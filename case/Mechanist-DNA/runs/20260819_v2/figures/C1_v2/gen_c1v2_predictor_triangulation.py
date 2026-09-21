#!/usr/bin/env python3
"""C1_v2 predictor triangulation: baseline vs steered predicted-%H for 3 SS predictors."""
import os, sys, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import plt, COLORS, save_fig
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FID = 'c1v2_predictor_triangulation'
SRC = 'runs/iteration_round_1/M5_structural_gc_control.json'
status = {'id': FID, 'type': 'grouped_bar', 'source_data': SRC}

LABELS = {'esm2': 'ESM2-probe', 'gor': 'GOR', 'cf': 'Chou-Fasman'}
ORDER = ['esm2', 'gor', 'cf']

try:
    with open(os.path.join(ROOT, SRC)) as f:
        d = json.load(f)
    up = d['full_set_uplift_by_predictor']
    names = [LABELS[k] for k in ORDER]
    base = [up[k]['mean_baseline'] for k in ORDER]
    lib = [up[k]['mean_library'] for k in ORDER]
    delta = [up[k]['delta'] for k in ORDER]
    # asymmetric CI on the library (steered) bar, expressed as delta CI around library
    ci = [up[k].get('ci95') for k in ORDER]

    x = np.arange(len(ORDER))
    w = 0.38
    fig, ax = plt.subplots(1, 1, figsize=(5.2, 3.6))
    b1 = ax.bar(x - w / 2, base, w, color=COLORS[7], label='baseline')
    b2 = ax.bar(x + w / 2, lib, w, color=COLORS[0], label='steered (library)')
    for xi, bl, lb, dl in zip(x, base, lib, delta):
        ax.text(xi + w / 2, lb + 1.2, f'+{dl:.1f}', ha='center', va='bottom',
                fontsize=8, color=COLORS[0])
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel('Mean predicted %H')
    ax.set_ylim(0, max(lib) * 1.2)
    ax.legend(frameon=False, loc='upper left')
    save_fig(fig, FID, formats=('pdf', 'png'), out_dir=OUT_DIR)
    plt.close(fig)
    status.update(status='ok',
                  pdf=f'figures/C1_v2/{FID}.pdf', png=f'figures/C1_v2/{FID}.png')
except FileNotFoundError:
    status.update(status='skipped', reason=f'source_data missing: {SRC}')
except Exception as e:
    status.update(status='error', error_detail=f'{e}\n{traceback.format_exc()}')

with open(os.path.join(OUT_DIR, FID + '.status.json'), 'w') as f:
    json.dump(status, f, indent=2)
print(json.dumps({k: status.get(k) for k in ('id', 'status', 'reason', 'error_detail')}))
