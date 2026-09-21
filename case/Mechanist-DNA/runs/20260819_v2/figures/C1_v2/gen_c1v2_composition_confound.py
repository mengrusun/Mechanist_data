#!/usr/bin/env python3
"""C1_v2 composition confound: GC collapse + AA/low-complexity shifts track the %H gain."""
import os, sys, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import plt, COLORS, save_fig
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FID = 'c1v2_composition_confound'
SRC = 'runs/iteration_round_1/M5_structural_gc_control.json'
status = {'id': FID, 'type': 'multi_panel', 'source_data': SRC}

try:
    with open(os.path.join(ROOT, SRC)) as f:
        d = json.load(f)
    gc = d['gc_distribution']
    gc_base, gc_lib = gc['baseline']['mean'], gc['library']['mean']
    comp = d['composition_controls']
    lc_base = comp['baseline']['lowcplx_maxAAfrac']['mean']
    lc_lib = comp['library']['lowcplx_maxAAfrac']['mean']
    aa = d['aa_composition_shift_top']  # list of [residue, delta_frac]

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))

    # Panel A: GC content
    ax = axes[0]
    ax.bar([0, 1], [gc_base, gc_lib], color=[COLORS[7], COLORS[0]], width=0.6)
    for xi, v in zip([0, 1], [gc_base, gc_lib]):
        ax.text(xi, v + 0.01, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['baseline', 'steered'])
    ax.set_ylabel('Mean GC content')
    ax.set_ylim(0, max(gc_base, gc_lib) * 1.25)
    ax.set_title('(a) GC collapse')

    # Panel B: top AA composition shifts
    ax = axes[1]
    res = [r for r, _ in aa]
    dv = [v for _, v in aa]
    colors = [COLORS[3] if v > 0 else COLORS[9] for v in dv]
    ax.bar(range(len(res)), dv, color=colors, width=0.7)
    ax.axhline(0, color='0.5', lw=0.8)
    ax.set_xticks(range(len(res))); ax.set_xticklabels(res)
    ax.set_xlabel('Amino acid')
    ax.set_ylabel(r'$\Delta$ frac (steered - baseline)')
    ax.set_title('(b) AA composition shift')

    # Panel C: low-complexity (max single-AA fraction)
    ax = axes[2]
    ax.bar([0, 1], [lc_base, lc_lib], color=[COLORS[7], COLORS[0]], width=0.6)
    for xi, v in zip([0, 1], [lc_base, lc_lib]):
        ax.text(xi, v + 0.008, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks([0, 1]); ax.set_xticklabels(['baseline', 'steered'])
    ax.set_ylabel('Low-complexity (max AA frac)')
    ax.set_ylim(0, max(lc_base, lc_lib) * 1.3)
    ax.set_title('(c) Low-complexity rise')

    fig.tight_layout()
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
