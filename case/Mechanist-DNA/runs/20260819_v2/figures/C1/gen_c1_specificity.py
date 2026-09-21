#!/usr/bin/env python3
"""C1 specificity: real block-26 direction vs norm-matched random control, %H across alpha."""
import os, sys, json, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import plt, COLORS, save_fig

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FID = 'c1_specificity'
SRC = 'results/M3_specificity.json'
status = {'id': FID, 'type': 'line', 'source_data': SRC}


def get(m, a):
    for k in (str(a), format(a, 'g'), f'{a:.1f}'):
        if k in m:
            return m[k]
    return None


try:
    with open(os.path.join(ROOT, SRC)) as f:
        m3 = json.load(f)
    rand = m3['a_random_control']['mean_pctH_by_alpha']
    rand_alphas = sorted(float(k) for k in rand.keys())
    # real block-26 direction per-alpha comes from the M2 dose-response (same direction);
    # subset to the control's alpha grid for a matched comparison.
    with open(os.path.join(ROOT, 'results/M2_dose_response.json')) as f:
        m2 = json.load(f)
    real_map = m2['mean_pctH_by_alpha']
    real_y = [get(real_map, a) for a in rand_alphas]
    rand_y = [get(rand, a) for a in rand_alphas]
    if any(v is None for v in real_y):  # real not available on control grid
        raise KeyError('real-direction %H missing on control alpha grid')

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
    ax.plot(rand_alphas, real_y, marker='o', color=COLORS[0], lw=1.8, ms=5,
            label='real block-26 direction')
    ax.plot(rand_alphas, rand_y, marker='s', color=COLORS[1], lw=1.8, ms=5,
            ls='--', label='norm-matched random control')
    ax.set_xlabel(r'Steering coefficient $\alpha$')
    ax.set_ylabel('Mean predicted %H')
    ax.legend(frameon=False, loc='upper left')
    save_fig(fig, FID, formats=('pdf', 'png'), out_dir=OUT_DIR)
    plt.close(fig)
    status.update(status='ok',
                  pdf=f'figures/C1/{FID}.pdf', png=f'figures/C1/{FID}.png')
except FileNotFoundError:
    status.update(status='skipped', reason=f'source_data missing: {SRC}')
except Exception as e:
    status.update(status='error', error_detail=f'{e}\n{traceback.format_exc()}')

with open(os.path.join(OUT_DIR, FID + '.status.json'), 'w') as f:
    json.dump(status, f, indent=2)
print(json.dumps({k: status.get(k) for k in ('id', 'status', 'reason', 'error_detail')}))
