# gen_c3_dose_response.py
# C3: predicted %-helix and valid-ORF rate vs amplification α (M2 dev sweep).
import os, sys, json, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paper_plot_style import *

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FORMATS = ('pdf', 'png')

ARMS = [
    (0.0, 'm2_a0_D.json'),
    (0.5, 'm2_a0.5_D.json'),
    (1.0, 'm2_a1_D.json'),
    (2.0, 'm2_a2_D.json'),
    (4.0, 'm2_a4_D.json'),
    (8.0, 'm2_a8_D.json'),
    (16.0, 'm2_a16_D.json'),
]


def load_arm(fname):
    with open(os.path.join(ROOT, 'results', fname)) as f:
        d = json.load(f)
    seqs = d['per_seq']
    n = len(seqs)
    valid = [s for s in seqs if s.get('qc_ok')]
    helix = [float(s['helix_all']) for s in valid if s.get('helix_all') is not None]
    mean = sum(helix) / len(helix)
    var = sum((x - mean) ** 2 for x in helix) / max(len(helix) - 1, 1)
    sem = math.sqrt(var / len(helix))
    return dict(
        n=n, n_valid=len(valid), valid_rate=len(valid) / n,
        helix_mean=mean, helix_sem=sem,
    )


rows = [load_arm(fn) for _, fn in ARMS]
alphas = [a for a, _ in ARMS]
means = [r['helix_mean'] for r in rows]
sems = [r['helix_sem'] for r in rows]
valid = [r['valid_rate'] for r in rows]
peak_i = max(range(len(means)), key=lambda i: means[i])

# Equal-spaced categorical x so α=0.5 is readable next to 0–16.
xs = list(range(len(alphas)))
xticklabels = ['0', '0.5', '1', '2', '4', '8', '16']

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.4, 5.2), sharex=True)

lo = [m - e for m, e in zip(means, sems)]
hi = [m + e for m, e in zip(means, sems)]
ax1.fill_between(xs, lo, hi, color=COLORS[0], alpha=0.18, linewidth=0)
ax1.plot(xs, means, '-o', color=COLORS[0], markersize=5, linewidth=1.4,
         label=r'Mean $\alpha$-helix fraction')
ax1.scatter([xs[peak_i]], [means[peak_i]], s=90, facecolors='none',
            edgecolors=COLORS[3], linewidths=1.6, zorder=5)
ax1.annotate(f'peak {means[peak_i]:.3f}\n@ $\\alpha$={alphas[peak_i]:g}',
             (xs[peak_i], means[peak_i]),
             textcoords='offset points', xytext=(18, 8),
             ha='left', fontsize=FONT_SIZE - 2, color=COLORS[3])
ax1.axhline(means[0], color='grey', linestyle='--', linewidth=0.7, zorder=0)
ax1.set_ylabel(r'Mean $\alpha$-helix fraction')
ax1.set_ylim(0.04, 0.34)

ax2.plot(xs, valid, '-s', color=COLORS[2], markersize=5, linewidth=1.4)
for x, v in zip(xs, valid):
    ax2.text(x, v + 0.012, f'{v:.2f}', ha='center', va='bottom',
             fontsize=FONT_SIZE - 2, color=COLORS[2])
ax2.set_ylabel('Valid-ORF rate')
ax2.set_xlabel(r'Amplification strength $\alpha$ (0 = baseline)')
ax2.set_ylim(0.55, 0.95)
ax2.set_xticks(xs)
ax2.set_xticklabels(xticklabels)

save_fig(fig, 'c3_dose_response', formats=FORMATS, out_dir=OUT_DIR)
plt.close(fig)
