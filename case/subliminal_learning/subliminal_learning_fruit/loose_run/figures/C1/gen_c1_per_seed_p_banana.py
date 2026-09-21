"""C1 — P(banana) per seed for teacher-arm students vs Ctrl-B students, with Ctrl-A baseline."""
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paper_plot_style import COLORS, REF_COLOR, save_fig, FONT_SIZE  # noqa: E402

DATA_PATH = '/path/to/project/multi_modal_B_loose1/runs/M0_verdict.json'
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

with open(DATA_PATH) as f:
    d = json.load(f)
ev = d['evidence']

teacher = ev['P_teacher_arm_per_seed']  # {seed_str: P}
ctrl_b = ev['P_ctrl_B_per_seed']
ctrl_a = ev['P_ctrl_A']

seeds = sorted(teacher.keys(), key=int)
seed_ints = [int(s) for s in seeds]
p_teacher = [teacher[s] for s in seeds]
p_ctrl_b = [ctrl_b[s] for s in seeds]

x = np.arange(len(seeds))
w = 0.36

fig, ax = plt.subplots(1, 1, figsize=(6.4, 3.4))
b1 = ax.bar(x - w / 2, p_teacher, w, label='Teacher-arm students', color=COLORS[0], edgecolor='none')
b2 = ax.bar(x + w / 2, p_ctrl_b, w, label='Ctrl-B students', color=COLORS[1], edgecolor='none')

ax.axhline(ctrl_a, color=REF_COLOR, linestyle='--', linewidth=1.2,
           label=f'Ctrl-A baseline (P = {ctrl_a:.3f}, seed-agnostic)')

ax.set_xlabel('Student training seed')
ax.set_ylabel('P(banana)')
ax.set_xticks(x)
ax.set_xticklabels(seed_ints)
ax.set_ylim(0, 1.0)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.legend(loc='upper right', frameon=False, ncol=1)

# Annotate teacher bars with value
for xi, v in zip(x, p_teacher):
    ax.text(xi - w / 2, v + 0.015, f'{v:.2f}', ha='center', va='bottom',
            fontsize=FONT_SIZE - 2, color=COLORS[0])
for xi, v in zip(x, p_ctrl_b):
    ax.text(xi + w / 2, v + 0.015, f'{v:.02f}', ha='center', va='bottom',
            fontsize=FONT_SIZE - 2, color=COLORS[1])

fig.tight_layout()
save_fig(fig, 'c1_per_seed_p_banana', formats=('pdf', 'png'), out_dir=OUT_DIR)
plt.close(fig)
