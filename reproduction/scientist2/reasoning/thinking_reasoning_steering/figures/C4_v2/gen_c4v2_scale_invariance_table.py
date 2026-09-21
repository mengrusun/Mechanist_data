"""C4_v2 table: Scale-invariant steering on Llama-8B.

Emits both a Markdown (.md) file for inline ledger embed and a LaTeX (.tex)
file for paper-write. Rows are the four scale-invariant steering controllers
plus the prompt suppress/amplify baselines and the thinking-intervention
controllers; columns are n, coherence, accuracy, and behaviour rate.

Data source: runs/iteration_round_1/M4_C4_scale_invariant/llama8b/results_summary.json
"""

import json
import os
import sys
from textwrap import dedent

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PROJECT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    'runs',
    'iteration_round_1',
    'M4_C4_scale_invariant',
    'llama8b',
    'results_summary.json',
)
OUT_DIR = HERE
NAME = 'c4v2_scale_invariance_table'

# Column order and label.
COLS = [
    ('controller', 'Controller'),
    ('n', 'n'),
    ('coherence_rate', 'Coherence'),
    ('accuracy', 'Accuracy'),
    ('behaviour_rate', 'Behaviour rate'),
]

# Pretty controller names (LaTeX-safe versions produced on the fly).
PRETTY = {
    'steering_negfrac15': 'Steering (α_frac = -0.15)',
    'steering_negfrac05': 'Steering (α_frac = -0.05)',
    'steering_posfrac05': 'Steering (α_frac = +0.05)',
    'steering_posfrac15': 'Steering (α_frac = +0.15)',
    'prompt_suppress': 'Prompt (suppress)',
    'prompt_amplify': 'Prompt (amplify)',
    'thinking_intervention_suppress': 'Thinking-intervention (suppress)',
}

# Order the rows exactly as we want them printed.
ROW_ORDER = [
    'steering_negfrac15',
    'steering_negfrac05',
    'steering_posfrac05',
    'steering_posfrac15',
    'prompt_suppress',
    'prompt_amplify',
    'thinking_intervention_suppress',
]


def fmt_pct(x: float) -> str:
    return f'{x * 100:.1f}%'


def latex_escape(s: str) -> str:
    return (
        s.replace('_', r'\_')
        .replace('α', r'$\alpha$')
        .replace('±', r'$\pm$')
        .replace('%', r'\%')
    )


def main() -> None:
    with open(DATA_PATH) as f:
        data = json.load(f)

    controllers = data['controllers']
    l_star = data['L_star']
    mean_norm = data['mean_residual_norm']

    rows = []
    for key in ROW_ORDER:
        if key not in controllers:
            continue
        s = controllers[key]['summary']
        rows.append({
            'controller': PRETTY.get(key, key),
            'n': s['n'],
            'coherence_rate': s['coherence_rate'],
            'accuracy': s['accuracy'],
            'behaviour_rate': s['behaviour_rate'],
        })

    # ---------- Markdown ----------
    md_header = '| ' + ' | '.join(label for _, label in COLS) + ' |'
    md_sep = '|' + '|'.join(['---'] * len(COLS)) + '|'
    md_body_lines = []
    for r in rows:
        cells = [
            r['controller'],
            str(r['n']),
            fmt_pct(r['coherence_rate']),
            fmt_pct(r['accuracy']),
            fmt_pct(r['behaviour_rate']),
        ]
        md_body_lines.append('| ' + ' | '.join(cells) + ' |')

    md_caption = (
        f'**Table.** Scale-invariant steering on DeepSeek-R1-Distill-Llama-8B at '
        f'L* = {l_star} (mean residual norm = {mean_norm:.2f}). '
        f'Steering coefficient = α_frac · mean_residual_norm(L*). '
        f'Coherence stays ≥ 0.98 for all four steering rows, but the range of '
        f'behaviour rates across steering rows equals the range across prompt rows '
        f'(no fine-grained advantage over prompting).'
    )

    md = md_caption + '\n\n' + md_header + '\n' + md_sep + '\n' + '\n'.join(md_body_lines) + '\n'
    with open(os.path.join(OUT_DIR, f'{NAME}.md'), 'w') as f:
        f.write(md)

    # ---------- LaTeX ----------
    n_cols = len(COLS)
    col_spec = 'l' + 'c' * (n_cols - 1)
    tex_header = ' & '.join(latex_escape(label) for _, label in COLS) + r' \\'
    tex_body_lines = []
    for r in rows:
        cells = [
            latex_escape(r['controller']),
            str(r['n']),
            fmt_pct(r['coherence_rate']).replace('%', r'\%'),
            fmt_pct(r['accuracy']).replace('%', r'\%'),
            fmt_pct(r['behaviour_rate']).replace('%', r'\%'),
        ]
        tex_body_lines.append(' & '.join(cells) + r' \\')

    tex_caption = (
        f'Scale-invariant steering on DeepSeek-R1-Distill-Llama-8B at '
        f'$L^* = {l_star}$ (mean residual norm $ = {mean_norm:.2f}$). '
        f'Steering coefficient $=$ $\\alpha_{{\\mathrm{{frac}}}} \\cdot '
        f'\\lVert h(L^*) \\rVert_2^{{\\mathrm{{mean}}}}$. '
        f'Coherence $\\geq 0.98$ across all four steering rows, but the '
        f'behaviour-rate spread across steering rows equals the spread across '
        f'prompt rows, eliminating the fine-grained advantage of the '
        f'parameterisation-dependent original claim.'
    )

    tex = dedent(r"""
    \begin{table}[t]
    \centering
    \caption{__CAPTION__}
    \label{tab:c4v2_scale_invariance}
    \begin{tabular}{__COLSPEC__}
    \toprule
    __HEADER__
    \midrule
    """).strip() + '\n' + '\n'.join(tex_body_lines) + '\n' + dedent(r"""
    \bottomrule
    \end{tabular}
    \end{table}
    """).lstrip()

    tex = (
        tex
        .replace('__COLSPEC__', col_spec)
        .replace('__HEADER__', tex_header)
        .replace('__CAPTION__', tex_caption)
    )

    with open(os.path.join(OUT_DIR, f'{NAME}.tex'), 'w') as f:
        f.write(tex)


if __name__ == '__main__':
    main()
