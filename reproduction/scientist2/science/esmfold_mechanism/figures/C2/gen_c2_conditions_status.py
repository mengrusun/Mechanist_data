import json, os
from textwrap import dedent

d = json.load(open('results/M2/summary_stats.json'))
rows = d['per_condition_stats']

# Known intended conditions (from EXPERIMENT_PLAN.md)
INTENDED = ['seq2pair_donor', 'pair2seq_donor', 'seq2pair_zero', 'seq2pair_matched_ctrl']

# Build lookup
present = {r['condition']: r for r in rows}

# Note from EXPERIMENT_RESULTS.md: only seq2pair_donor executed; other three had 3 records each (hook bug)
KNOWN_N_ATTEMPTED = {
    'seq2pair_donor': (176, 'complete'),
    'pair2seq_donor': (3, 'hook shape bug (expected [B,L,S], actual [B,num_heads,L,L,32])'),
    'seq2pair_zero':  (3, 'hook shape bug (same as pair2seq)'),
    'seq2pair_matched_ctrl': (3, 'hook shape bug (same as pair2seq)'),
}

rows_out = []
for cond in INTENDED:
    if cond in present:
        p = present[cond]
        n = p.get('n_chains')
        delta = p.get('delta_rate')
        pval = p.get('test', {}).get('p') if isinstance(p.get('test'), dict) else None
        status = 'complete'
        note = ''
        if n and n <= 5:
            status = 'hook shape bug'
            note = 'only 3 records — hook shape bug'
    else:
        n = KNOWN_N_ATTEMPTED.get(cond, (0,''))[0]
        delta = None
        pval = None
        status = 'hook shape bug'
        note = KNOWN_N_ATTEMPTED.get(cond, (0,''))[1]

    rows_out.append({
        'cond': cond,
        'n': n,
        'delta': delta,
        'p': pval,
        'status': status,
        'note': note,
    })

# Markdown
md_lines = [
    '| Condition | N (records written) | Δ hairpin rate | Wilcoxon p | Status |',
    '|-----------|--------------------:|---------------:|-----------:|--------|',
]
for r in rows_out:
    delta_s = f"{r['delta']:+.3f}" if r['delta'] is not None else '—'
    p_s     = f"{r['p']:.3f}"      if r['p'] is not None else '—'
    md_lines.append(f"| `{r['cond']}` | {r['n']} | {delta_s} | {p_s} | {r['status']} |")

md_body = '\n'.join(md_lines) + '\n\n_Note: `pair2seq_donor`, `seq2pair_zero`, and `seq2pair_matched_ctrl` failed to write full records due to a `pair_to_sequence` hook shape-mismatch bug in `scripts/m2_worker.py` (expected `[B, L, S]`, actual `[B, num_heads, L, L, 32]`). The three specificity-control cells above are therefore evidence-incomplete, not scientifically null. Fix + re-run recommended in iteration._\n'

with open('figures/C2/c2_conditions_status.md', 'w') as f:
    f.write(md_body)

# LaTeX
def esc(s):
    return s.replace('_', r'\_')

tex = dedent(r"""
\begin{table}[t]
\centering
\caption{M2 evidence completeness --- seq2pair\_donor N=176 (only fully executed condition); pair2seq / zero-abl / matched-ctrl N=3 each due to pair\_to\_sequence hook shape bug.}
\label{tab:c2_conditions_status}
\begin{tabular}{lrrrl}
\toprule
Condition & N (records written) & $\Delta$ hairpin rate & Wilcoxon $p$ & Status \\
\midrule
""")
for r in rows_out:
    delta_s = f"{r['delta']:+.3f}" if r['delta'] is not None else '---'
    p_s     = f"{r['p']:.3f}"      if r['p'] is not None else '---'
    tex += f"\\texttt{{{esc(r['cond'])}}} & {r['n']} & {delta_s} & {p_s} & {esc(r['status'])} \\\\\n"
tex += dedent(r"""\bottomrule
\end{tabular}
\end{table}
""")

with open('figures/C2/c2_conditions_status.tex', 'w') as f:
    f.write(tex)

print('OK c2_conditions_status')
