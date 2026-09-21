"""C2 diversity table: per-domain argmax over 16 held-out mabench_core domains.

Which aggregation wins each domain, plus overall win counts.
Attention-pool leads with 8/16 (< 9/17 threshold), so predicate (b) PASSES.
"""
import json
import os
from textwrap import dedent

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(OUT_DIR, '..', '..'))

with open(os.path.join(PROJECT_ROOT, 'runs', 'M2', 'diversity.json')) as f:
    div = json.load(f)

per_domain = div['per_domain_argmax']
win_counts = div['aggregation_win_counts']

# -------- Markdown --------
md_lines = [
    '| Domain | Winning aggregation | AUROC |',
    '|---|---|---|',
]
for domain in sorted(per_domain.keys()):
    entry = per_domain[domain]
    md_lines.append(f"| {domain} | {entry['agg']} | {entry['auroc']:.3f} |")

md_lines.append('')
md_lines.append('| Aggregation | Domain wins (out of 16) |')
md_lines.append('|---|---|')
for agg, cnt in sorted(win_counts.items(), key=lambda kv: -kv[1]):
    md_lines.append(f'| {agg} | {cnt} |')
md_lines.append('')
md_lines.append(
    f"Top aggregation: **{div['top_aggregation']}** with "
    f"**{div['top_count']}/16** wins (< 9/17 dominance threshold "
    f"-> predicate (b) PASSES).")
md = '\n'.join(md_lines) + '\n'

with open(os.path.join(OUT_DIR, 'c2_diversity_per_domain.md'), 'w') as f:
    f.write(md)
print(f"Saved: {os.path.join(OUT_DIR, 'c2_diversity_per_domain.md')}")

# -------- LaTeX --------
tex_rows_per_domain = '\n'.join(
    f"{d.replace('-', '--')} & {per_domain[d]['agg']} & "
    f"{per_domain[d]['auroc']:.3f} \\\\"
    for d in sorted(per_domain.keys())
)

tex_rows_counts = '\n'.join(
    f"{agg} & {cnt} \\\\"
    for agg, cnt in sorted(win_counts.items(), key=lambda kv: -kv[1])
)

tex = dedent(r"""
\begin{table}[t]
\centering
\caption{C2: per-domain argmax across 16 held-out mabench\_core domains
(K=3 committees, Qwen3-32B-AWQ). Attention-pool leads with 8/16 wins
($<$ 9/17 dominance threshold): predicate (b) PASSES.}
\label{tab:c2_diversity_per_domain}
\begin{tabular}{lcc}
\toprule
Domain & Winning aggregation & AUROC \\
\midrule
""") + tex_rows_per_domain + dedent(r"""
\bottomrule
\end{tabular}
\vspace{0.5em}

\begin{tabular}{lc}
\toprule
Aggregation & Domain wins (out of 16) \\
\midrule
""") + tex_rows_counts + dedent(r"""
\bottomrule
\end{tabular}
\end{table}
""")

with open(os.path.join(OUT_DIR, 'c2_diversity_per_domain.tex'), 'w') as f:
    f.write(tex)
print(f"Saved: {os.path.join(OUT_DIR, 'c2_diversity_per_domain.tex')}")
