"""C2 — role-dissociation table (anchor cell + cross-cell Jaccard stability)."""
import json, os
from textwrap import dedent

OUT_DIR = os.path.dirname(__file__)

with open('results/M4_roles.json') as f:
    m4 = json.load(f)
with open('results/M4stab.json') as f:
    m4stab = json.load(f)

anchor = m4['per_cell']['k3_chain2_natural']
diss = anchor['dissociation']
p    = anchor['null_shuffle_p_value']
dom_median_anchor = anchor['median_dominance_ratio']

stab_data = m4stab['stability']

# Per M4stab metadata: single pair (k5_chain2_natural, k3_chain3_natural). We label the two Jaccards.
def jaccard_pair(role):
    entry = stab_data[role]['pairwise'][0]
    return entry['jaccard']

roles = ['fact', 'rule', 'answer']
rows = []
for r in roles:
    rows.append({
        'role': r,
        'dissociation': f"{diss[r]:.3f}",
        'null_shuffle_p': f"{p[r]:.2f}",
        'jaccard_k5c2_vs_k3c3': f"{jaccard_pair(r):.2f}",
        'signal': ('borderline ✓' if r == 'fact' else 'random-like ✗'),
    })

# Median dominance is per-cell, not per-role. Add as a footnote row/entry.
# Markdown
cols = ['role', 'dissociation (target ≥ 0.1)', 'null-shuffle p (target ≤ 0.01)', 'Jaccard k5c2↔k3c3 (target ≥ 0.6)', 'signal']
md_header = "| " + " | ".join(cols) + " |\n" + "|" + "|".join(["---"] * len(cols)) + "|"
md_lines = [md_header]
for row in rows:
    md_lines.append(f"| {row['role']} | {row['dissociation']} | {row['null_shuffle_p']} | {row['jaccard_k5c2_vs_k3c3']} | {row['signal']} |")
md_lines.append("")
md_lines.append(f"Median per-component dominance ratio on the anchor cell: **{dom_median_anchor:.2f}** (target ≥ 2.0) — refuted.")
md = "\n".join(md_lines) + "\n"

with open(os.path.join(OUT_DIR, 'c2_role_dissociation_table.md'), 'w') as f:
    f.write(md)

# LaTeX (target ≥ symbols rendered with $\geq$)
tex_rows = []
for row in rows:
    signal = 'borderline $\\checkmark$' if 'borderline' in row['signal'] else 'random-like $\\times$'
    tex_rows.append(f"{row['role']} & {row['dissociation']} & {row['null_shuffle_p']} & {row['jaccard_k5c2_vs_k3c3']} & {signal} \\\\")

tex = dedent(r"""
\begin{table}[t]
\centering
\caption{Role-dissociation statistics for C2 on Mistral-7B. Only the ``fact'' role has borderline signal; ``rule'' and ``answer'' partitions are statistically indistinguishable from random. Median per-component dominance ratio on the anchor cell is """ + f"{dom_median_anchor:.2f}" + r""" (target $\geq 2.0$) — modular decomposition into three role-specialists is refuted.}
\label{tab:c2_role_dissociation}
\begin{tabular}{lcccc}
\toprule
role & dissociation ($\geq 0.1$) & null-shuffle $p$ ($\leq 0.01$) & Jaccard k5c2$\leftrightarrow$k3c3 ($\geq 0.6$) & signal \\
\midrule
""") + "\n".join(tex_rows) + "\n" + r"""\bottomrule
\end{tabular}
\end{table}
"""

with open(os.path.join(OUT_DIR, 'c2_role_dissociation_table.tex'), 'w') as f:
    f.write(tex)

print(f'Wrote {OUT_DIR}/c2_role_dissociation_table.md and .tex')
