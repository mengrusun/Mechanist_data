"""C3 — full recovery-metric comparison table across necessity + sufficiency + models."""
import json, os
from textwrap import dedent

OUT_DIR = os.path.dirname(__file__)

with open('results/M2_necessity.json') as f:
    m2 = json.load(f)
with open('results/M3_sufficiency.json') as f:
    m3 = json.load(f)
with open('results/M5_gemma9b.json') as f:
    m5 = json.load(f)

# Mistral row
mistral = {
    'model': 'Mistral-7B-v0.1',
    'LD_nec':  m2['recovery']['logit_diff'],
    'PD_nec':  m2['recovery']['prob_diff'],
    'spec_nec': m2['specificity_gap'],
    'LD_suf':  m3['sufficient_recovery']['logit_diff'],
    'PD_suf':  m3['sufficient_recovery']['prob_diff'],
    'KL_suf':  m3['sufficient_recovery']['KL'],
    'spec_suf': m3['specificity_gap'],
}
# Gemma row
gemma = {
    'model': 'Gemma-2-9B',
    'LD_nec':  m5['necessity_recovery']['logit_diff'],
    'PD_nec':  m5['necessity_recovery']['prob_diff'],
    'spec_nec': None,  # M5 does not compute specificity separately
    'LD_suf':  m5['sufficiency_recovery']['logit_diff'],
    'PD_suf':  m5['sufficiency_recovery']['prob_diff'],
    'KL_suf':  m5['sufficiency_recovery']['KL'],
    'spec_suf': None,
}
rows = [mistral, gemma]

def fmt(v):
    if v is None:
        return '—'
    if isinstance(v, float):
        return f'{v:.3f}'
    return str(v)

cols_md = ['model', 'LD nec (≥0.8)', 'PD nec', 'specificity nec (≥0.6)', 'LD suf (≥0.8)', 'PD suf', 'KL suf', 'specificity suf']
md_header = "| " + " | ".join(cols_md) + " |\n" + "|" + "|".join(["---"] * len(cols_md)) + "|"
md_lines = [md_header]
for r in rows:
    md_lines.append(
        f"| {r['model']} | {fmt(r['LD_nec'])} | {fmt(r['PD_nec'])} | {fmt(r['spec_nec'])} | "
        f"{fmt(r['LD_suf'])} | {fmt(r['PD_suf'])} | {fmt(r['KL_suf'])} | {fmt(r['spec_suf'])} |"
    )
md_lines.append("")
md_lines.append("KL recovery on the anchor cell is de-emphasised because the baseline KL(clean‖corrupt) is 0.043, so the recovery ratio blows up; LD and PD are the definitive metrics and tell the same story.")
md = "\n".join(md_lines) + "\n"

with open(os.path.join(OUT_DIR, 'c3_metric_comparison_table.md'), 'w') as f:
    f.write(md)

# LaTeX
def fmt_tex(v):
    if v is None:
        return '$-$'
    if isinstance(v, float):
        return f'{v:.3f}'
    return str(v)

tex_rows = []
for r in rows:
    tex_rows.append(
        f"{r['model']} & {fmt_tex(r['LD_nec'])} & {fmt_tex(r['PD_nec'])} & {fmt_tex(r['spec_nec'])} & "
        f"{fmt_tex(r['LD_suf'])} & {fmt_tex(r['PD_suf'])} & {fmt_tex(r['KL_suf'])} & {fmt_tex(r['spec_suf'])} \\\\"
    )

tex = dedent(r"""
\begin{table*}[t]
\centering
\caption{Full recovery-metric comparison for C3 (necessity + sufficiency) across Mistral-7B and Gemma-2-9B. LD (logit-diff) and PD (prob-diff) tell a consistent necessary-but-not-sufficient story. KL recovery is de-emphasised because the anchor-cell baseline KL(clean$\parallel$corrupt) is $0.043$, so the recovery ratio suffers a denominator-collapse artifact.}
\label{tab:c3_metric_comparison}
\begin{tabular}{lccccccc}
\toprule
model & LD nec ($\geq 0.8$) & PD nec & spec.\ nec ($\geq 0.6$) & LD suf ($\geq 0.8$) & PD suf & KL suf & spec.\ suf \\
\midrule
""") + "\n".join(tex_rows) + "\n" + r"""\bottomrule
\end{tabular}
\end{table*}
"""

with open(os.path.join(OUT_DIR, 'c3_metric_comparison_table.tex'), 'w') as f:
    f.write(tex)

print(f'Wrote {OUT_DIR}/c3_metric_comparison_table.md and .tex')
