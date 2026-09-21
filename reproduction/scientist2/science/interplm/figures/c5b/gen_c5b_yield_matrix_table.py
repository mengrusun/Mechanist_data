"""
c5b_yield_matrix_table: Yield matrix (target feature x arm x dose) for the M6 steering assay.
Data source: refine-logs/EXPERIMENT_RESULTS.md#M6 (yield ranges across alpha in {0.5, 1, 2, 4} * sigma_f).
"""
import os
from textwrap import dedent

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# From EXPERIMENT_RESULTS.md M6 table (mean yield across 30 gens = 10 seqs x 3 seeds)
# rows: (feature_id, property, no_steer, sae_clamp_range, mean_add_range, random_clamp_range)
ROWS = [
    (3998, 'TM helix',   0.333, '0.200 (all α)',      '0.133–0.200', '0.200 (all α)'),
    (4209, 'Zn finger',  0.133, '0.100 (all α)',      '0.100 (all α)', '0.100 (all α)'),
    (1240, 'SP core',    0.167, '0.133 (all α)',      '0.133 (all α)', '0.133 (all α)'),
]

# --- Markdown ---
md_lines = []
md_lines.append("| Feature | Property | no_steer (baseline) | sae_clamp (α∈{0.5,1,2,4}·σ_f) | mean_add | random_clamp |")
md_lines.append("|---|---|---|---|---|---|")
for (fid, prop, ns, sc, ma, rc) in ROWS:
    md_lines.append(f"| {fid} | {prop} | **{ns:.3f}** | {sc} | {ma} | {rc} |")
md_lines.append("")
md_lines.append("Notes: yield = fraction of the 30 generated sequences (10 seqs × 3 seeds) that satisfy the target property's rule-based checker (Kyte–Doolittle window for TM, ProSite regex for Zn, SignalP-like N-terminal core rule for SP). Baseline yields dominate every steered arm on every feature. Plausibility band-pass ≥ 0.67 on all steered arms, so drops are not off-distribution collapse — the steering is ineffective, not destructive, under this narrow ≤1-OOM dose ladder.")
md = "\n".join(md_lines) + "\n"

with open(f"{OUT_DIR}/c5b_yield_matrix_table.md", "w") as f:
    f.write(md)
print(f"Saved: {OUT_DIR}/c5b_yield_matrix_table.md")

# --- LaTeX ---
tex_rows = []
for (fid, prop, ns, sc, ma, rc) in ROWS:
    tex_rows.append(f"{fid} & {prop} & \\textbf{{{ns:.3f}}} & {sc} & {ma} & {rc} \\\\")
tex_rows_str = "\n".join(tex_rows)

tex = dedent(r"""
\begin{table}[t]
\centering
\caption{Mean yield per (target-property feature $\times$ steering arm) for the M6 clamp assay ($\alpha \in \{0.5, 1, 2, 4\} \cdot \sigma_f$, averaged over 30 generations = 10 seqs $\times$ 3 seeds). The no-steer baseline dominates every steered arm on every feature; plausibility band-pass~$\geq 0.67$ across all steered arms shows the drops are not off-distribution collapse. Interpretation: steering is ineffective (not destructive) under this narrow $\leq 1$-OOM dose ladder + rule-based checkers.}
\label{tab:c5b_yield_matrix}
\begin{tabular}{ll | c | ccc}
\toprule
Feat.\ id & Property & no\_steer & sae\_clamp & mean\_add & random\_clamp \\
\midrule
""").strip() + "\n" + tex_rows_str + "\n" + dedent(r"""
\bottomrule
\end{tabular}
\end{table}
""").rstrip() + "\n"

with open(f"{OUT_DIR}/c5b_yield_matrix_table.tex", "w") as f:
    f.write(tex)
print(f"Saved: {OUT_DIR}/c5b_yield_matrix_table.tex")
