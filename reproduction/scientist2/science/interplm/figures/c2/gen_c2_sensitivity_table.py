"""
c2_sensitivity_table: (q_top, tau_F1) sensitivity sweep of Swiss-Prot concept coverage,
main ESM-2-650M vs ESM-2-8M model-swap.
Data source: verify/c2_sae_concept_alignment_gap/ROBUSTNESS.md (+ refine-logs/EXPERIMENT_RESULTS.md M2)
"""
import os
from textwrap import dedent

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# rows: (q_top, tau_F1, SAE_650M, Neu_650M, SAE_8M, Neu_8M)
# 650M numbers from refine-logs/EXPERIMENT_RESULTS.md M2 sensitivity sweep
# 8M numbers from verify/c2_sae_concept_alignment_gap/ROBUSTNESS.md variant #1
ROWS = [
    (0.95, 0.3, 64, 2, 56, 2),
    (0.95, 0.5, 16, 1, 13, 2),
    (0.95, 0.7, 3, 0, 2, 0),
    (0.99, 0.3, 65, 0, 58, 3),
    (0.99, 0.5, 15, 0, 14, 0),  # primary setting
    (0.99, 0.7, 3, 0, 2, 0),
]

PRIMARY = (0.99, 0.5)


def ratio(a, b):
    if b == 0:
        return "inf" if a > 0 else "n/a"
    return f"{a / b:.1f}x"


def md_ratio(a, b):
    return ratio(a, b).replace("x", "×").replace("inf", "∞")


def tex_ratio(a, b):
    r = ratio(a, b)
    if r == "inf":
        return r"$\infty$"
    if r == "n/a":
        return "n/a"
    return r.replace("x", r"$\times$")


# --- Markdown ---
md_lines = []
md_lines.append("| $q_{\\text{top}}$ | $\\tau_{F1}$ | SAE (650M) | Neu (650M) | Ratio (650M) | SAE (8M) | Neu (8M) | Ratio (8M) |")
md_lines.append("|---|---|---|---|---|---|---|---|")
for (q, t, s650, n650, s8, n8) in ROWS:
    marker = " *" if (q, t) == PRIMARY else ""
    md_lines.append(
        f"| {q:.2f}{marker} | {t:.1f} | {s650} | {n650} | {md_ratio(s650, n650)} | {s8} | {n8} | {md_ratio(s8, n8)} |"
    )
md_lines.append("")
md_lines.append("`*` primary setting used for the main verdict.")
md = "\n".join(md_lines) + "\n"

with open(f"{OUT_DIR}/c2_sensitivity_table.md", "w") as f:
    f.write(md)
print(f"Saved: {OUT_DIR}/c2_sensitivity_table.md")

# --- LaTeX ---
tex_rows = []
for (q, t, s650, n650, s8, n8) in ROWS:
    marker = "$^{\\ast}$" if (q, t) == PRIMARY else ""
    tex_rows.append(
        f"    {q:.2f}{marker} & {t:.1f} & {s650} & {n650} & {tex_ratio(s650, n650)} & {s8} & {n8} & {tex_ratio(s8, n8)} \\\\"
    )
tex = dedent(r"""
\begin{table}[t]
\centering
\caption{Sensitivity sweep of Swiss-Prot concept coverage (covered / 400) across the ($q_{\text{top}}$, $\tau_{F1}$) grid, for the main ESM-2-650M experiment and the ESM-2-8M model-swap verify variant. The SAE~$\gg$~neurons gap is preserved across the 80$\times$ parameter-count reduction. $^{\ast}$~primary setting.}
\label{tab:c2_sensitivity}
\begin{tabular}{cc | ccc | ccc}
\toprule
$q_{\text{top}}$ & $\tau_{F1}$ & SAE (650M) & Neu (650M) & Ratio & SAE (8M) & Neu (8M) & Ratio \\
\midrule
""").strip() + "\n" + "\n".join(tex_rows) + "\n" + dedent(r"""
\bottomrule
\end{tabular}
\end{table}
""").rstrip() + "\n"

with open(f"{OUT_DIR}/c2_sensitivity_table.tex", "w") as f:
    f.write(tex)
print(f"Saved: {OUT_DIR}/c2_sensitivity_table.tex")
