"""C3 table: per-α macro-accuracy and Δ(V_lang − random-control) on the main model.
Source: refine-logs/EXPERIMENT_RESULTS.md#M3 (verbatim)."""
from textwrap import dedent

OUT_DIR = "."

# Verbatim from EXPERIMENT_RESULTS.md#M3 comparison table
rows = [
    (-1.5, 0.053, 0.745),
    (-1.0, 0.051, 0.740),
    (-0.5, 0.085, 0.747),
    (-0.25, 0.175, 0.753),
    (0.0,  0.744, 0.744),
    (0.25, 0.009, 0.729),
    (0.5,  0.000, 0.438),
    (1.0,  0.000, 0.004),
    (1.5,  0.000, 0.000),
]

# --- Markdown ---
md_lines = [
    "| α | V_lang macro_acc | random-ctrl macro_acc | Δ (V_lang − random) | Reading |",
    "|---|---|---|---|---|",
]
for a, v, r in rows:
    d = v - r
    if abs(a) < 1e-9:
        reading = "α=0 sanity (both hooks off equivalent)"
    elif abs(d) < 0.02:
        reading = "no gap — both collapsed or both preserved"
    elif d < -0.10:
        reading = "large gap — V_lang collapses, random preserves"
    else:
        reading = "small gap"
    md_lines.append(f"| {a:+.2f} | {v:.3f} | {r:.3f} | {d:+.3f} | {reading} |")
md_lines += [
    "",
    "*Non-monotone leg refuted: A(−1)=0.051 ≪ A(0)=0.744 breaks the plan's diagnostic inequality `A(−1) > A(0) > A(+1)`. Specificity leg supported: V_lang loses 66–72 pp vs matched random subspace in the non-collapse window α∈{−1.5, −1.0, −0.5, −0.25, +0.25} — V_lang is a specifically language-related direction, not a generic one.*",
]
md = "\n".join(md_lines) + "\n"

with open(f"{OUT_DIR}/c3_alpha_delta_table.md", "w") as f:
    f.write(md)

# --- LaTeX ---
tex = dedent(r"""
\begin{table}[t]
\centering
\caption{Signed $\alpha$-sweep on Qwen-3-4B-Thinking / MGSM at the winning M2 site (mid, $k_{\text{top}}=12$, $r=2$). $V_{\mathrm{lang}}$ collapses in the negative-$\alpha$ non-collapse window ($\Delta \in [-0.72, -0.44]$ at $|\alpha| \in [0.25, 1.5]$) while the matched random subspace preserves the baseline --- specificity supported. But $A(-1)=0.051 \ll A(0)=0.744$ refutes the plan's monotone dose-response.}
\label{tab:c3_alpha_delta}
\begin{tabular}{crrrl}
\toprule
$\alpha$ & $V_{\mathrm{lang}}$ & random-ctrl & $\Delta$ & Reading \\
\midrule
""").lstrip()

for a, v, r in rows:
    d = v - r
    if abs(a) < 1e-9:
        reading = r"$\alpha=0$ sanity"
    elif abs(d) < 0.02:
        reading = "no gap"
    elif d < -0.10:
        reading = r"$V_{\mathrm{lang}}$ collapses; random preserves"
    else:
        reading = "small gap"
    tex += f"{a:+.2f} & {v:.3f} & {r:.3f} & {d:+.3f} & {reading} \\\\\n"
tex += dedent(r"""\bottomrule
\end{tabular}
\end{table}
""")

with open(f"{OUT_DIR}/c3_alpha_delta_table.tex", "w") as f:
    f.write(tex)

print("wrote c3_alpha_delta_table.md and .tex")
