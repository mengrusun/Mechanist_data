"""C2 table: (rank_r, k_top) screen grid on layer_group=mid at α=-1; uniform collapse.
Source: refine-logs/EXPERIMENT_RESULTS.md#M2."""
from textwrap import dedent

OUT_DIR = "."

BASELINE = 0.762  # M2 baseline (no hook), n=50/lang

# From EXPERIMENT_RESULTS.md#M2 Stage-A screen (layer_group=mid, α=-1, n=25/lang × 11):
# rank_r=2 at k_top={4,8,12} → all three configs identical macro_acc = 0.065
# rank_r=8 at k_top={4,8,12} → all three configs identical macro_acc = 0.029
rows = [
    (2, 4,  0.065),
    (2, 8,  0.065),
    (2, 12, 0.065),
    (8, 4,  0.029),
    (8, 8,  0.029),
    (8, 12, 0.029),
]

# --- Markdown ---
md_lines = [
    "| rank_r | k_top | macro_acc @ α=−1 | Δ vs baseline | Verdict |",
    "|---|---|---|---|---|",
]
for r, k, acc in rows:
    md_lines.append(f"| {r} | {k} | {acc:.3f} | {(acc - BASELINE):+.3f} pp | catastrophic collapse |")
md_lines += [
    f"| — | — | **baseline = {BASELINE:.3f}** | 0.000 | reference (no hook) |",
    "",
    "*Layer group = `mid`; n = 25/lang × 11 languages = 275 problems per config, seed=42. Every screened (rank, k_top) config on layer_group=mid drops macro-accuracy by 50–73 pp vs baseline. Off-plan Gate G2 (aggregate regression at every k_top) fires — Claim 2's positive-gain prediction is refuted before matched-random and leave-one-out specificity tests are needed.*",
]
md = "\n".join(md_lines) + "\n"

with open(f"{OUT_DIR}/c2_screen_grid_uniform_collapse.md", "w") as f:
    f.write(md)

# --- LaTeX ---
tex = dedent(r"""
\begin{table}[t]
\centering
\caption{Null-space projection screen on layer\_group=\emph{mid} at $\alpha=-1$: macro-accuracy on Qwen-3-4B-Thinking / MGSM (n=25/lang $\times$ 11 langs, seed=42). Every $(r, k_{\text{top}})$ config collapses reasoning by 50--73 pp vs the no-hook baseline 0.762 (Off-plan Gate G2 fires).}
\label{tab:c2_screen_grid}
\begin{tabular}{ccccl}
\toprule
$r$ & $k_{\text{top}}$ & macro\_acc @ $\alpha=-1$ & $\Delta$ vs baseline (pp) & Verdict \\
\midrule
""").lstrip()

for r, k, acc in rows:
    tex += f"{r} & {k} & {acc:.3f} & {(acc - BASELINE)*100:+.1f} & catastrophic collapse \\\\\n"
tex += rf"\midrule --- & --- & baseline = {BASELINE:.3f} & 0.0 & reference (no hook) \\" + "\n"
tex += dedent(r"""\bottomrule
\end{tabular}
\end{table}
""")

with open(f"{OUT_DIR}/c2_screen_grid_uniform_collapse.tex", "w") as f:
    f.write(tex)

print("wrote c2_screen_grid_uniform_collapse.md and .tex")
