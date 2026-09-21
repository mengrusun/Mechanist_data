"""C1 table: V_lang classifier vs orthogonal-complement classifier at each layer_group's best (n_probe, rank_r).
Source: refine-logs/EXPERIMENT_RESULTS.md#M1 (best-per-layer-group table)."""
from textwrap import dedent

OUT_DIR = "."

# Extracted verbatim from refine-logs/EXPERIMENT_RESULTS.md#M1
rows = [
    ("early", 250, 16, 43, 0.968, 0.491, 0.113, "V_lang PASS; complement FAIL (≤0.20 required)"),
    ("mid",  500, 16, 42, 0.841, 0.373, 0.158, "V_lang FAIL (<0.90); complement FAIL"),
    ("all_non_upper", 1000, 32, 43, 0.864, 0.332, 0.126, "V_lang FAIL (<0.90); complement FAIL"),
]

# --- Markdown ---
md_lines = [
    "| layer_group | n_probe | rank_r | seed | V_lang classifier | Complement classifier | Median cos(V, content) | Predicate |",
    "|---|---|---|---|---|---|---|---|",
]
for lg, n, r, s, vc, cc, cos, pred in rows:
    md_lines.append(f"| {lg} | {n} | {r} | {s} | {vc:.3f} | {cc:.3f} | {cos:.3f} | {pred} |")
md_lines += [
    "",
    "*Baseline (chance) ≈ 1/11 ≈ 0.091 for the language classifier. V_lang passes the ≥0.90 bar only at `early` with n_probe=250, rank_r=16 (`small probe set` qualifier satisfied). Complement classifier never collapses toward chance — the orthogonal-decomposition leg of Claim 1 is only approximately satisfied.*",
]
md = "\n".join(md_lines) + "\n"

with open(f"{OUT_DIR}/c1_v_lang_vs_complement_by_layergroup.md", "w") as f:
    f.write(md)

# --- LaTeX (mirror) ---
tex = dedent(r"""
\begin{table}[t]
\centering
\caption{V\_lang classifier vs orthogonal-complement classifier at each layer group's best $(n_{\text{probe}}, r)$ on Qwen-3-4B-Thinking. V\_lang passes the $\geq0.90$ bar with $n_{\text{probe}}=250$ at layer group \emph{early}; the complement classifier never collapses to the chance-adjacent $\leq 0.20$ threshold (chance $\approx 1/11 \approx 0.091$).}
\label{tab:c1_v_lang_vs_complement}
\begin{tabular}{lccccccl}
\toprule
layer\_group & $n_{\text{probe}}$ & $r$ & seed & V\_lang acc. & Complement acc. & median $\cos(V, \text{content})$ & Predicate \\
\midrule
""").lstrip()

for lg, n, r, s, vc, cc, cos, pred in rows:
    tex += f"{lg} & {n} & {r} & {s} & {vc:.3f} & {cc:.3f} & {cos:.3f} & {pred} \\\\\n"

tex += dedent(r"""\bottomrule
\end{tabular}
\end{table}
""")

with open(f"{OUT_DIR}/c1_v_lang_vs_complement_by_layergroup.tex", "w") as f:
    f.write(tex)

print("wrote c1_v_lang_vs_complement_by_layergroup.md and .tex")
