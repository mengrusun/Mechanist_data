"""C5 head-to-head table — probe (linear + shallow_mlp) vs Llama Guard 3 8B.

Data: results/m5/claim5_verdict.json
Output: figures/C5/c5_probe_vs_llamaguard_headtohead.{md,tex}
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))

d = json.load(open(os.path.join(PROJECT_ROOT, "results/m5/claim5_verdict.json")))
variants = d["variants_evaluated"]
per_variant = d["per_variant"]


def fmt_time(secs):
    if secs is None:
        return "—"
    if secs < 1e-3:
        return f"{secs * 1e6:.0f} µs"
    if secs < 1:
        return f"{secs * 1e3:.1f} ms"
    return f"{secs:.3f} s"


def fmt_flops(x):
    if x is None:
        return "—"
    if x >= 1e9:
        return f"{x / 1e9:.2f} G"
    if x >= 1e6:
        return f"{x / 1e6:.1f} M"
    if x >= 1e3:
        return f"{x / 1e3:.1f} K"
    return f"{x:.0f}"


def fmt_ratio(x):
    if x is None or x >= 1:
        return f"{x:.4f}" if x is not None else "—"
    return f"{x:.2e}"


# Build one row per system: linear_probe, shallow_mlp, Llama Guard 3 8B (baseline)
first = per_variant[variants[0]]
lg_row = {
    "system": "Llama Guard 3 8B (baseline)",
    "auroc": first["auroc_llamaguard"],
    "f1": first["f1@fpr5_llamaguard"],
    "wallclock": first["wallclock_per_query_llamaguard"],
    "flops": first["flops_per_query_llamaguard"],
    "ratio": 1.0,
}

rows = []
for v in variants:
    r = per_variant[v]
    rows.append({
        "system": v,
        "auroc": r["auroc_probe"],
        "f1": r["f1@fpr5_probe"],
        "wallclock": r["wallclock_per_query_probe"],
        "flops": r["flops_per_query_probe"],
        "ratio": r["compute_ratio_probe_over_lg"],
    })
rows.append(lg_row)

# ---- Markdown ----
md = (
    "| Classifier | AUROC | F1 @ FPR=5% | Per-query wall-clock | Per-query FLOPs | Compute ratio vs LG |\n"
    "|---|---|---|---|---|---|\n"
)
for r in rows:
    md += (
        f"| **{r['system']}** | {r['auroc']:.4f} | {r['f1']:.3f} | "
        f"{fmt_time(r['wallclock'])} | {fmt_flops(r['flops'])} | {fmt_ratio(r['ratio'])} |\n"
    )
with open(os.path.join(HERE, "c5_probe_vs_llamaguard_headtohead.md"), "w") as f:
    f.write(md)

# ---- LaTeX ----
tex = r"""\begin{table}[t]
\centering
\caption{C5 — head-to-head against Llama Guard 3 8B on the 400-item mixed test set (bare-harmful vs benign/safe-lookalike; 0 successful-jailbreak items due to M4 ASR=0). Both a 1-d logistic-regression probe on $\langle \text{activation}, h\rangle$ and a 2-layer 64-unit MLP match/edge-past LG on AUROC (1.000 vs 0.9992), at $\sim\!10^{-7}$--$10^{-8}$ of the per-query FLOPs and $\sim\!440\times$ / $320\times$ per-query wall-clock speedup. Scope narrowed in iteration~1 to bare-harmful vs benign/safe-lookalike (jailbreak-detection framing deferred until an attack family with ASR>0 is available).}
\label{tab:c5_probe_vs_lg}
\begin{tabular}{lrrrrr}
\toprule
Classifier & AUROC & F1 @ FPR=5\% & Wall-clock/query & FLOPs/query & Compute ratio vs LG \\
\midrule
"""
for r in rows:
    tex += (
        f"{r['system'].replace('_', r'\_')} & {r['auroc']:.4f} & {r['f1']:.3f} & "
        f"{fmt_time(r['wallclock'])} & {fmt_flops(r['flops'])} & {fmt_ratio(r['ratio'])} \\\\\n"
    )
tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
with open(os.path.join(HERE, "c5_probe_vs_llamaguard_headtohead.tex"), "w") as f:
    f.write(tex)
