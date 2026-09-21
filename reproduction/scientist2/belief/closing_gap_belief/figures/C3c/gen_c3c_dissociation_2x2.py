"""C3c table — 2×2 dissociation-when-disagree cells (failed / underpowered diagnostic)."""
import json
import sys
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).parent

with open(ROOT / "artifacts" / "dissociation.json") as f:
    d = json.load(f)

cells = d["cells"]
test = d["test"]

# Rows = probe (high/low), Cols = verbal (low/high) — the pre-registered contrast
def fmt_pct(x):
    return f"{100*x:.1f}%"

rows = [
    ("probe HIGH", cells["probe_high_verbal_low"]["n"],  fmt_pct(cells["probe_high_verbal_low"]["accuracy"]),
                    cells["probe_high_verbal_high"]["n"], fmt_pct(cells["probe_high_verbal_high"]["accuracy"])),
    ("probe LOW",  cells["probe_low_verbal_low"]["n"],   fmt_pct(cells["probe_low_verbal_low"]["accuracy"]),
                    cells["probe_low_verbal_high"]["n"],  fmt_pct(cells["probe_low_verbal_high"]["accuracy"])),
]

md_lines = ["| probe \\ verbal | verbal LOW: n | verbal LOW: acc | verbal HIGH: n | verbal HIGH: acc |",
            "|---|---|---|---|---|"]
for r in rows:
    md_lines.append("| " + " | ".join(str(x) for x in r) + " |")
md_lines.append("")
md_lines.append(f"**Test**: two-proportion z on the (probe LOW, verbal HIGH) vs. (probe LOW, verbal LOW) contrast — z = {test['z']:.2f}, p_one_sided = {test['p_one_sided']:.3f}, significant at 0.05? {'**yes**' if test['significant_at_0.05'] else '**no**'}.")
md_lines.append("")
md_lines.append(f"**Success criterion**: {d['success_criterion']} — **passes: {'yes' if d['passes'] else 'no'}**.")
md_lines.append("")
md_lines.append(f"Root cause: the (probe LOW, verbal LOW) cell has n=4 of 2000 test — the extreme skew of verbalized-c toward 100 (96% ≥ 95) collapses the load-bearing cell and makes the test untestable in its planned form.")
md = "\n".join(md_lines) + "\n"
(OUT_DIR / "c3c_dissociation_2x2.md").write_text(md)

tex = dedent(r"""
\begin{table}[t]
\centering
\caption{2$\times$2 dissociation-when-disagree cells. The $(\text{probe LOW}, \text{verbal LOW})$ cell has $n=4$ of 2{,}000 test, driving the failed diagnostic; 96\% of verbalized $c \ge 95$ under Llama-3.1-8B-Instruct's default prompting collapses the load-bearing contrast.}
\label{tab:c3c_dissociation}
\begin{tabular}{lcccc}
\toprule
 & verbal LOW ($n$) & verbal LOW (acc) & verbal HIGH ($n$) & verbal HIGH (acc) \\
\midrule
""").strip() + "\n"
for r in rows:
    tex += " & ".join(str(x).replace("%", r"\%") for x in r) + r" \\" + "\n"
tex += dedent(r"""
\bottomrule
\end{tabular}
""").strip() + "\n" + dedent(rf"""
\vspace{{0.5em}}
\noindent\footnotesize Two-proportion $z$ test: $z = {test['z']:.2f}$, $p_{{\text{{one-sided}}}} = {test['p_one_sided']:.3f}$; not significant at 0.05. Diagnostic \textbf{{fails}} (accuracy sign reversed from predicted).
\end{{table}}
""").strip() + "\n"
(OUT_DIR / "c3c_dissociation_2x2.tex").write_text(tex)
