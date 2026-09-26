"""Table: judge-swap replication at LR=1e-3 (gpt-5.4 vs gpt-4o).

Reads results/m0_headline.json (gpt-5.4 judge)
+   verify/C1_cross_modal_subliminal_transfer/variants/model-swap-judge-gpt4o/result.json (gpt-4o).
Emits <stem>.md (inline for the ledger) and <stem>.tex (paper-write).
"""
import json
import os
from textwrap import dedent

OUT_DIR = "figures/C1"
STEM = "c1_judge_swap_replication"
os.makedirs(OUT_DIR, exist_ok=True)

with open("results/m0_headline.json") as f:
    hd = json.load(f)
with open("verify/C1_cross_modal_subliminal_transfer/variants/model-swap-judge-gpt4o/result.json") as f:
    v = json.load(f)


def fmt_drop(d):
    sign = "+" if d >= 0 else "−"
    return f"{sign}{abs(d):.2f}"


def ci(ci_dict):
    lo = ci_dict["lo"] * 100
    hi = ci_dict["hi"] * 100
    return f"[{lo:+.1f}, {hi:+.1f}]"


seeds = ["100", "200", "300"]
rows = []
for s in seeds:
    drop_5_4 = hd["per_seed_drop_pp"][s]
    drop_4o = v["per_seed_drop_pp"][s]
    ci_5_4 = ci(hd["per_seed_paired_bootstrap_ci"][s])
    ci_4o = ci(v["per_seed_paired_bootstrap_ci"][s])
    delta = drop_4o - drop_5_4
    pass_5_4 = hd["per_seed_pass"][s]
    pass_4o = v["per_seed_pass"][s]
    rows.append((s, drop_5_4, ci_5_4, drop_4o, ci_4o, delta, pass_5_4, pass_4o))

# Markdown
md_lines = [
    "| seed | drop (gpt-5.4) pp | 95 % CI (gpt-5.4) pp | drop (gpt-4o) pp | 95 % CI (gpt-4o) pp | Δ (4o − 5.4) pp | ≥ 3 pp pass |",
    "|------|------------------:|---------------------:|-----------------:|--------------------:|----------------:|-------------|",
]
for s, d5, c5, d4, c4, delta, p5, p4 in rows:
    verdict = f"5.4 {'✓' if p5 else '✗'} / 4o {'✓' if p4 else '✗'}"
    md_lines.append(
        f"| {s} | {fmt_drop(d5)} | {c5} | {fmt_drop(d4)} | {c4} | {fmt_drop(delta)} | {verdict} |"
    )
md = "\n".join(md_lines) + "\n"
md_path = f"{OUT_DIR}/{STEM}.md"
with open(md_path, "w") as f:
    f.write(md)
print(f"Saved: {md_path}")

# LaTeX (mirror)
tex_rows = "\n".join(
    f"{s} & ${fmt_drop(d5)}$ & {c5.replace('[', '$[').replace(']', ']$').replace('−', '-')} & ${fmt_drop(d4)}$ & {c4.replace('[', '$[').replace(']', ']$').replace('−', '-')} & ${fmt_drop(delta)}$ & {'PASS' if p5 else 'FAIL'} / {'PASS' if p4 else 'FAIL'} \\\\"
    for (s, d5, c5, d4, c4, delta, p5, p4) in rows
)
tex = dedent(r"""
\begin{table}[t]
\centering
\caption{Judge-swap replication at LR=1e-3 --- \texttt{gpt-4o} reproduces per-seed \texttt{QA\_I} drops within $\sim 1$~pp of \texttt{gpt-5.4}, ruling out judge calibration bias as the cause of the seed 300 reversal. Bootstrap 95\% CIs are item-paired, $n=133$.}
\label{tab:c1_judge_swap_replication}
\begin{tabular}{lcccccc}
\toprule
seed & drop (5.4) & 95\% CI (5.4) & drop (4o) & 95\% CI (4o) & $\Delta$ (4o$-$5.4) & pass ($\ge 3$~pp) \\
\midrule
""") + tex_rows + "\n" + dedent(r"""
\bottomrule
\end{tabular}
\end{table}
""")
tex_path = f"{OUT_DIR}/{STEM}.tex"
with open(tex_path, "w") as f:
    f.write(tex)
print(f"Saved: {tex_path}")
