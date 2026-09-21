"""C1 Table figure: Top-5 site sweep + joint 5-site steering — logit-level
E[first_digit] max |span| across α ∈ [-16, +16], per seed and site vs. joint.
All single-site spans < 0.15; joint spans (0.14, 0.20, 0.32) — small distributed
effect below the 0.5 first-digit-units threshold. Grounds dissociation_generalizes
+ distributed_null verdicts and the paper's central negative finding.

Writes both `<STEM>.md` (for inline ledger embed) and `<STEM>.tex` (paper-write).
"""
from __future__ import annotations
import json
import os
from textwrap import dedent

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
STEM = "c1_top5_site_and_joint_sweep"

SEEDS = [42, 123, 2024]
SITES = ["E4L10", "E1L5", "E2L5", "E3L10", "E3L5"]


def compute_span(per_alpha):
    """max |E[first_digit] at α| - min value across the α grid."""
    vals = [rec["e_first_digit_mean"] for rec in per_alpha.values()]
    return max(vals) - min(vals)


def load_v3_span(site, seed):
    path = f"results/m5_v3/m5v3_expected_score_{site}_seed{seed}.json"
    with open(path) as f:
        d = json.load(f)["summary"]
    # The summary carries e_first_digit_span_neg16_to_pos16 directly; use it
    # if present, else recompute from per_alpha.
    span = d.get("e_first_digit_span_neg16_to_pos16")
    if span is None:
        span = compute_span(d["per_alpha"])
    return abs(span)


def load_joint_span(seed):
    path = f"results/m5_v3_joint/m5v3_joint_expected_score_seed{seed}.json"
    with open(path) as f:
        d = json.load(f)["summary"]
    span = d.get("e_first_digit_span_neg16_to_pos16")
    if span is None:
        span = compute_span(d["per_alpha"])
    return abs(span)


rows = []  # (label, per-seed spans, max_across_seeds)
for site in SITES:
    per_seed = [load_v3_span(site, s) for s in SEEDS]
    rows.append((f"single-site: {site}", per_seed, max(per_seed)))

joint = [load_joint_span(s) for s in SEEDS]
rows.append(("joint top-5 (MultiSiteSteeringHook)", joint, max(joint)))


def verdict(m):
    return "distributed_null (< 0.5)" if m < 0.5 else "signal"


# ---------- Markdown ----------
md_header = "| Intervention target | seed 42 | seed 123 | seed 2024 | max across seeds | verdict |"
md_sep = "|---|---:|---:|---:|---:|---|"
md_lines = [md_header, md_sep]
for label, spans, mx in rows:
    md_lines.append(
        f"| {label} | {spans[0]:.3f} | {spans[1]:.3f} | {spans[2]:.3f} | {mx:.3f} | {verdict(mx)} |"
    )
md_lines.append("")
md_lines.append(
    "*Numbers are max−min of E[first_digit] over α ∈ {−16, −8, −4, −1, 0, 1, 4, 8, 16} "
    "(3 seeds × 40 items × 9 α), Gemma-3-27B-pt on TriviaQA (rc.nocontext) validation.*"
)
md_text = "\n".join(md_lines) + "\n"
with open(f"{OUT_DIR}/{STEM}.md", "w") as f:
    f.write(md_text)
print(f"Saved: {OUT_DIR}/{STEM}.md")


# ---------- LaTeX ----------
def latex_row(label, spans, mx):
    v = "$<$ 0.5 (distributed\\_null)" if mx < 0.5 else "signal"
    label_tex = label.replace("_", r"\_")
    return (
        f"{label_tex} & {spans[0]:.3f} & {spans[1]:.3f} & {spans[2]:.3f} & "
        f"{mx:.3f} & {v} \\\\"
    )


tex = dedent(r"""
\begin{table}[t]
\centering
\caption{Top-5 cache-site sweep and joint 5-site steering.
Numbers are max$-$min of $\mathbb{E}[\text{first\_digit}]$ over
$\alpha \in \{{-16,-8,-4,-1,0,1,4,8,16\}}$ (3 seeds $\times$ 40 items).
All single-site spans are below $0.15$; the joint 5-site intervention shows
a small distributed effect (up to $0.32$), still below the $0.5$ threshold.
Gemma-3-27B-pt on TriviaQA (rc.nocontext) validation.}
\label{tab:c1_top5_site_and_joint_sweep}
\begin{tabular}{lccccl}
\toprule
Intervention target & seed 42 & seed 123 & seed 2024 & max & verdict \\
\midrule
""") + "\n".join(latex_row(l, s, m) for l, s, m in rows) + "\n" + dedent(r"""\bottomrule
\end{tabular}
\end{table}
""")

with open(f"{OUT_DIR}/{STEM}.tex", "w") as f:
    f.write(tex)
print(f"Saved: {OUT_DIR}/{STEM}.tex")
