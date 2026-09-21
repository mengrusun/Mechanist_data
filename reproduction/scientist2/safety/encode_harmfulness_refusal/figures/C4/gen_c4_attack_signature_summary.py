"""C4 table — per-family ASR + failed-subset specificity summary.

Data: results/m4/claim4_verdict.json
Output: figures/C4/c4_attack_signature_summary.{md,tex}
"""
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))

d = json.load(open(os.path.join(PROJECT_ROOT, "results/m4/claim4_verdict.json")))
families = d["families_evaluated"]
per_family = d["per_family"]


def fmt_num(x, digits=2):
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and math.isnan(x):
            return "n/a (empty subset)"
    except Exception:
        pass
    return f"{x:.{digits}f}"


rows = []
for fam in families:
    r = per_family[fam]
    rows.append({
        "family": fam,
        "n_attempts": r.get("n_attacked", "—"),
        "asr": f"{r.get('asr', 0.0)*100:.1f}% ({r.get('n_successful', 0)}/{r.get('n_attacked', 0)})",
        "delta_r_success": fmt_num(r.get("delta_r_successful_mean"), 3),
        "abs_delta_r_failed": fmt_num(abs(r.get("delta_r_failed_mean") or 0), 3),
        "eps_null_r": fmt_num(r.get("eps_null_r"), 3),
        "verdict": r.get("verdict", "—"),
    })

# ---- Markdown ----
md = (
    "| Attack family | N attempts | ASR | Δr (successful subset) | |Δr| (failed subset) | ε_null (r) | Verdict |\n"
    "|---|---|---|---|---|---|---|\n"
)
for r in rows:
    md += (
        f"| **{r['family']}** | {r['n_attempts']} | {r['asr']} | {r['delta_r_success']} "
        f"| {r['abs_delta_r_failed']} | {r['eps_null_r']} | {r['verdict']} |\n"
    )
with open(os.path.join(HERE, "c4_attack_signature_summary.md"), "w") as f:
    f.write(md)

# ---- LaTeX ----
tex = r"""\begin{table}[t]
\centering
\caption{C4 — per-family attack success rate + failed-subset specificity. Both GCG (5 Zou-2023 published transferable suffixes) and PAP (5 Zeng-2024 published-style templates) yield ASR=0/500 on 100 held-out AdvBench behaviors; $\Delta$ metrics are untestable on an empty successful subset. Failed-subset $|\Delta_r|$ is within $\epsilon_{\mathrm{null}}=1.69$ for GCG and marginally over for PAP.}
\label{tab:c4_attack_signature}
\begin{tabular}{lllllll}
\toprule
Attack family & N attempts & ASR & $\Delta_r$ (success) & $|\Delta_r|$ (failed) & $\epsilon_{\mathrm{null}}(r)$ & Verdict \\
\midrule
"""
for r in rows:
    tex += (
        f"{r['family']} & {r['n_attempts']} & {r['asr'].replace('%', r'\%')} & "
        f"{r['delta_r_success']} & {r['abs_delta_r_failed']} & {r['eps_null_r']} & "
        f"{r['verdict'].replace('_', r'\_')} \\\\\n"
    )
tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
with open(os.path.join(HERE, "c4_attack_signature_summary.tex"), "w") as f:
    f.write(tex)
