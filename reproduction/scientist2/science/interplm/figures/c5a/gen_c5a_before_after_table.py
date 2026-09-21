"""
c5a_before_after_table: PR-AUC on annotation-filling probes, before/after iter-1 probe-code fix.
Data source: runs/iteration_round_1/m5_c5a_logreg_fix/wilcoxon.json (post-fix)
              refine-logs/EXPERIMENT_RESULTS.md#M5 (pre-fix baseline)
"""
import json
import os
from textwrap import dedent

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(OUT_DIR, '..', '..'))

# Post-fix numbers from wilcoxon.json
POST_PATH = os.path.join(PROJECT_ROOT, 'runs', 'iteration_round_1',
                         'm5_c5a_logreg_fix', 'wilcoxon.json')
with open(POST_PATH) as f:
    post = json.load(f)

post_n = post['n_concepts']
post_sae = post['mean_sae']
post_neu = post['mean_neuron']
post_p = post['pvalue']
post_w = post['statistic']

# Pre-fix (baseline) from EXPERIMENT_RESULTS.md M5 table
pre_n = 30
pre_sae = 0.5907
pre_neu = 0.5906
pre_p = 0.191
pre_w = 276.0

# --- Markdown ---
md = dedent(f"""
| Stage | n_concepts | mean PR-AUC (SAE) | mean PR-AUC (neurons) | Paired-Wilcoxon W | one-sided p (SAE > Neu) |
|---|---|---|---|---|---|
| Before fix (baseline) | {pre_n} | {pre_sae:.4f} | {pre_neu:.4f} | {pre_w:.1f} | {pre_p:.3f} |
| After fix (iter-1) | {post_n} | {post_sae:.4f} | {post_neu:.4f} | {post_w:.1f} | {post_p:.3f} |

**Interpretation.** The fair-test null holds both before and after the probe-code fix (well-converged log-loss SGD, `class_weight='balanced'`, tol-early-stop). Post-fix, mean PR-AUC(SAE) = {post_sae:.4f} is actually below mean PR-AUC(neurons) = {post_neu:.4f}, with paired-Wilcoxon p = {post_p:.3f}. Credible negative: the SAE code at layer 9 does not carry more annotation-decodable information than the raw residual.
""").lstrip()

with open(f"{OUT_DIR}/c5a_before_after_table.md", "w") as f:
    f.write(md)
print(f"Saved: {OUT_DIR}/c5a_before_after_table.md")

# --- LaTeX ---
tex = dedent(rf"""
\begin{{table}}[t]
\centering
\caption{{Per-arm PR-AUC on Swiss-Prot annotation-filling probes before and after the iter-1 probe-code fix. The fair-test null holds in both regimes: post-fix, paired-Wilcoxon (one-sided) yields $W={post_w:.1f}$, $p={post_p:.3f}$ on $n={post_n}$ concepts. Credible negative---the SAE code at layer~9 does not carry more annotation-decodable information than the raw residual stream.}}
\label{{tab:c5a_before_after}}
\begin{{tabular}}{{lccccc}}
\toprule
Stage & $n$ & PR-AUC (SAE) & PR-AUC (Neu) & $W$ & $p$ \\
\midrule
Before fix (baseline)  & {pre_n}  & {pre_sae:.4f} & {pre_neu:.4f} & {pre_w:.1f} & {pre_p:.3f} \\
After fix (iter-1)     & {post_n} & {post_sae:.4f} & {post_neu:.4f} & {post_w:.1f} & {post_p:.3f} \\
\bottomrule
\end{{tabular}}
\end{{table}}
""").lstrip()

with open(f"{OUT_DIR}/c5a_before_after_table.tex", "w") as f:
    f.write(tex)
print(f"Saved: {OUT_DIR}/c5a_before_after_table.tex")
