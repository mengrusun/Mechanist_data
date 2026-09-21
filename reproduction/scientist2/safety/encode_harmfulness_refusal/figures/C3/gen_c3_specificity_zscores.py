"""C3 specificity table — direction-specificity z-scores at h-site (iter 1) and r-site (iter 2).

Data:
  runs/iteration_round_1/m3_extended_analysis.json
  runs/iteration_round_2/r_site_specificity_analysis.json

Output: figures/C3/c3_specificity_zscores.{md,tex}
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))

iter1 = json.load(open(os.path.join(PROJECT_ROOT, "runs/iteration_round_1/m3_extended_analysis.json")))
iter2 = json.load(open(os.path.join(PROJECT_ROOT, "runs/iteration_round_2/r_site_specificity_analysis.json")))

h_low = iter1["true_direction_vs_random_control"]["h_at_alpha_sigma_~1.87"]
h_high = iter1["true_direction_vs_random_control"]["h_at_alpha_sigma_~3.73"]
r_cross = iter1["true_direction_vs_random_control"]["r_refusal_ben_vs_random_direction"]
r_site_high = iter2["alpha_raw=2.0"]

rows = [
    {
        "direction": "true h",
        "site": r"h-site (layer 11, $t_{\text{final\_instr}}$)",
        "metric": "h-readout (harmful)",
        "true_effect": f"{h_low['true_h_readout_delta_from_baseline']:.3f}",
        "random_ctrl": f"{h_low['random_mean_delta']:.4f} $\\pm$ {h_low['random_std_delta']:.3f} (n={h_low['random_n']})",
        "z_score": f"{h_low['z_score_of_true_delta_wrt_random_dist']:.2f}",
        "alpha": r"$\alpha_\sigma \!\approx\! 1.87$",
    },
    {
        "direction": "true h",
        "site": r"h-site (layer 11, $t_{\text{final\_instr}}$)",
        "metric": "h-readout (harmful)",
        "true_effect": f"{h_high['true_h_readout_delta_from_baseline']:.3f}",
        "random_ctrl": f"{h_high['random_mean_delta']:.4f} $\\pm$ {h_high['random_std_delta']:.3f} (n={h_high['random_n']})",
        "z_score": f"{h_high['z_score_of_true_delta_wrt_random_dist']:.2f}",
        "alpha": r"$\alpha_\sigma \!\approx\! 3.73$",
    },
    {
        "direction": "true r",
        "site": r"h-site (cross-site check, iter 1)",
        "metric": "refusal (benign)",
        "true_effect": f"{r_cross['true_r_refusal_ben_delta_from_baseline']:.3f}",
        "random_ctrl": f"{r_cross['random_direction_refusal_ben_delta_mean']:.4f} $\\pm$ {r_cross['random_direction_refusal_ben_delta_std']:.3f} (n={r_cross['random_n']})",
        "z_score": f"{r_cross['z_score']:.2f}",
        "alpha": r"$\alpha_\sigma \!\approx\! 3.78$",
    },
    {
        "direction": "true r",
        "site": r"r-site (layer 13, $t_{\text{post\_instr}}$, iter 2 closure)",
        "metric": "refusal (benign)",
        "true_effect": f"{r_site_high['true_r_effect']['delta_refusal_ben']:.3f}",
        "random_ctrl": f"{r_site_high['random_r_site_refusal_ben']['delta_from_baseline_mean']:.4f} $\\pm$ {r_site_high['random_r_site_refusal_ben']['std']:.4f} (n={r_site_high['n_random_r_site']})",
        "z_score": f"{r_site_high['z_score_true_r_vs_random_r_site_refusal_ben']:.2f}",
        "alpha": r"$\alpha_\sigma \!\approx\! 3.78$",
    },
]

# ---- Markdown ----
md_lines = [
    "| Direction | Site | α (σ_proj) | Metric | Effect (true) | Random control (mean ± SD, n=30) | z vs random |",
    "|---|---|---|---|---|---|---|",
]
for r in rows:
    md_lines.append(
        f"| {r['direction']} | {r['site'].replace('$', '').replace('_{', '').replace('}', '').replace('text', '').replace('\\', '').replace('!approx!', '≈')} "
        f"| {r['alpha'].replace('$', '').replace('_{', '').replace('}', '').replace('\\', '').replace('!approx!', '≈').replace('sigma', 'σ')} "
        f"| {r['metric']} | {r['true_effect']} "
        f"| {r['random_ctrl'].replace('$\\\\pm$', '±').replace('$', '')} "
        f"| **{r['z_score']}** |"
    )
md = "\n".join(md_lines) + "\n"
with open(os.path.join(HERE, "c3_specificity_zscores.md"), "w") as f:
    f.write(md)

# ---- LaTeX ----
tex = r"""\begin{table}[t]
\centering
\caption{C3 — direction-specificity: z-scores of true-direction effects vs 30 matched-norm random-direction controls, at both h's site (iteration 1) and r's site (iteration 2). $z=128.31$ at r's site is the decisive result — the effect is direction-specific, not merely site-driven.}
\label{tab:c3_specificity}
\begin{tabular}{llllrlr}
\toprule
Direction & Site & $\alpha$ & Metric & Effect (true) & Random control (n=30) & $z$ vs random \\
\midrule
"""
for r in rows:
    tex += (
        f"{r['direction']} & {r['site']} & {r['alpha']} & {r['metric']} & "
        f"{r['true_effect']} & {r['random_ctrl']} & \\textbf{{{r['z_score']}}} \\\\\n"
    )
tex += r"""\bottomrule
\end{tabular}
\end{table}
"""
with open(os.path.join(HERE, "c3_specificity_zscores.tex"), "w") as f:
    f.write(tex)
