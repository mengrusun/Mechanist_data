"""C4 figure: OOD summary table across both Pythia scales."""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

MODELS = ["pythia-1b", "pythia-2.8b"]

rows = []
for m in MODELS:
    with open(os.path.join(ROOT, "refine-logs/artifacts/controller", m, "M4_report.json")) as f:
        r = json.load(f)
    pt = r["per_task"]
    rows.append({
        "model": m,
        "alpha_p": r["alpha_personal"],
        "alpha_a": r["alpha_attributed"],
        "personal_pair": f"{pt['personal_belief']['acc_baseline_no_control']:.3f} → {pt['personal_belief']['acc_controller']:.3f}",
        "attributed_pair": f"{pt['attributed_belief']['acc_baseline_no_control']:.3f} → {pt['attributed_belief']['acc_controller']:.3f}",
        "wk_pair": f"{pt['world_knowledge']['acc_baseline_no_control']:.3f} → {pt['world_knowledge']['acc_controller']:.3f}",
        "ctrl_net": r["controller_vs_baseline"]["net_improvement"],
        "ph_net": r["prompt_hint_vs_baseline"]["net_improvement"],
        "frame_acc": r["frame_acc_ood"],
        "ppl_ratio": r["ppl_controller_ood"]["ppl"] / r["ppl_clean"]["ppl"],
    })

# Markdown
md_lines = [
    "| Model | α_p* | α_a* | OOD personal (baseline → controller) | OOD attributed (baseline → controller) | OOD WK (baseline → controller) | Controller net_impr | Prompt-hint net_impr | Frame classifier OOD acc | PPL ratio (controller / clean) |",
    "|---|---|---|---|---|---|---|---|---|---|",
]
for r in rows:
    md_lines.append(
        f"| {r['model']} | {r['alpha_p']} | {r['alpha_a']} | "
        f"{r['personal_pair']} | {r['attributed_pair']} | {r['wk_pair']} | "
        f"**{r['ctrl_net']:+d}** | {r['ph_net']:+d} | {r['frame_acc']:.4f} | {r['ppl_ratio']:.3f}× |"
    )
md = "\n".join(md_lines) + "\n"
with open(os.path.join(OUT_DIR, "c4_summary_table.md"), "w") as f:
    f.write(md)
print(f"Saved: {OUT_DIR}/c4_summary_table.md")

# LaTeX
tex_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{M4.3 OOD summary on \texttt{belief\_holdout}. Controller strictly beats prompt-hint on both Pythia scales; effect size differs (+151 on 1b vs +27 on 2.8b) — see caveat C4.2 (non-uniform gain across scale).}",
    r"\label{tab:c4_summary}",
    r"\resizebox{\textwidth}{!}{",
    r"\begin{tabular}{lcccccccccc}",
    r"\toprule",
    r"Model & $\alpha_p^*$ & $\alpha_a^*$ & OOD personal & OOD attributed & OOD WK & Ctrl net & PH net & Frame acc & PPL ratio \\",
    r" & & & (base $\to$ ctrl) & (base $\to$ ctrl) & (base $\to$ ctrl) & impr & impr & OOD & ctrl/clean \\",
    r"\midrule",
]
ARROW = chr(0x2192)
TO_TEX = "$\\to$"
for r in rows:
    pp = r["personal_pair"].replace(ARROW, TO_TEX)
    ap = r["attributed_pair"].replace(ARROW, TO_TEX)
    wp = r["wk_pair"].replace(ARROW, TO_TEX)
    tex_lines.append(
        f"{r['model']} & {r['alpha_p']} & {r['alpha_a']} & "
        f"{pp} & {ap} & {wp} & "
        f"\\textbf{{{r['ctrl_net']:+d}}} & {r['ph_net']:+d} & {r['frame_acc']:.4f} & {r['ppl_ratio']:.3f} \\\\"
    )
tex_lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
with open(os.path.join(OUT_DIR, "c4_summary_table.tex"), "w") as f:
    f.write("\n".join(tex_lines))
print(f"Saved: {OUT_DIR}/c4_summary_table.tex")
