"""C2 figure: M2 acceptance table for the 5 admissible (model, target) pairs."""
import json
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

PAIRS = [
    ("pythia-410m", "personal"),
    ("pythia-1b", "personal"),
    ("pythia-1b", "attributed"),
    ("pythia-2.8b", "personal"),
    ("pythia-2.8b", "attributed"),
]


def load_accept(model, target):
    p = os.path.join(ROOT, "refine-logs/artifacts/hstar", model, f"H_{target}_acceptance.json")
    with open(p) as f:
        return json.load(f)


def fmt_heads(heads):
    return ", ".join(f"({l},{h})" for l, h in heads)


def check(x):
    return "PASS" if x else "FAIL"


rows = []
for m, t in PAIRS:
    d = load_accept(m, t)
    heads = fmt_heads(d["hstar_heads"])
    dr = d["drops"]
    rh = d["random_head"]
    crit = d["criteria"]
    fv = d.get("final_verdict", "")
    if isinstance(fv, dict):
        status = fv.get("localization_status", fv.get("status", "")) or "localized"
    else:
        status = str(fv) if fv else "localized"
    rows.append({
        "model": m,
        "target": t,
        "hstar_size": d["hstar_size"],
        "heads": heads,
        "d_target": dr["target"],
        "d_off_belief": dr["other_belief"],
        "d_wk": dr["world_knowledge"],
        "ppl_ratio": dr["ppl_ratio"],
        "threshold_2sigma": rh["threshold_2sigma"],
        "c2a": check(crit.get("C2a_drop_target_ge_0.30", crit.get("C2a", False))),
        "c2b": check(crit.get("C2b_drop_target_gt_meanplus2sigma", crit.get("C2b", False))),
        "c2c": check(crit.get("C2c_off_target_drops_le_0.10", crit.get("C2c", False))),
        "c2d": check(crit.get("C2d_ppl_ratio_le_1.05", crit.get("C2d", False))),
        "status": str(status).upper(),
    })


# Markdown
md_lines = [
    "| Model | Target | |H*| | Heads (layer, head) | Δ target | Δ off-belief | Δ WK | PPL ratio | mean_rh + 2σ | C2a | C2b | C2c | C2d | Status |",
    "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
]
for r in rows:
    md_lines.append(
        f"| {r['model']} | {r['target']} | {r['hstar_size']} | {r['heads']} | "
        f"{r['d_target']:.3f} | {r['d_off_belief']:+.3f} | {r['d_wk']:+.3f} | "
        f"{r['ppl_ratio']:.3f}× | {r['threshold_2sigma']:.3f} | "
        f"{r['c2a']} | {r['c2b']} | {r['c2c']} | {r['c2d']} | **{r['status']}** |"
    )
md = "\n".join(md_lines) + "\n"
with open(os.path.join(OUT_DIR, "c2_localization_table.md"), "w") as f:
    f.write(md)
print(f"Saved: {OUT_DIR}/c2_localization_table.md")

# LaTeX
tex_lines = [
    r"\begin{table}[t]",
    r"\centering",
    r"\caption{M2 final acceptance for the 5 admissible (model, target) pairs. Every pair LOCALIZED under all four verbatim criteria (target drop $\geq$ 0.30 AND $>$ mean $+$ 2$\sigma$ of 20 random-head controls; off-target and world\_knowledge drops $\leq$ 0.10; post-ablation PPL $\leq$ 1.05$\times$ clean). $|H^*| \in \{1, 2, 4\}$; the attributed circuit is often a single head.}",
    r"\label{tab:c2_localization}",
    r"\resizebox{\textwidth}{!}{",
    r"\begin{tabular}{llccccccccccccc}",
    r"\toprule",
    r"Model & Target & $|H^*|$ & Heads & $\Delta$ target & $\Delta$ off & $\Delta$ WK & PPL & $\mu_{rh}+2\sigma$ & C2a & C2b & C2c & C2d & Status \\",
    r"\midrule",
]
for r in rows:
    heads_tex = r["heads"].replace("_", r"\_")
    tex_lines.append(
        f"{r['model'].replace('_', '_')} & {r['target']} & {r['hstar_size']} & {heads_tex} & "
        f"{r['d_target']:.3f} & {r['d_off_belief']:+.3f} & {r['d_wk']:+.3f} & "
        f"{r['ppl_ratio']:.3f} & {r['threshold_2sigma']:.3f} & "
        f"{r['c2a']} & {r['c2b']} & {r['c2c']} & {r['c2d']} & \\textbf{{{r['status']}}} \\\\"
    )
tex_lines += [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}", ""]
with open(os.path.join(OUT_DIR, "c2_localization_table.tex"), "w") as f:
    f.write("\n".join(tex_lines))
print(f"Saved: {OUT_DIR}/c2_localization_table.tex")
