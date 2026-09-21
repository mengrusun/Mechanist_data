"""Generate C4 figures: (a) amplifier α-sweep table (.md + .tex);
                        (b) amplifier vs oracle prompt-hint bar chart."""
import json, os, sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from paper_plot_style import plt, COLORS, save_fig  # noqa: E402
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# (a) α-sweep table — pythia-1b amplifier joint tolerance
# ---------------------------------------------------------------------------
alphas = ["1.5", "2.0", "3.0", "4.0"]
rows = []
for a in alphas:
    m4c_path = os.path.join(PROJECT_ROOT, f"runs/M4c_pythia-1b_alpha{a}_seed42/results.json")
    m4e_path = os.path.join(PROJECT_ROOT, f"runs/M4e_pythia-1b_alpha{a}_seed42/ppl.json")
    with open(m4c_path) as f:
        m4c = json.load(f)
    with open(m4e_path) as f:
        m4e = json.load(f)
    net_imp_pb = m4c["per_frame"]["personal_belief"]["net_improvement"]
    net_imp_ab = m4c["per_frame"]["attributed_belief"]["net_improvement"]
    wk_delta = m4c["per_frame"]["world_knowledge"]["amplified_accuracy"] - m4c["per_frame"]["world_knowledge"]["baseline_accuracy"]
    ppl_ratio = m4e["ppl_ratio"]
    passes_ppl = ppl_ratio <= 1.05
    meets_joint = (net_imp_pb > 0) and (net_imp_ab > 0) and (abs(wk_delta) <= 0.05) and passes_ppl
    rows.append({
        "alpha": a,
        "net_imp_pb": net_imp_pb,
        "net_imp_ab": net_imp_ab,
        "wk_delta_acc": wk_delta,
        "pile_ppl_ratio": ppl_ratio,
        "passes_ppl": passes_ppl,
        "meets_joint": meets_joint,
    })

# Markdown table
md = "| α | net_imp(PB) | net_imp(AB) | WK Δacc | Pile PPL ratio | Passes ≤ 1.05× | Meets joint tolerance |\n"
md += "|---:|---:|---:|---:|---:|:---:|:---:|\n"
for r in rows:
    md += (f"| {r['alpha']} | {r['net_imp_pb']:+d} | {r['net_imp_ab']:+d} "
           f"| {r['wk_delta_acc']:+.4f} | {r['pile_ppl_ratio']:.4f}× "
           f"| {'✓' if r['passes_ppl'] else '✗'} "
           f"| {'✓ (best joint α)' if r['meets_joint'] and r['alpha'] == '2.0' else ('✓' if r['meets_joint'] else '✗')} |\n")

with open(os.path.join(OUT_DIR, "c4_alpha_sweep_table.md"), "w") as f:
    f.write(md)

# LaTeX table
tex = r"""\begin{table}[t]
\centering
\caption{Amplifier $\alpha$-sweep on pythia-1b: net item-level improvement on belief\_holdout (2569 items), world-knowledge $\Delta\mathrm{acc}$, and Pile PPL ratio, per $\alpha \in \{1.5, 2.0, 3.0, 4.0\}$. Joint tolerance requires $\mathrm{net\_imp} > 0$ on both belief frames, $|\Delta\mathrm{acc}_{\mathrm{WK}}| \le 0.05$, and $\mathrm{PPL~ratio} \le 1.05\times$; best joint $\alpha = 2.0$.}
\label{tab:c4_alpha_sweep}
\begin{tabular}{rrrrrcc}
\toprule
$\alpha$ & net\_imp(PB) & net\_imp(AB) & WK $\Delta$acc & Pile PPL ratio & Passes $\le 1.05\times$ & Meets joint tol. \\
\midrule
"""
for r in rows:
    check_ppl = r"$\checkmark$" if r["passes_ppl"] else r"$\times$"
    if r["meets_joint"] and r["alpha"] == "2.0":
        check_joint = r"$\checkmark$ (best)"
    elif r["meets_joint"]:
        check_joint = r"$\checkmark$"
    else:
        check_joint = r"$\times$"
    tex += (f"{r['alpha']} & {r['net_imp_pb']:+d} & {r['net_imp_ab']:+d} "
            f"& {r['wk_delta_acc']:+.4f} & {r['pile_ppl_ratio']:.4f}$\\times$ "
            f"& {check_ppl} & {check_joint} \\\\\n")
tex += r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}" + "\n"
with open(os.path.join(OUT_DIR, "c4_alpha_sweep_table.tex"), "w") as f:
    f.write(tex)

# ---------------------------------------------------------------------------
# (b) amplifier vs oracle prompt-hint — bar chart (α=2.0, best joint)
# ---------------------------------------------------------------------------
with open(os.path.join(PROJECT_ROOT, "runs/M4c_pythia-1b_alpha2.0_seed42/results.json")) as f:
    amp = json.load(f)
with open(os.path.join(PROJECT_ROOT, "runs/M4d_pythia-1b/oracle_prompt.json")) as f:
    oracle = json.load(f)

frames = ["world_knowledge", "personal_belief", "attributed_belief"]
frame_labels = {"world_knowledge": "world knowledge",
                "personal_belief": "personal belief",
                "attributed_belief": "attributed belief"}
amp_ni = [amp["per_frame"][fr]["net_improvement"] for fr in frames]
orc_ni = [oracle["per_frame"][fr]["net_improvement"] for fr in frames]

fig, ax = plt.subplots(figsize=(6.5, 3.5))
x = np.arange(len(frames))
bar_w = 0.35
b1 = ax.bar(x - bar_w / 2, amp_ni, bar_w, color=COLORS[3],
            label="amplifier (α = 2.0, best joint)", edgecolor="white", linewidth=0.4)
b2 = ax.bar(x + bar_w / 2, orc_ni, bar_w, color=COLORS[0],
            label="oracle prompt-hint baseline", edgecolor="white", linewidth=0.4)
ax.axhline(0.0, color="grey", linewidth=0.6)

for i, v in enumerate(amp_ni):
    ax.text(i - bar_w / 2, v + (6 if v >= 0 else -12),
            f"{v:+d}", ha="center", va="bottom", fontsize=8, color=COLORS[3])
for i, v in enumerate(orc_ni):
    ax.text(i + bar_w / 2, v + (6 if v >= 0 else -12),
            f"{v:+d}", ha="center", va="bottom" if v >= 0 else "top", fontsize=8, color=COLORS[0])

# Δ annotation
for i, (a, o) in enumerate(zip(amp_ni, orc_ni)):
    delta = a - o
    if abs(delta) < 5:
        continue
    y = max(a, o) + 22
    ax.text(i, y, f"Δ = {delta:+d}", ha="center", va="bottom", fontsize=7,
            color="tab:red", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([frame_labels[fr] for fr in frames])
ax.set_ylabel("net item-level improvement on belief_holdout (2569 items)")
ax.set_ylim(min(min(amp_ni), min(orc_ni)) - 30, max(max(amp_ni), max(orc_ni)) + 55)
ax.legend(frameon=False, loc="upper left", fontsize=8)

save_fig(fig, "c4_amplifier_vs_oracle", formats=("pdf", "png"), out_dir=OUT_DIR)
plt.close(fig)

# ---------------------------------------------------------------------------
# INDEX.json
# ---------------------------------------------------------------------------
index = {
    "claim_id": "C4",
    "claim_title": "Dynamic Controllability — amplifier joint tolerance sweep on pythia-1b",
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "figures": [
        {
            "id": "c4_alpha_sweep_table",
            "type": "table",
            "caption": (
                "Amplifier α-sweep on pythia-1b: net item-level improvement on belief_holdout (2569 items), "
                "world-knowledge Δacc, and Pile PPL ratio, per α ∈ {1.5, 2.0, 3.0, 4.0}. Joint tolerance "
                "(net_imp > 0 on belief frames AND |WK Δacc| ≤ 0.05 AND Pile PPL ≤ 1.05× clean) is satisfied "
                "at α = 1.5 and α = 2.0; α = 3.0 / 4.0 give higher belief gain but fail Pile PPL. Best joint "
                "α = 2.0."
            ),
            "png": None,
            "pdf": None,
            "md": "figures/C4/c4_alpha_sweep_table.md",
            "tex": "figures/C4/c4_alpha_sweep_table.tex",
            "source_data": "runs/M4c_pythia-1b_alpha*_seed42/results.json (net_imp per α) + runs/M4e_pythia-1b_alpha*_seed42/ppl.json (Pile PPL)",
            "status": "ok",
        },
        {
            "id": "c4_amplifier_vs_oracle",
            "type": "bar",
            "caption": (
                "Amplifier (α = 2.0, best joint) vs. oracle prompt-hint baseline — net item-level improvement "
                "per frame on belief_holdout. Amplifier delivers positive net gain on both belief frames "
                "(PB +138, AB +165) with WK Δacc = 0.0000; oracle prompt-hint HURTS personal_belief (−18) while "
                "modestly helping attributed_belief (+64) and world_knowledge (+13). Amplifier beats oracle by "
                "Δ = +156 on PB and Δ = +101 on AB at the joint-tolerance-satisfying α."
            ),
            "png": "figures/C4/c4_amplifier_vs_oracle.png",
            "pdf": "figures/C4/c4_amplifier_vs_oracle.pdf",
            "md": None,
            "tex": None,
            "source_data": "runs/M4c_pythia-1b_alpha2.0_seed42/results.json + runs/M4d_pythia-1b/oracle_prompt.json",
            "status": "ok",
        },
    ],
    "skipped": [],
}
with open(os.path.join(OUT_DIR, "INDEX.json"), "w") as f:
    json.dump(index, f, indent=2)
print("C4: 2/2 figures generated")
