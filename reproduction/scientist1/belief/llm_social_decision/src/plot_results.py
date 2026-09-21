"""Plot: (1) linear-probe accuracy by layer, (2) intervention specificity heatmap."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/"results"
FIG = ROOT/"figures"
FIG.mkdir(exist_ok=True)

# probes
probe = json.load(open(RES/"probe_baseline_accs.json"))
plt.figure(figsize=(6,4))
for v, accs in probe.items():
    plt.plot(range(len(accs)), accs, label=v, lw=2, marker='.')
plt.xlabel("Layer index (0=embedding)")
plt.ylabel("Probe accuracy (5-fold CV)")
plt.title("Claim 1: linear probe accuracy on 1000 dictator trials")
plt.legend()
plt.axhline(0.5, ls=":", c="gray")
plt.grid(alpha=.3)
plt.tight_layout()
plt.savefig(FIG/"probe_accs.png", dpi=140)
print("saved", FIG/"probe_accs.png")

# intervention heatmap
intv = json.load(open(RES/"intervention_llama.json"))
VARS = ["gender","age","instruction","meeting"]

# For each kind, build a table: rows=injected variable, cols=alpha, values = effect on that var
for kind in ["raw", "pure"]:
    conds = [c for c in intv["conditions"] if c["kind"]==kind]
    alphas = sorted(set(c["alpha"] for c in conds))
    fig, axs = plt.subplots(1, 4, figsize=(20,4), sharey=True)
    for i, tv in enumerate(VARS):
        ax = axs[i]
        # For each alpha, plot the effect on every variable when injecting direction of tv.
        for target_v in VARS:
            ys = []
            xs = []
            for a in alphas:
                m = [c for c in conds if c["variable"]==tv and c["alpha"]==a]
                if not m: continue
                xs.append(a)
                ys.append(m[0]["summary"][target_v])
            ax.plot(xs, ys, marker='.', label=f"effect_on_{target_v}")
        # baseline reference
        for target_v in VARS:
            base = intv["baseline_summary"][target_v]
            ax.axhline(base, ls=":", alpha=0.3)
        ax.set_title(f"inject {kind} {tv}")
        ax.set_xlabel(r"$\alpha$")
        if i==0: ax.set_ylabel("effect on E[transfer]")
        ax.axvline(0, c="gray", ls=":")
        ax.grid(alpha=.3)
    axs[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"Claim 3&4: causal intervention with {kind} directions (layer {intv['target_layer']})")
    plt.tight_layout()
    plt.savefig(FIG/f"intervention_{kind}.png", dpi=140)
    print("saved", FIG/f"intervention_{kind}.png")

# Also produce a compact summary table
def specificity_table(kind, alpha=3.0):
    conds = [c for c in intv["conditions"] if c["kind"]==kind and c["alpha"]==alpha]
    rows = []
    for tv in VARS:
        m = [c for c in conds if c["variable"]==tv]
        if not m: continue
        s = m[0]["delta_summary_vs_base"]
        rows.append([tv, s["gender"], s["age"], s["instruction"], s["meeting"]])
    return rows

print("\nSpecificity delta-vs-baseline at alpha=+3, RAW directions:")
print(f"  {'inject':>12s}  {'d_gender':>9s} {'d_age':>7s} {'d_instr':>8s} {'d_meet':>7s}")
for r in specificity_table("raw", 3.0):
    print(f"  {r[0]:>12s}  {r[1]:+.3f}    {r[2]:+.3f}   {r[3]:+.3f}    {r[4]:+.3f}")

print("\nSpecificity delta-vs-baseline at alpha=+3, PURE directions:")
print(f"  {'inject':>12s}  {'d_gender':>9s} {'d_age':>7s} {'d_instr':>8s} {'d_meet':>7s}")
for r in specificity_table("pure", 3.0):
    print(f"  {r[0]:>12s}  {r[1]:+.3f}    {r[2]:+.3f}   {r[3]:+.3f}    {r[4]:+.3f}")
