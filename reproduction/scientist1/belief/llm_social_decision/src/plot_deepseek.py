"""Plot DeepSeek intervention curves + combined comparison."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT/"results"
FIG = ROOT/"figures"; FIG.mkdir(exist_ok=True)

intv = json.load(open(RES/"intervention_deepseek.json"))
VARS = ["gender","age","instruction","meeting"]
for kind in ["raw", "pure"]:
    conds = [c for c in intv["conditions"] if c["kind"]==kind]
    alphas = sorted(set(c["alpha"] for c in conds))
    fig, axs = plt.subplots(1, 4, figsize=(20,4), sharey=True)
    for i, tv in enumerate(VARS):
        ax = axs[i]
        for target_v in VARS:
            ys, xs = [], []
            for a in alphas:
                m = [c for c in conds if c["variable"]==tv and c["alpha"]==a]
                if not m: continue
                xs.append(a); ys.append(m[0]["summary"][target_v])
            ax.plot(xs, ys, marker='.', label=f"effect_on_{target_v}")
        for target_v in VARS:
            ax.axhline(intv["baseline_summary"][target_v], ls=":", alpha=0.3)
        ax.set_title(f"inject {kind} {tv}")
        ax.set_xlabel(r"$\alpha$")
        if i==0: ax.set_ylabel("effect on E[transfer]")
        ax.axvline(0, c="gray", ls=":")
        ax.grid(alpha=.3)
    axs[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle(f"DeepSeek: {kind} directions, layer {intv['target_layer']}")
    plt.tight_layout()
    plt.savefig(FIG/f"intervention_{kind}_deepseek.png", dpi=140)
    print("saved", FIG/f"intervention_{kind}_deepseek.png")
