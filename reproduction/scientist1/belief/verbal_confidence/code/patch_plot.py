"""Bar-chart of mean patch deltas across positions for each experiment."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


POS_ORDER = [
    ("c_patch_pre",    "pre",     "at Answer:"),
    ("c_patch_ans",    "ans",     "last answer tok"),
    ("c_patch_nl",     "nl",      "newline"),
    ("c_patch_ans_nl", "ans+nl",  "cache region"),
    ("c_patch_post",   "post",    "at Confidence:"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsons", nargs="+", required=True,
                    help="list of results/patch_*.json  (label may be overridden with =)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # each json may be "path" or "path=label"
    specs = []
    for a in args.jsons:
        if "=" in a:
            p, lab = a.split("=", 1)
        else:
            p, lab = a, Path(a).stem
        specs.append((p, lab))

    fig, ax = plt.subplots(figsize=(9, 4.5))
    xs = np.arange(len(POS_ORDER))
    w = 0.8 / max(1, len(specs))

    for i, (p, lab) in enumerate(specs):
        with open(p) as f:
            data = json.load(f)
        rows = [r for r in data["rows"]
                if all(r.get(k) is not None for k, _, _ in POS_ORDER) and r["c_base"] is not None]
        target = np.mean([r["c_hi"] - r["c_base"] for r in rows])
        deltas = [np.mean([r[k] - r["c_base"] for r in rows]) for k, _, _ in POS_ORDER]
        ax.bar(xs + w * (i - (len(specs) - 1) / 2), deltas, width=w, label=f"{lab}  (target={target:+.1f})")

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{name}\n{note}" for _, name, note in POS_ORDER])
    ax.axhline(0, color="k", linewidth=0.7)
    ax.set_ylabel("mean Δ(confidence)  =  c_patch − c_base")
    ax.set_title("Causal patching: mean shift in verbalized confidence")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9)
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
