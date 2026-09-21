"""Plot probing R2 across layers for each anchor position."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe_json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metric", default="r2", choices=["r2", "spearman"])
    args = ap.parse_args()

    with open(args.probe_json) as f:
        r = json.load(f)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in r["positions"]:
        y = np.array(r[args.metric][p])
        ax.plot(y, label=p, linewidth=2)
    ax.set_xlabel("layer index")
    ax.set_ylabel(args.metric.upper())
    ax.set_title(f"[{r['tag']}] {args.metric.upper()} across layers  (N={r['N']})")
    ax.axhline(0, color="k", linestyle=":", alpha=0.4)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"[done] wrote {args.out}")


if __name__ == "__main__":
    main()
