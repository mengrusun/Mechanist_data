"""Plot prefix diversity for a knowledge-graph / no-knowledge-graph pair.

The diversity value at k is the mean Euclidean distance of the first k
embeddings to their own centroid. This is the metric used by the supplied
reference figure. The script also writes the values used for the plot as CSV
and JSON, so the figure is reproducible and auditable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


SERIES_STYLE = {
    "with_kg": {"color": "#2E7D5B", "marker": "s", "label": "With knowledge graph"},
    "without_kg": {"color": "#7B7B7B", "marker": "^", "label": "No knowledge graph"},
}
FALLBACK_COLORS = ["#4E79B8", "#A8A8A8", "#5B8FD6", "#92BFA3"]
FALLBACK_MARKERS = ["o", "D", "v", "P"]


def numeric_key(cid: str) -> tuple[int, str]:
    match = re.match(r"^(\d+)", cid)
    return (int(match.group(1)) if match else 10**9, cid)


def prefix_mean_distance(embeddings: np.ndarray, ks: list[int]) -> list[dict[str, float]]:
    rows = []
    for k in ks:
        if k > len(embeddings):
            continue
        subset = embeddings[:k]
        center = subset.mean(axis=0)
        distances = np.linalg.norm(subset - center, axis=1)
        rows.append({"k": k, "mean_distance": float(distances.mean())})
    return rows


def load_rows(path: str, ks: list[int]) -> tuple[list[dict[str, float]], dict]:
    with np.load(path, allow_pickle=True) as data:
        if "embeddings" not in data or "ids" not in data:
            raise ValueError(f"{path}: expected embeddings and ids arrays")
        embeddings = data["embeddings"].astype(np.float64)
        ids = [str(value) for value in data["ids"]]
        meta = json.loads(str(data["meta"])) if "meta" in data else {}

    encoder = str(meta.get("encoder", "")).lower()
    if "specter2" not in encoder:
        raise ValueError(f"{path}: expected SPECTER2 embeddings, got {meta.get('encoder')!r}")
    order = sorted(range(len(ids)), key=lambda index: numeric_key(ids[index]))
    rows = prefix_mean_distance(embeddings[order], ks)
    if not rows:
        raise ValueError(f"{path}: no requested k is <= the number of embeddings ({len(ids)})")
    return rows, meta


def style_for(label: str, index: int) -> dict[str, str]:
    key = label.strip().lower().replace("-", "_").replace(" ", "_")
    if key in {"kg", "with_kg", "knowledge_graph", "with_knowledge_graph"}:
        return SERIES_STYLE["with_kg"]
    if key in {"ablation", "no_kg", "without_kg", "no_knowledge_graph", "without_knowledge_graph"}:
        return SERIES_STYLE["without_kg"]
    return {
        "color": FALLBACK_COLORS[index % len(FALLBACK_COLORS)],
        "marker": FALLBACK_MARKERS[index % len(FALLBACK_MARKERS)],
        "label": label,
    }


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7,
        "axes.titlesize": 7,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.linewidth": 0.7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.dpi": 600,
    })


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SPECTER2 prefix diversity for one kg/no-kg claim pair."
    )
    parser.add_argument(
        "--emb", action="append", required=True, metavar="LABEL=PATH",
        help="embedding file; repeat for with_kg=... and without_kg=...",
    )
    parser.add_argument("--ks", default="10,20,30,40,50,60,70,80,90,100,110,120,130,140")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--name", default="prefix_ke_absolute_specter2")
    parser.add_argument("--title", default=None)
    args = parser.parse_args()

    ks = [int(value) for value in args.ks.split(",") if value.strip()]
    runs: list[tuple[str, list[dict[str, float]], dict]] = []
    for spec in args.emb:
        label, separator, path = spec.partition("=")
        if not separator or not label or not path:
            parser.error(f"--emb needs LABEL=PATH, got {spec!r}")
        rows, meta = load_rows(path, ks)
        runs.append((label, rows, meta))
    if not runs:
        parser.error("at least one --emb is required")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    apply_style()

    fig, ax = plt.subplots(figsize=(6.2, 4.5))
    records = []
    for index, (label, rows, _meta) in enumerate(runs):
        style = style_for(label, index)
        xs = [row["k"] for row in rows]
        ys = [row["mean_distance"] for row in rows]
        ax.plot(
            xs, ys, linestyle=(0, (1.2, 1.8)), color=style["color"], lw=1.8,
            marker=style["marker"], ms=6.0, mfc=style["color"], mec="white",
            mew=0.45, zorder=3,
        )
        ax.text(xs[-1] + 1.8, ys[-1], style["label"], color=style["color"],
                fontsize=6.5, va="center", ha="left")
        records.extend({"series": label, **row} for row in rows)

    reference_ks = sorted({row["k"] for _label, rows, _meta in runs for row in rows})
    ax.set_xticks(reference_ks)
    ax.set_xlabel("number of claims")
    ax.set_ylabel("mean distance to centroid")
    ax.grid(axis="y", color="#E7E7E7", linewidth=0.55, zorder=0)
    ax.grid(axis="x", color="#F0F0F0", linewidth=0.45, zorder=0)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")
    ax.tick_params(length=2.2, width=0.6, color="#333333", pad=2)
    ax.tick_params(axis="both", labelcolor="#222222")
    ax.margins(x=0.10)
    if args.title:
        ax.set_title(args.title)
    fig.tight_layout()

    png_path = outdir / f"{args.name}.png"
    fig.savefig(png_path, bbox_inches="tight", dpi=600)
    plt.close(fig)

    csv_path = outdir / f"{args.name}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["series", "k", "mean_distance"])
        writer.writeheader()
        writer.writerows(records)

    json_path = outdir / f"{args.name}.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump({"metric": "mean_distance_to_prefix_centroid", "rows": records},
                  handle, indent=2)

    print(f"wrote {png_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
