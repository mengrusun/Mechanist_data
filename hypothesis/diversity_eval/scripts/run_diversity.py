"""End-to-end SPECTER2 diversity analysis for any subtopic.

The two inputs are ordered claim pools: one produced with a knowledge graph
and one without it. Each pool is embedded independently, then the prefix
mean distance to centroid is calculated and plotted.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load_config(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return config


def config_path(value: str | None, config_dir: Path, name: str) -> Path:
    if not value:
        raise SystemExit(f"Missing {name}; fill it in config.json")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = config_dir / path
    if not path.exists():
        raise SystemExit(f"{name} does not exist: {path}")
    return path


def embed(
    claims: Path,
    output: Path,
    config_path_value: Path,
    config: dict,
    reuse: bool,
) -> None:
    if reuse and output.exists():
        print(f"[reuse] {output}")
        return

    command = [
        sys.executable,
        str(HERE / "embed_specter2.py"),
        "--input", str(claims),
        "--out", str(output),
        "--config", str(config_path_value),
    ]
    if config.get("no_adapter", False):
        command.append("--no-adapter")
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SPECTER2 embeddings and plot prefix diversity for one subtopic."
    )
    parser.add_argument("--kg-claims", required=True, help="ordered claims JSON or claim directory with knowledge graph")
    parser.add_argument("--ablation-claims", required=True, help="ordered claims JSON or claim directory without knowledge graph")
    parser.add_argument("--outdir", required=True, help="directory for embeddings and results")
    parser.add_argument("--config", default=str(HERE / "config.json"))
    parser.add_argument("--name", default=None, help="result basename; defaults to config output_name")
    parser.add_argument("--ks", default=None, help="comma-separated prefix sizes; overrides config")
    parser.add_argument("--reuse-embeddings", action="store_true")
    args = parser.parse_args()

    config_file = Path(args.config).expanduser().resolve()
    config = load_config(config_file)
    config_dir = config_file.parent
    config_path(config.get("specter2_model"), config_dir, "specter2_model")
    if not config.get("no_adapter", False):
        config_path(config.get("specter2_adapter"), config_dir, "specter2_adapter")

    kg_claims = Path(args.kg_claims).expanduser().resolve()
    ablation_claims = Path(args.ablation_claims).expanduser().resolve()
    for path in (kg_claims, ablation_claims):
        if not path.is_file() and not path.is_dir():
            raise SystemExit(f"claims input does not exist: {path}")

    outdir = Path(args.outdir).expanduser().resolve()
    embeddings_dir = outdir / "embeddings"
    results_dir = outdir / "results"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    name = args.name or config.get("output_name", "prefix_ke_absolute_specter2")
    ks = args.ks or ",".join(str(value) for value in config.get(
        "ks", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140]
    ))
    kg_embedding = embeddings_dir / f"specter2_{name}_with_kg.npz"
    ablation_embedding = embeddings_dir / f"specter2_{name}_without_kg.npz"

    print("=== embedding with knowledge graph ===")
    embed(kg_claims, kg_embedding, config_file, config, args.reuse_embeddings)
    print("=== embedding without knowledge graph ===")
    embed(ablation_claims, ablation_embedding, config_file, config, args.reuse_embeddings)

    plot_command = [
        sys.executable,
        str(HERE / "plot_prefix_ke_absolute.py"),
        "--emb", f"with_kg={kg_embedding}",
        "--emb", f"without_kg={ablation_embedding}",
        "--ks", ks,
        "--outdir", str(results_dir),
        "--name", name,
    ]
    if config.get("title"):
        plot_command.extend(["--title", str(config["title"])])
    subprocess.run(plot_command, check=True)
    print(f"done: {results_dir / (name + '.png')}")


if __name__ == "__main__":
    main()
