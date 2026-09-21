#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild `summary_hypothesis_quality.csv` from the raw LLM-judge scores.

Source
    <evaluation-run>/
        result/test_group_1/ result/test_group_2/ result/test_group_3/

    Three INDEPENDENT judge samplings of the SAME 240 hypotheses
    (3 systems x 8 subtopics x 10 claims). Verified: the claim key sets are
    identical across the three runs, and the scores differ between them, so
    they are repeats rather than copies.

Aggregation
    Per hypothesis, the three repeats are averaged first — this averages out
    judge noise without inflating the sample. The bootstrap then resamples the
    240 hypotheses, so `n_hypotheses` counts hypotheses, not judge calls.

Metrics (raw scores are 0-10 integers; the panel is on a 0-100 scale)
    Novelty      = novelty.score
    Impact       = mean(impact.significance.score, impact.reach.score)
    Testability  = mean(dimension_3_testability.{clarity,feasibility}.score)

Usage:  python3 make_summary.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
# This figure folder lives inside the eval run it plots, so resolve the run
# root relative to the script: <run>/figure/<panel>/data/make_summary.py
RESULT_ROOT = HERE.parents[2]
RUNS = ["test_group_1", "test_group_2", "test_group_3"]
OUT = HERE / "summary_hypothesis_quality.csv"

# result directory name -> the system label printed on the panel
SYSTEMS = {
    "cc-opus4.8": "Claude Code",
    "v2-opus4.8": "AI-Scientist",
    "mechanist-opus4.8-gpt5.4": "Mechanist",
}
METRICS = ["Novelty", "Impact", "Testability"]
SCALE = 10.0                     # 0-10 judge scale -> 0-100 panel scale
N_BOOT = 10000
SEED = 0


def dig(data, keys):
    for k in keys:
        if not isinstance(data, dict) or k not in data:
            return None
        data = data[k]
    return data if isinstance(data, (int, float)) and not isinstance(data, bool) else None


def read_claim(claim_dir: Path) -> dict[str, float | None]:
    """The three metric values for one claim in one run, or None where missing."""
    def load(name):
        p = claim_dir / name
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            return None

    nov, imp, tst = load("score_novelty.json"), load("score_impact.json"), load("score_testability.json")

    def pair(doc, a, b):
        """Mean of two sub-scores; falls back to whichever one is present."""
        vals = [v for v in (dig(doc, a), dig(doc, b)) if v is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "Novelty": dig(nov, ("novelty", "score")),
        "Impact": pair(imp, ("impact", "significance", "score"), ("impact", "reach", "score")),
        "Testability": pair(tst, ("dimension_3_testability", "clarity", "score"),
                                 ("dimension_3_testability", "feasibility", "score")),
    }


def collect() -> dict[tuple[str, str], np.ndarray]:
    """{(system, metric): per-hypothesis scores, averaged over the three runs}"""
    # (system, claim key) -> metric -> [one value per run]
    acc: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for run in RUNS:
        run_root = RESULT_ROOT / run
        for claim_json in run_root.rglob("claim.json"):
            rel = claim_json.parent.relative_to(run_root).parts
            system = SYSTEMS.get(rel[0])
            if system is None:
                continue
            key = (system, "/".join(rel[1:]))
            for metric, value in read_claim(claim_json.parent).items():
                if value is not None:
                    acc[key][metric].append(value)

    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (system, _), per_metric in acc.items():
        for metric in METRICS:
            vals = per_metric.get(metric)
            if vals:
                out[(system, metric)].append(sum(vals) / len(vals))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def bootstrap_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    data = collect()
    rng = np.random.default_rng(SEED)
    rows = []
    for system in SYSTEMS.values():
        for metric in METRICS:
            x = data[(system, metric)] * SCALE
            lo, hi = bootstrap_ci(x, rng)
            rows.append([system, metric, len(x), round(float(x.mean()), 2),
                         round(lo, 2), round(hi, 2)])

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["system", "metric", "n_hypotheses", "mean_score", "ci95_low", "ci95_high"])
        w.writerows(rows)

    print(f"wrote {OUT}")
    for r in rows:
        print(f"  {r[0]:14s} {r[1]:12s} n={r[2]:4d}  {r[3]:6.2f}  [{r[4]:.2f}, {r[5]:.2f}]")


if __name__ == "__main__":
    main()
