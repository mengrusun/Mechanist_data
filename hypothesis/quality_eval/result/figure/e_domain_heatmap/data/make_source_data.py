#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild `source_data_hypotheses.csv` from the raw LLM-judge scores.

Source
    <evaluation-run>/
        result/test_group_1/ result/test_group_2/ result/test_group_3/

    Three INDEPENDENT judge samplings of the SAME 80 hypotheses per system
    (8 subtopics x 10 claims). The three repeats are averaged per hypothesis,
    so each output row is one hypothesis, not one judge call.

Columns (kept identical to the previous export so the panel script is unchanged)
    novelty / impact / testability  - 0-100 (raw 0-10 judge scores x10)
    composite_score                 - mean of those three
    high_value / data_source        - vestigial columns the panel does not read;
                                      emitted as False / "real" for compatibility.

Dimension definitions match panel f: Impact is the mean of its Significance and
Reach sub-scores, Testability the mean of Clarity and Feasibility.

Subtopics roll up to the four panel domains by their top-level category. Note
that Science now carries 10 hypotheses per system rather than 20: the
`science/mechanism` subtopic was dropped from the evaluation, leaving
`science/general`.

Usage:  python3 make_source_data.py
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
# This figure folder lives inside the eval run it plots, so resolve the run
# root relative to the script: <run>/figure/<panel>/data/make_source_data.py
RESULT_ROOT = HERE.parents[2]
RUNS = ["test_group_1", "test_group_2", "test_group_3"]
OUT = HERE / "source_data_hypotheses.csv"

SYSTEMS = {
    "cc-opus4.8": "Claude Code",
    "v2-opus4.8": "AI-Scientist",
    "mechanist-opus4.8-gpt5.4": "Mechanist",
}
DOMAINS = {"knowledge": "Knowledge", "language": "Language",
           "safety": "Safety", "science": "Science"}
SCALE = 10.0


def dig(data, keys):
    for k in keys:
        if not isinstance(data, dict) or k not in data:
            return None
        data = data[k]
    return data if isinstance(data, (int, float)) and not isinstance(data, bool) else None


def read_claim(claim_dir: Path) -> dict[str, float | None]:
    def load(name):
        try:
            return json.loads((claim_dir / name).read_text(encoding="utf-8"))
        except Exception:                                     # noqa: BLE001
            return None

    nov, imp, tst = load("score_novelty.json"), load("score_impact.json"), load("score_testability.json")

    def pair(doc, a, b):
        vals = [v for v in (dig(doc, a), dig(doc, b)) if v is not None]
        return sum(vals) / len(vals) if vals else None

    return {
        "novelty": dig(nov, ("novelty", "score")),
        "impact": pair(imp, ("impact", "significance", "score"), ("impact", "reach", "score")),
        "testability": pair(tst, ("dimension_3_testability", "clarity", "score"),
                                 ("dimension_3_testability", "feasibility", "score")),
    }


def main() -> None:
    # (system, domain, hypothesis_id) -> metric -> [one value per run]
    acc = defaultdict(lambda: defaultdict(list))
    for run in RUNS:
        run_root = RESULT_ROOT / run
        for claim_json in run_root.rglob("claim.json"):
            rel = claim_json.parent.relative_to(run_root).parts
            system, domain = SYSTEMS.get(rel[0]), DOMAINS.get(rel[1])
            if system is None or domain is None:
                continue
            key = (system, domain, rel[-1])
            for metric, value in read_claim(claim_json.parent).items():
                if value is not None:
                    acc[key][metric].append(value)

    rows = []
    for (system, domain, hid), per_metric in acc.items():
        vals = {m: sum(v) / len(v) * SCALE for m, v in per_metric.items()}
        if len(vals) < 3:                       # skip a claim missing a dimension
            continue
        composite = sum(vals.values()) / 3
        rows.append([domain, system, hid,
                     round(vals["novelty"], 2), round(vals["impact"], 2),
                     round(vals["testability"], 2), round(composite, 2), False, "real"])

    order = {d: i for i, d in enumerate(DOMAINS.values())}
    sord = {s: i for i, s in enumerate(SYSTEMS.values())}
    rows.sort(key=lambda r: (order[r[0]], sord[r[1]], r[2]))

    with OUT.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["domain", "system", "hypothesis_id", "novelty", "impact",
                    "testability", "composite_score", "high_value", "data_source"])
        w.writerows(rows)

    print(f"wrote {OUT}  ({len(rows)} rows)")
    cells = defaultdict(list)
    for r in rows:
        cells[(r[0], r[1])].append(r[6])
    for d in DOMAINS.values():
        line = "  ".join(f"{s}={sum(cells[(d, s)]) / len(cells[(d, s)]):5.2f}(n={len(cells[(d, s)])})"
                         for s in SYSTEMS.values())
        print(f"  {d:10s} {line}")


if __name__ == "__main__":
    main()
