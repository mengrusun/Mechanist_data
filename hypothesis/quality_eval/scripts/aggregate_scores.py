#!/usr/bin/env python3
"""Aggregate score JSON files into a Method-by-Topic mean-score table.

Supported result layouts are::

    <root>/<method>/<category>/<subtopic>/<claim>/score_*.json
    <root>/test_group_N/<method>/<category>/<subtopic>/<claim>/score_*.json

The optional ``test_group_N`` component identifies a repeated judge run. It is
excluded from method/topic labels, and repeats are pooled in the same cells.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

BASE_DIMENSIONS = ["Novelty", "Significance", "Reach", "Clarity", "Feasibility"]
DIM_SOURCE = {
    "Novelty": ("score_novelty.json", ("novelty", "score")),
    "Significance": ("score_impact.json", ("impact", "significance", "score")),
    "Reach": ("score_impact.json", ("impact", "reach", "score")),
    "Clarity": ("score_testability.json", ("dimension_3_testability", "clarity", "score")),
    "Feasibility": ("score_testability.json", ("dimension_3_testability", "feasibility", "score")),
}
LEGACY_DIM_SOURCE = {"Significance": ("score_impact.json", ("impact", "score"))}
DERIVED = {
    "Impact": ("Significance", "Reach"),
    "Testability": ("Clarity", "Feasibility"),
}
DIMENSIONS = ["Novelty", "Significance", "Reach", "Impact", "Clarity", "Feasibility", "Testability"]
SCORE_FILES = sorted({filename for filename, _ in DIM_SOURCE.values()})
METHOD_LABELS = {
    "cc-opus4.8": "Claude Code",
    "v2-opus4.8": "AI-Scientist",
    "mechanist-opus4.8-gpt5.4": "Mechanist",
}
METHOD_ORDER = list(METHOD_LABELS)
TOPIC_LABELS: dict[str, str] = {}
IGNORE_TOPIC_DIRS = {"claims", "claim", "results", "result", "output", "outputs"}
DEFAULT_JUDGE_MODEL = "GPT-5.6 Sol"
DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "result"


def dig(data, keys):
    """Return the numeric value at a nested key path, or None."""
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current if isinstance(current, (int, float)) and not isinstance(current, bool) else None


def mean(values):
    """Return the mean of non-missing values."""
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def titleize(name: str) -> str:
    """Convert a directory name to a display label."""
    if name in TOPIC_LABELS:
        return TOPIC_LABELS[name]
    words = name.replace("-", " ").replace("_", " ").split()
    return " ".join(word.capitalize() if word.islower() else word for word in words)


def method_sort_key(name: str):
    """Return the configured method order followed by alphabetical fallbacks."""
    if name in METHOD_ORDER:
        return 0, METHOD_ORDER.index(name), ""
    return 1, 0, name.lower()


def find_claim_dirs(root: Path):
    """Find every directory containing a recognized score file."""
    directories = set()
    for filename in SCORE_FILES:
        directories.update(path.parent for path in root.rglob(filename))
    return sorted(directories)


def read_scores(claim_dir: Path, warnings: list[str]):
    """Read all available score dimensions for one claim directory."""
    documents = {}
    for filename in SCORE_FILES:
        path = claim_dir / filename
        if not path.exists():
            continue
        try:
            documents[filename] = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:  # noqa: BLE001
            warnings.append(f"{path}: {type(error).__name__}: {error}")

    scores = {}
    for dimension in BASE_DIMENSIONS:
        filename, keys = DIM_SOURCE[dimension]
        scores[dimension] = dig(documents.get(filename), keys)
        if scores[dimension] is None and dimension in LEGACY_DIM_SOURCE:
            legacy_filename, legacy_keys = LEGACY_DIM_SOURCE[dimension]
            scores[dimension] = dig(documents.get(legacy_filename), legacy_keys)
        if filename in documents and scores[dimension] is None:
            warnings.append(f"{claim_dir / filename}: missing {'.'.join(keys)}")
    for dimension, components in DERIVED.items():
        scores[dimension] = mean(scores[component] for component in components)
    return scores


def split_path(claim_dir: Path, root: Path, ignored: set[str]):
    """Split a claim path into repeat, method, and topic components."""
    parts = claim_dir.relative_to(root).parts
    repeat = ""
    if parts and re.fullmatch(r"test_group_\d+", parts[0]):
        repeat, parts = parts[0], parts[1:]
    if len(parts) < 2:
        return None
    topics = tuple(part for part in parts[1:-1] if part.lower() not in ignored)
    return repeat, parts[0], topics


def collect(root: Path, warnings: list[str], ignored: set[str]):
    """Collect cell values and per-claim records from all repeat directories."""
    cells = defaultdict(lambda: defaultdict(list))
    per_claim = []
    for claim_dir in find_claim_dirs(root):
        parsed = split_path(claim_dir, root, ignored)
        if parsed is None:
            warnings.append(f"unexpected depth, skipped: {claim_dir}")
            continue
        repeat, method, topics = parsed
        scores = read_scores(claim_dir, warnings)
        if all(value is None for value in scores.values()):
            warnings.append(f"no usable scores, skipped: {claim_dir}")
            continue
        for dimension, value in scores.items():
            if value is not None:
                cells[(method, topics)][dimension].append(value)
        per_claim.append((repeat, method, topics, claim_dir.name, scores))
    return cells, per_claim


def format_score(value, digits: int) -> str:
    """Format a score while preserving missing values as empty cells."""
    return "" if value is None else f"{float(value):.{digits}f}"


def write_csv(destination, header, rows) -> None:
    """Write CSV output; '-' selects stdout."""
    if str(destination) == "-":
        writer = csv.writer(sys.stdout)
        writer.writerow(header)
        writer.writerows(rows)
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate score JSON files into a Method-by-Topic table.")
    parser.add_argument("root", nargs="?", default=str(DEFAULT_ROOT), help=f"Result root (default: {DEFAULT_ROOT})")
    parser.add_argument("-o", "--out", default="scores.csv", help="Output CSV path; '-' means stdout")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Value for the Judge Model column")
    parser.add_argument("--digits", type=int, default=2, help="Decimal places (default: 2)")
    parser.add_argument("--with-n", action="store_true", help="Include the score count per cell")
    parser.add_argument("--overall", action="store_true", help="Add an overall row for each method")
    parser.add_argument("--per-claim", help="Also write per-claim details to this CSV")
    parser.add_argument("--group-by-method", action="store_true", help="Group output by method instead of topic")
    parser.add_argument("--ignore-dirs", nargs="*", default=sorted(IGNORE_TOPIC_DIRS), help="Directories excluded from topic labels")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"root not found: {root}")
    warnings: list[str] = []
    cells, per_claim = collect(root, warnings, {name.lower() for name in args.ignore_dirs})
    if not cells:
        sys.exit(f"no score JSON files found under: {root}")

    topic_label = lambda topics: "/".join(titleize(topic) for topic in topics) or "(root)"
    method_label = lambda method: METHOD_LABELS.get(method, method)
    if args.group_by_method:
        order = sorted(cells, key=lambda key: (method_sort_key(key[0]), topic_label(key[1])))
    else:
        order = sorted(cells, key=lambda key: (topic_label(key[1]), method_sort_key(key[0])))

    header = ["Method", "Topic", "Judge Model", *DIMENSIONS]
    if args.with_n:
        header.append("N")
    rows = []
    for key in order:
        values = cells[key]
        row = [method_label(key[0]), topic_label(key[1]), args.judge_model]
        row.extend(format_score(mean(values[dimension]), args.digits) for dimension in DIMENSIONS)
        if args.with_n:
            row.append(max(len(values[dimension]) for dimension in DIMENSIONS))
        rows.append(row)

    if args.overall:
        by_method = defaultdict(lambda: defaultdict(list))
        for (method, _), values in cells.items():
            for dimension in DIMENSIONS:
                by_method[method][dimension].extend(values[dimension])
        for method in sorted(by_method, key=method_sort_key):
            values = by_method[method]
            row = [method_label(method), "Overall", args.judge_model]
            row.extend(format_score(mean(values[dimension]), args.digits) for dimension in DIMENSIONS)
            if args.with_n:
                row.append(max(len(values[dimension]) for dimension in DIMENSIONS))
            rows.append(row)

    write_csv(args.out, header, rows)
    if str(args.out) != "-":
        print(f"wrote {len(rows)} row(s) -> {Path(args.out).resolve()}")

    if args.per_claim:
        detail_header = ["Repeat", "Method", "Topic", "Claim", "Judge Model", *DIMENSIONS]
        detail_rows = []
        for repeat, method, topics, claim, scores in sorted(per_claim):
            detail_rows.append([repeat, method_label(method), topic_label(topics), claim, args.judge_model,
                                *(format_score(scores[dimension], args.digits) for dimension in DIMENSIONS)])
        write_csv(args.per_claim, detail_header, detail_rows)
        if str(args.per_claim) != "-":
            print(f"wrote {len(detail_rows)} claim row(s) -> {Path(args.per_claim).resolve()}")

    if warnings:
        print(f"{len(warnings)} warning(s):", file=sys.stderr)
        for warning in warnings:
            print(f"  {warning}", file=sys.stderr)


if __name__ == "__main__":
    main()
