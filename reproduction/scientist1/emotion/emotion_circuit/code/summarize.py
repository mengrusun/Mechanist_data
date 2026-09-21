"""Compute per-emotion / per-theme / overall accuracy stats from a labeled jsonl."""
from __future__ import annotations
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from common import OUT_DIR, EMOTIONS, load_jsonl


def compute(rows, split_key=None):
    total = 0
    correct = 0
    per = defaultdict(lambda: [0, 0])  # {key: [correct, total]}
    per_theme = defaultdict(lambda: [0, 0])
    per_ec = defaultdict(lambda: [0, 0])  # (emotion, theme)
    for r in rows:
        total += 1
        c = int(r.get("correct", 0))
        correct += c
        per[r["emotion"]][0] += c
        per[r["emotion"]][1] += 1
        per_theme[r["theme"]][0] += c
        per_theme[r["theme"]][1] += 1
        per_ec[(r["emotion"], r["theme"])][0] += c
        per_ec[(r["emotion"], r["theme"])][1] += 1
    return {
        "n": total,
        "overall": correct / max(total, 1),
        "per_emotion": {k: v[0] / max(v[1], 1) for k, v in per.items()},
        "per_theme": {k: v[0] / max(v[1], 1) for k, v in per_theme.items()},
        "per_emotion_n": {k: v[1] for k, v in per.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None,
                    help="Optional labels for each input file")
    ap.add_argument("--exclude_neutral", action="store_true",
                    help="Exclude rows with target=neutral from stats")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    results = {}
    for i, path in enumerate(args.input):
        p = OUT_DIR / path if not os.path.isabs(path) else Path(path)
        rows = load_jsonl(str(p))
        if args.exclude_neutral:
            rows = [r for r in rows if r.get("emotion") != "neutral"]
        label = args.labels[i] if args.labels else str(path)
        results[label] = compute(rows)

    print(json.dumps(results, indent=2))
    if args.out:
        outp = OUT_DIR / args.out if not os.path.isabs(args.out) else Path(args.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with open(outp, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[write] {outp}")


if __name__ == "__main__":
    main()
