"""Re-score existing JSONL result files with the current extraction/grading logic.

Reads runs' *.jsonl (schema: {lang, gold, generation, ...}) and rewrites `is_correct` + `extracted`.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mlr.mgsm_eval import extract_answer, numeric_equal


def rescore(in_path: Path) -> dict:
    n_by_lang = defaultdict(int)
    correct_by_lang = defaultdict(int)
    records = []
    with open(in_path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            gold_n = r.get("gold")
            gen = r.get("generation", "")
            pred = extract_answer(gen)
            is_correct = numeric_equal(pred, gold_n)
            r["extracted"] = pred
            r["is_correct"] = bool(is_correct)
            n_by_lang[r["lang"]] += 1
            if is_correct:
                correct_by_lang[r["lang"]] += 1
            records.append(r)
    # Rewrite jsonl
    with open(in_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Rewrite summary
    summary_path = Path(str(in_path).replace(".jsonl", "_summary.json"))
    if summary_path.exists():
        with open(summary_path, "r") as f:
            s = json.load(f)
        per_lang = {l: {"n": n_by_lang[l], "accuracy": correct_by_lang[l] / max(n_by_lang[l], 1), "fidelity": s.get("per_language", {}).get(l, {}).get("fidelity")} for l in n_by_lang}
        s["per_language"] = per_lang
        s["macro_accuracy"] = sum(v["accuracy"] for v in per_lang.values()) / max(len(per_lang), 1)
        with open(summary_path, "w") as f:
            json.dump(s, f, indent=2, ensure_ascii=False)
    return {l: correct_by_lang[l] / max(n_by_lang[l], 1) for l in n_by_lang}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()
    for p in args.paths:
        for pp in Path(".").glob(p):
            if pp.suffix == ".jsonl":
                stats = rescore(pp)
                mac = sum(stats.values()) / max(len(stats), 1)
                print(f"{pp}: macro={mac:.3f}  {dict(stats)}")


if __name__ == "__main__":
    main()
