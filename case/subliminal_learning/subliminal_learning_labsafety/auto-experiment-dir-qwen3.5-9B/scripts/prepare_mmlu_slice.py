#!/usr/bin/env python3
"""
Prepare MMLU capability slice for M2 specificity check.

Selects 500 items from three MMLU test subjects:
  abstract_algebra  (100)
  college_mathematics (100)
  professional_law (first 300)

Output: mechanism/mmlu_slice.jsonl with fields
  { subject, mmlu_row_id, question, choices, gold_letter, gold_idx }

gold_letter is derived from `answer` (int 0-3) mapped to 'A'/'B'/'C'/'D'.
"""
import json
import sys
from pathlib import Path

import pandas as pd


MMLU_ROOT = Path("/data/zhenqian/data/mmlu")
SUBJECTS = [
    ("abstract_algebra", 100),
    ("college_mathematics", 100),
    ("professional_law", 300),
]
OUT = Path("mechanism/mmlu_slice.jsonl")


def main():
    out_recs = []
    for subject, n_take in SUBJECTS:
        df = pd.read_parquet(MMLU_ROOT / subject / "test-00000-of-00001.parquet")
        df = df.iloc[:n_take].reset_index(drop=True)
        for i, row in df.iterrows():
            gold_idx = int(row["answer"])
            gold_letter = "ABCD"[gold_idx]
            rec = {
                "subject": subject,
                "mmlu_row_id": f"{subject}_{i:04d}",
                "question": str(row["question"]),
                "choices": [str(c) for c in row["choices"]],
                "gold_letter": gold_letter,
                "gold_idx": gold_idx,
            }
            out_recs.append(rec)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        for r in out_recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    per_subj = {}
    for r in out_recs:
        per_subj.setdefault(r["subject"], 0)
        per_subj[r["subject"]] += 1
    print(f"[mmlu-slice] wrote {len(out_recs)} items → {OUT}")
    print(f"[mmlu-slice] per-subject: {per_subj}")


if __name__ == "__main__":
    sys.exit(main() or 0)
