"""Aggregate per-rank QA_I eval JSONs into one accuracy summary.

Input: --run_dir containing rank_i_qa_i.json files from m0_qa_i_eval.py.
Output: --out qa_i_acc.json with accuracy + per-verdict breakdown.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True,
                    help="Dir containing rank_*_qa_i.json shards.")
    ap.add_argument("--pattern", type=str, default="rank_*_qa_i.json")
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args()

    d = Path(args.run_dir)
    shards = sorted(d.glob(args.pattern))
    if not shards:
        raise RuntimeError(f"No shard files matched {args.pattern} in {d}")

    all_items: list[dict] = []
    meta = {}
    for s in shards:
        with open(s) as f:
            data = json.load(f)
        all_items.extend(data["items"])
        # First-seen wins for meta (arm, seed, lr_tag)
        for k in ("arm", "seed", "lr_tag", "ckpt"):
            if k not in meta and k in data:
                meta[k] = data[k]

    # Deduplicate by idx (in case sharding overlapped)
    seen_idxs: set[int] = set()
    unique = []
    for it in all_items:
        if it["idx"] not in seen_idxs:
            seen_idxs.add(it["idx"])
            unique.append(it)
    unique.sort(key=lambda x: x["idx"])

    n = len(unique)
    n_correct = sum(1 for it in unique if it.get("judge_verdict") == "CORRECT")
    n_incorrect = sum(1 for it in unique if it.get("judge_verdict") == "INCORRECT")
    n_other = sum(1 for it in unique if it.get("judge_verdict") == "OTHER")
    n_error = sum(1 for it in unique if it.get("judge_verdict") == "ERROR")
    n_judged = n_correct + n_incorrect + n_other
    # Accuracy over EVERYTHING (excluding errors — those are API failures).
    denom = max(n_judged, 1)
    acc = n_correct / denom

    summary = {
        **meta,
        "n_items": n,
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_other": n_other,
        "n_error": n_error,
        "n_judged": n_judged,
        "accuracy": acc,
        "other_rate": n_other / denom if denom else 0.0,
        "error_rate_over_all": n_error / max(n, 1),
        "shards_merged": [str(s) for s in shards],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": summary, "items": unique}, f, indent=2, ensure_ascii=False)
    print(f"[aggregate] n={n} acc={acc:.4f} correct={n_correct} incorrect={n_incorrect} "
          f"other={n_other} error={n_error} -> {out}", flush=True)


if __name__ == "__main__":
    main()
