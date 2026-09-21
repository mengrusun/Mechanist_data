"""Step 3 — Stage 1.5 Verbalized-Confidence Variance Diagnostic (B5).

Reads:
  - artifacts/turn2_gen.jsonl (P0 confidences on the 10k slice)
  - artifacts/turn1_gen.jsonl (for parseable-vs-unparseable correctness gap)
  - artifacts/split_manifest.json

Outputs:
  - artifacts/stage15_variance.json  (metrics + selected path + threshold config)

Decision rule (pre-registered in FINAL_PROPOSAL.md §6.6):
- If std(c) >= 15 AND share(c >= 95) <= 0.7 → continuous path (Spearman rho ≥ 0.5)
- Else → ordinal path (top-1 acc ≥ 0.55, macro-F1 ≥ 0.4)

For probe_v_binary (v_v extraction):
- continuous path: binarize at population median
- ordinal path: binarize at 30th percentile (guarantees >= 30% minority class)
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils as U


def main():
    turn1 = {r["question_id"]: r for r in U.load_jsonl(U.ARTIFACT_DIR / "turn1_gen.jsonl")}
    turn2 = U.load_jsonl(U.ARTIFACT_DIR / "turn2_gen.jsonl")
    split = U.load_json(U.ARTIFACT_DIR / "split_manifest.json")
    train_ids = set(split["train_ids"])

    # Filter to train slice for the variance diagnostic (per plan §7 stage 1.5)
    train_turn2 = [r for r in turn2 if r["question_id"] in train_ids]

    # Values c and parseable
    cs = np.array([r["c"] for r in train_turn2 if r["c_parseable"]], dtype=np.float64)
    parseable = sum(1 for r in train_turn2 if r["c_parseable"])
    total = len(train_turn2)

    stats = {
        "n_total_train": total,
        "n_parseable": int(parseable),
        "parseable_rate": float(parseable / max(total, 1)),
        "unparseable_rate": float(1 - parseable / max(total, 1)),
    }
    if cs.size:
        # Quantiles + share_at_max
        qs = [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        stats["mean_c"] = float(np.mean(cs))
        stats["std_c"] = float(np.std(cs))
        stats["quantiles"] = {str(q): float(np.quantile(cs, q)) for q in qs}
        stats["share_c_ge_95"] = float(np.mean(cs >= 95))
        stats["share_c_le_5"] = float(np.mean(cs <= 5))
        # Entropy on 10 fixed bins
        bins = np.arange(0, 101, 10)
        hist, _ = np.histogram(cs, bins=bins, density=True)
        p = hist * 10.0  # step size
        p = p[p > 0]
        stats["entropy_c_bits"] = float(-(p * np.log2(p)).sum() * 10.0) if p.size else 0.0

    # Parse-bias diagnostic: parseable-vs-unparseable correctness gap
    y_p, y_u = [], []
    for r in train_turn2:
        t1 = turn1.get(r["question_id"])
        if t1 is None:
            continue
        y = int(t1["y_correct"])
        if r["c_parseable"]:
            y_p.append(y)
        else:
            y_u.append(y)
    if len(y_p) and len(y_u):
        stats["accuracy_parseable"] = float(np.mean(y_p))
        stats["accuracy_unparseable"] = float(np.mean(y_u))
        stats["accuracy_gap_pp"] = float(100 * (stats["accuracy_parseable"] - stats["accuracy_unparseable"]))
        stats["parse_bias_flag"] = bool(abs(stats["accuracy_gap_pp"]) > 5.0)

    # Path decision
    std_c = stats.get("std_c", 0.0)
    share_max = stats.get("share_c_ge_95", 1.0)
    if std_c >= 15.0 and share_max <= 0.7:
        stats["path"] = "continuous"
        stats["primary_metric"] = "spearman_rho"
        stats["primary_threshold"] = 0.5
        stats["binarize_at"] = "median"
    else:
        stats["path"] = "ordinal"
        stats["primary_metric"] = "ordinal_top1_and_macro_f1"
        stats["primary_threshold_top1"] = 0.55
        stats["primary_threshold_macro_f1"] = 0.4
        stats["binarize_at"] = "30th_percentile"

    # Compute the binarize threshold from the observed c distribution (train)
    if cs.size:
        if stats["binarize_at"] == "median":
            stats["binarize_threshold"] = float(np.median(cs))
        else:
            stats["binarize_threshold"] = float(np.quantile(cs, 0.3))
    else:
        stats["binarize_threshold"] = 50.0

    # Fallback P1 recommendation
    if stats["unparseable_rate"] >= 0.10:
        stats["fallback_switch_to_p1"] = True
    else:
        stats["fallback_switch_to_p1"] = False

    U.dump_json(U.ARTIFACT_DIR / "stage15_variance.json", stats)
    print("[step3] Stage 1.5 diagnostic:")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
