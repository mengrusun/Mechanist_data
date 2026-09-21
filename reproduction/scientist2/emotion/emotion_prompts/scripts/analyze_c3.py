#!/usr/bin/env python3
"""M4: post-hoc analysis for C3a (argmax flipping) and C3b (intensity monotonicity).

Reads per-condition result JSONs from runs/M2 (gsm8k) and runs/M3 (socialiqa, medqa),
computes:
- per-task per-emotion argmax across 4 (intensity × wording-source) cells
- checks whether argmax identity flips ≥ once across {gsm8k, socialiqa, medqa}   → C3a
- per-task per-emotion paired bootstrap CI on Δ(intensity-2 − intensity-1) averaged
  across wording sources; reports fraction failing monotonicity                  → C3b

Writes reports/M4_c3_analysis.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

EMOTIONS = ["happiness", "sadness", "fear", "anger", "disgust", "surprise"]


def load_task_condition(dir_path: str, task: str) -> dict:
    """Load task_condition_id.json files; return dict of condition_id → per_item list of correctness."""
    out = {}
    pattern = os.path.join(dir_path, f"{task}_*.json")
    for path in sorted(glob.glob(pattern)):
        try:
            d = json.load(open(path))
        except Exception:
            continue
        if "per_item" not in d:
            continue
        cid = d["condition_id"]
        # extract correctness list
        corr = [r["correct"] for r in d["per_item"]]
        out[cid] = corr
    return out


def acc(vec: list) -> float:
    return sum(vec) / len(vec) if vec else 0.0


def bootstrap_ci(deltas: np.ndarray, n_boot: int = 1000, seed: int = 42) -> tuple[float, float]:
    """95 % CI of the mean of a paired-Δ array."""
    rng = np.random.default_rng(seed)
    means = []
    n = len(deltas)
    if n < 2:
        return (0.0, 0.0)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means.append(deltas[idx].mean())
    means = np.array(means)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsm8k", default="runs/M2")
    ap.add_argument("--socialiqa", default="runs/M3")
    ap.add_argument("--medqa", default="runs/M3")
    ap.add_argument("--out", default="reports/M4_c3_analysis.json")
    args = ap.parse_args()

    Path(os.path.dirname(args.out)).mkdir(parents=True, exist_ok=True)

    tasks = {
        "gsm8k": load_task_condition(args.gsm8k, "gsm8k"),
        "socialiqa": load_task_condition(args.socialiqa, "socialiqa"),
        "medqa": load_task_condition(args.medqa, "medqa"),
    }
    for tname, cond_dict in tasks.items():
        print(f"[load] task={tname} n_conditions={len(cond_dict)}")

    result = {"per_task_summary": {}, "C3a": {}, "C3b": {}}

    # Per-task accuracies
    per_task_acc = {}
    for tname, cond in tasks.items():
        per_task_acc[tname] = {cid: acc(v) for cid, v in cond.items()}

    # ---- C3a: argmax emotion identity across task families ----
    # For each task, compute mean acc for each emotion averaging over 4 cells
    # (intensity ∈ {1,2}, wording_source ∈ {human, llm}). Take argmax emotion.
    argmax_by_task = {}
    for tname in tasks:
        emo_means = {}
        for emo in EMOTIONS:
            cells = [f"{emo}_{i}_{ws}" for i in (1, 2) for ws in ("human", "llm")]
            accs = [per_task_acc[tname][c] for c in cells if c in per_task_acc[tname]]
            if accs:
                emo_means[emo] = float(np.mean(accs))
        if emo_means:
            argmax_by_task[tname] = max(emo_means, key=emo_means.get)
            result["per_task_summary"][tname] = {
                "per_emotion_mean_acc": emo_means,
                "argmax_emotion": argmax_by_task[tname],
                "neutral_acc": per_task_acc[tname].get("neutral", None),
            }

    flips = len(set(argmax_by_task.values())) > 1 if len(argmax_by_task) >= 2 else False
    result["C3a"] = {
        "argmax_by_task": argmax_by_task,
        "argmax_flips_across_tasks": bool(flips),
        "n_tasks_compared": len(argmax_by_task),
        "criterion": "flips >= 1 comparison",
        "PASS": bool(flips),
    }

    # ---- C3b: monotone intensity on GSM8K ----
    # For each emotion, per-item Δ(intensity-2 − intensity-1) averaged across wording sources.
    c3b_details = {}
    n_fail = 0
    for tname in tasks:
        c3b_details[tname] = {}
        for emo in EMOTIONS:
            # paired per-item Δ = 0.5*(int2_human - int1_human) + 0.5*(int2_llm - int1_llm)
            cells_i1 = [f"{emo}_1_human", f"{emo}_1_llm"]
            cells_i2 = [f"{emo}_2_human", f"{emo}_2_llm"]
            v1_list = [tasks[tname].get(c) for c in cells_i1]
            v2_list = [tasks[tname].get(c) for c in cells_i2]
            if any(v is None for v in v1_list + v2_list):
                continue
            # Take per-item mean across wording sources
            n_items = min(len(v) for v in v1_list + v2_list)
            v1 = 0.5 * (np.array(v1_list[0][:n_items]) + np.array(v1_list[1][:n_items]))
            v2 = 0.5 * (np.array(v2_list[0][:n_items]) + np.array(v2_list[1][:n_items]))
            delta = v2 - v1
            mean_delta = float(delta.mean())
            lo, hi = bootstrap_ci(delta, n_boot=1000)
            straddles_zero = (lo <= 0.0 <= hi)
            c3b_details[tname][emo] = {
                "mean_delta_i2_minus_i1": mean_delta,
                "ci95": [lo, hi],
                "straddles_zero_or_reversed": bool(straddles_zero or mean_delta < 0),
            }
            if tname == "gsm8k" and (straddles_zero or mean_delta < 0):
                n_fail += 1

    result["C3b"] = {
        "per_task": c3b_details,
        "n_emotions_failing_monotonicity_on_gsm8k": n_fail,
        "criterion": "n_fail >= 3 of 6 on GSM8K",
        "PASS": bool(n_fail >= 3),
    }

    # ---- C2 spread ordering (bonus context, since it uses same numbers) ----
    def spread(tname: str) -> float | None:
        accs = [per_task_acc[tname][c]
                for c in per_task_acc[tname]
                if c not in ("neutral", "filler_matched_length")]
        return float(np.max(accs) - np.min(accs)) if accs else None

    spread_math = spread("gsm8k")
    spread_social = spread("socialiqa")
    spread_factual = spread("medqa")
    c2_pass = False
    c2_details = {"spread_math": spread_math, "spread_social": spread_social,
                  "spread_factual": spread_factual}
    if spread_math is not None and spread_social is not None:
        c2_details["spread_social_ge_2x_spread_math"] = bool(spread_social >= 2 * spread_math)
        # ordering: social > factual > math in ≥ 2/3 comparisons
        comparisons = 0
        if spread_social is not None and spread_factual is not None:
            comparisons += int(spread_social > spread_factual)
        if spread_factual is not None and spread_math is not None:
            comparisons += int(spread_factual > spread_math)
        if spread_social is not None and spread_math is not None:
            comparisons += int(spread_social > spread_math)
        c2_details["n_ordering_comparisons_holding"] = comparisons
        c2_pass = c2_details.get("spread_social_ge_2x_spread_math", False) and comparisons >= 2
    c2_details["PASS"] = c2_pass
    result["C2"] = c2_details

    # Write
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[write] {args.out}")
    print(f"[C2] PASS={c2_pass} spread_math={spread_math} spread_social={spread_social} spread_factual={spread_factual}")
    print(f"[C3a] PASS={result['C3a']['PASS']} argmax_by_task={argmax_by_task}")
    print(f"[C3b] PASS={result['C3b']['PASS']} n_fail_gsm8k={n_fail}/6")


if __name__ == "__main__":
    main()
