#!/usr/bin/env python3
"""
C1 sign-consistency post-hoc analysis.

Reads per-item correctness from runs/M2/*.json (neutral + 24 emotional conditions)
and computes, per emotional condition, the per-item sign-consistency:

  sign_consistency(e) = fraction of items i where sign(correct_e[i] - correct_neutral[i]) is consistent
                       across items in a per-item comparison. But since each item's condition
                       yields a binary correct/incorrect, we adapt:

We use the standard per-item sign-consistency band interpretation for C1:
  band[e] = fraction of items where the emotional condition's outcome DIFFERS from neutral
            in the direction favoring the emotional condition (i.e., emotional_correct=1 &
            neutral_correct=0) divided by the total number of items where the two conditions
            disagree (i.e., emotional_correct != neutral_correct).

Equivalently: band[e] = #(emotional wins over neutral) / #(disagreements)

C1 predicts that sign_consistency ∈ [0.4, 0.6] for all 24 emotional conditions — meaning
the emotional condition wins about half the disagreements (no consistent direction beyond
noise). Values outside [0.4, 0.6] would indicate a consistent directional signal.

Reads:
  runs/M2/gsm8k_neutral.json  (baseline)
  runs/M2/gsm8k_<emotion>_<intensity>_<wording_source>.json  (24 emotional conditions)

Writes:
  reports/C1_sign_consistency.json  (per-condition band values + full breakdown)

Prints:
  Summary table + PASS/FAIL of the [0.4, 0.6] band gate.
"""
import json
import os
import glob
from pathlib import Path

M2_DIR = Path("runs/M2")
OUT_PATH = Path("reports/C1_sign_consistency.json")

EMOTIONS = ["happiness", "sadness", "fear", "anger", "disgust", "surprise"]
INTENSITIES = [1, 2]
WORDING_SOURCES = ["human", "llm"]


def load_per_item(path):
    """Load per-item correctness map: {item_id: correct}."""
    with open(path) as f:
        d = json.load(f)
    return {row["item_id"]: int(row["correct"]) for row in d["per_item"]}, d["accuracy"]


def main():
    # Load neutral baseline
    neutral_path = M2_DIR / "gsm8k_neutral.json"
    if not neutral_path.exists():
        raise FileNotFoundError(f"Missing baseline: {neutral_path}")
    neutral, neutral_acc = load_per_item(neutral_path)
    print(f"Neutral baseline: {len(neutral)} items, acc={neutral_acc:.4f}")

    results = {
        "meta": {
            "task": "gsm8k",
            "n_items_neutral": len(neutral),
            "neutral_accuracy": neutral_acc,
            "gate_band": [0.4, 0.6],
            "predicate": "sign_consistency = wins_over_neutral / disagreements; C1 predicts band ∈ [0.4, 0.6]"
        },
        "per_condition": {},
        "summary": {}
    }

    n_conditions_total = 0
    n_conditions_in_band = 0
    n_conditions_out_of_band = 0
    out_of_band_details = []

    for emotion in EMOTIONS:
        for intensity in INTENSITIES:
            for wording in WORDING_SOURCES:
                cond_id = f"{emotion}_{intensity}_{wording}"
                path = M2_DIR / f"gsm8k_{cond_id}.json"
                if not path.exists():
                    print(f"  MISSING: {path}")
                    continue
                emo, emo_acc = load_per_item(path)

                # Align items
                common = set(neutral) & set(emo)
                wins = 0  # emotional correct, neutral incorrect
                losses = 0  # emotional incorrect, neutral correct
                agrees = 0  # both same
                for iid in common:
                    n_c = neutral[iid]
                    e_c = emo[iid]
                    if e_c == n_c:
                        agrees += 1
                    elif e_c == 1 and n_c == 0:
                        wins += 1
                    elif e_c == 0 and n_c == 1:
                        losses += 1

                disagreements = wins + losses
                sign_consistency = wins / disagreements if disagreements > 0 else float("nan")
                delta_acc = emo_acc - neutral_acc

                in_band = 0.4 <= sign_consistency <= 0.6 if disagreements > 0 else True

                results["per_condition"][cond_id] = {
                    "accuracy": emo_acc,
                    "delta_acc_vs_neutral": delta_acc,
                    "n_items_compared": len(common),
                    "wins_over_neutral": wins,
                    "losses_to_neutral": losses,
                    "agrees_with_neutral": agrees,
                    "disagreements": disagreements,
                    "sign_consistency": sign_consistency,
                    "in_c1_band_0.4_0.6": in_band
                }

                n_conditions_total += 1
                if in_band:
                    n_conditions_in_band += 1
                else:
                    n_conditions_out_of_band += 1
                    out_of_band_details.append({
                        "cond_id": cond_id,
                        "sign_consistency": round(sign_consistency, 4),
                        "delta_acc": round(delta_acc, 4)
                    })

    results["summary"] = {
        "n_conditions_evaluated": n_conditions_total,
        "n_conditions_in_c1_band": n_conditions_in_band,
        "n_conditions_out_of_band": n_conditions_out_of_band,
        "fraction_in_band": n_conditions_in_band / n_conditions_total if n_conditions_total else 0.0,
        "out_of_band_details": out_of_band_details,
        # C1 PASS gate: all 24 conditions in band [0.4, 0.6]
        "c1_sign_consistency_gate_PASS": n_conditions_out_of_band == 0
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary table
    print(f"\n=== C1 sign-consistency per condition ===")
    print(f"{'cond_id':40s}  {'acc':>7s}  {'Δacc':>7s}  {'wins':>5s}  {'loss':>5s}  {'disag':>5s}  {'sign_cons':>9s}  {'in_band':>7s}")
    for cond_id, r in results["per_condition"].items():
        band_mark = "YES" if r["in_c1_band_0.4_0.6"] else "no"
        sc_str = f"{r['sign_consistency']:.4f}" if r['disagreements'] > 0 else "n/a"
        print(f"{cond_id:40s}  {r['accuracy']:.4f}  {r['delta_acc_vs_neutral']:+.4f}  {r['wins_over_neutral']:5d}  {r['losses_to_neutral']:5d}  {r['disagreements']:5d}  {sc_str:>9s}  {band_mark:>7s}")

    print(f"\n=== Summary ===")
    s = results["summary"]
    print(f"  Conditions in band [0.4, 0.6]: {s['n_conditions_in_c1_band']}/{s['n_conditions_evaluated']} ({s['fraction_in_band']:.2%})")
    print(f"  Conditions out of band: {s['n_conditions_out_of_band']}")
    if out_of_band_details:
        print(f"  Out-of-band cells:")
        for d in out_of_band_details:
            print(f"    - {d['cond_id']}: sign_consistency={d['sign_consistency']}, Δacc={d['delta_acc']:+.4f}")
    print(f"  C1 sign-consistency gate PASS: {s['c1_sign_consistency_gate_PASS']}")
    print(f"\nWritten to {OUT_PATH}")


if __name__ == "__main__":
    main()
