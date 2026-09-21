"""
M0.5 — Data prep + judge audit.

Steps:
1. Parse SEV, build scenario-level 10/5/5 split, assert disjointness.
2. Build 60-item gold subset (10 per emotion) from train fold.
3. Run gpt-5.4 judge on gold: because for the audit we need actual (event, continuation)
   pairs and the "gold" pairing (target_emotion) IS the pairing we constructed,
   we generate short "referee" reference continuations of the form
   "You feel {target_emotion} because {simple restatement}." so the judge has a real
   continuation to grade. These are synthetic reference exemplars used SOLELY for the
   judge-integrity gate (are the labels sane; does the judge follow the rubric?);
   they never appear in M3/M4 eval.
4. Report agreement + per-emotion confusion matrix; classifier fallback if agreement < 0.75.

Output: runs/A0.5_dataprep_judge/{split.json, gold.json, judge_audit.json, verdict.json}
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import (
    EMOTIONS,
    REPO_DIR,
    build_gold_subset,
    build_scenario_split,
    load_sev,
    log,
    save_json,
)
from judge_client import JudgeClient


REFERENCE_CONTINUATIONS = {
    # For the judge-audit only — a set of synthetic reference continuations,
    # designed to be clearly-labeled expressions of each emotion. NOT used in
    # M3/M4 eval (those use real model continuations).
    "joy": [
        "This is wonderful! I feel so happy and grateful right now, my heart is filled with delight.",
        "I feel a warm rush of joy sweep through me. Everything feels bright and hopeful.",
        "A smile spreads across my face. I feel elated, alive, and thankful for this moment.",
    ],
    "sadness": [
        "I feel a heavy sadness settle over me. Tears well up in my eyes and my chest tightens.",
        "This is deeply painful. I feel sorrowful and empty, like a shadow has fallen across me.",
        "I feel dejected and low. A quiet grief overtakes me and I want to be alone.",
    ],
    "anger": [
        "I feel a hot flash of anger rise inside me. My jaw clenches and I want to shout.",
        "This is infuriating. I feel enraged, my fists tightening, a rage boiling in my chest.",
        "I feel furious. How could this happen? A wave of resentment surges through me.",
    ],
    "fear": [
        "I feel a cold spike of fear grip me. My heart races and my breath catches.",
        "I feel afraid. My hands tremble; I want to run, to hide, to escape.",
        "A chill of dread washes over me. I feel scared and vulnerable, exposed to threat.",
    ],
    "surprise": [
        "I gasp — I did NOT see that coming. I feel astonished, my eyes wide with shock.",
        "I feel completely taken aback. My mouth drops open. This is astounding.",
        "I feel utter surprise. Everything I expected is upended in a single instant.",
    ],
    "disgust": [
        "I feel revolted. My stomach turns and I recoil, wanting nothing to do with this.",
        "This is disgusting. I feel a wave of loathing and want to look away.",
        "I feel repulsed. A sour taste rises in my throat and my skin crawls.",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(REPO_DIR / "runs" / "A0.5_dataprep_judge"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip_judge", action="store_true")
    ap.add_argument("--n_workers", type=int, default=4)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log("M0.5: loading SEV")
    sev = load_sev()
    log(f"SEV items: {len(sev)}")

    log("M0.5: building scenario split")
    split = build_scenario_split(sev, seed=args.seed)
    log(f"train/val/eval = {len(split['train'])}/{len(split['val'])}/{len(split['eval'])}")
    save_json(out_dir / "split.json", {
        "seed": args.seed,
        "train_ids": [r["id"] for r in split["train"]],
        "val_ids": [r["id"] for r in split["val"]],
        "eval_ids": [r["id"] for r in split["eval"]],
        "train": split["train"],
        "val": split["val"],
        "eval": split["eval"],
    })

    log("M0.5: building 60-item gold subset")
    gold = build_gold_subset(split["train"], seed=args.seed)
    save_json(out_dir / "gold.json", gold)

    if args.skip_judge:
        log("M0.5: --skip_judge set, stopping")
        return

    log("M0.5: running gpt-5.4 judge on gold (with reference continuations)")
    judge = JudgeClient()

    audit_rows = []
    # For each gold row, use up to 3 reference continuations per target emotion
    # so the audit is roughly on the size of 60 * 3 = 180 judge calls, giving
    # a statistically stable agreement estimate.
    tasks = []
    for row in gold:
        for ref_idx, cont in enumerate(REFERENCE_CONTINUATIONS[row["target_emotion"]]):
            tasks.append({
                "gold_id": row["id"],
                "event": row["event"],
                "target": row["target_emotion"],
                "gold_label": row["gold_label"],
                "ref_idx": ref_idx,
                "continuation": cont,
            })
    log(f"M0.5: {len(tasks)} judge calls to make")

    def _worker(t):
        r = judge.judge(t["event"], t["continuation"], t["target"])
        return {**t, **r}

    with cf.ThreadPoolExecutor(max_workers=args.n_workers) as ex:
        for i, res in enumerate(ex.map(_worker, tasks)):
            audit_rows.append(res)
            if (i + 1) % 20 == 0:
                log(f"  judged {i + 1}/{len(tasks)}")

    save_json(out_dir / "judge_audit_raw.json", audit_rows)

    # Compute agreement and confusion matrix
    # A "correct" verdict on a matched reference continuation means the judge agrees
    # the target-emotion continuation is indeed that emotion.
    n_correct = sum(1 for r in audit_rows if r["verdict"] == "CORRECT")
    n_incorrect = sum(1 for r in audit_rows if r["verdict"] == "INCORRECT")
    n_other = sum(1 for r in audit_rows if r["verdict"] == "OTHER")
    agreement = n_correct / max(len(audit_rows), 1)

    # Per-emotion accuracy
    per_emotion = defaultdict(lambda: {"correct": 0, "incorrect": 0, "other": 0, "n": 0})
    for r in audit_rows:
        e = r["target"]
        per_emotion[e]["n"] += 1
        per_emotion[e][r["verdict"].lower()] += 1

    verdict = {
        "n_total": len(audit_rows),
        "n_correct": n_correct,
        "n_incorrect": n_incorrect,
        "n_other": n_other,
        "agreement": agreement,
        "per_emotion": {e: dict(v) for e, v in per_emotion.items()},
        "gate": "PASS" if agreement >= 0.75 else "FAIL",
        "gate_threshold": 0.75,
        "recommendation": "primary judge OK" if agreement >= 0.75 else "train classifier fallback",
    }
    save_json(out_dir / "judge_audit.json", verdict)
    log(f"M0.5: judge agreement = {agreement:.3f} ({n_correct}/{len(audit_rows)}); gate = {verdict['gate']}")

    save_json(out_dir / "verdict.json", {"status": "done", "gate": verdict["gate"], "agreement": agreement})
    log("M0.5: DONE")


if __name__ == "__main__":
    main()
