"""M1: Behavioral evaluation for a (model, task) cell.

Reads a belief task jsonl, computes the log-prob-comparison accuracy on the FULL split,
writes a json with {acc, correct_count, total, wilson_ci_low, wilson_ci_high, per_example}.

Also (when --write-gate-summary is passed) aggregates the 9 (model, task) cells and writes
`behavioral/above_chance_gate.json` for M2 to consume.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from belief_utils import (
    load_model_and_tokenizer,
    load_task,
    evaluate_task_accuracy,
    save_json,
    load_json,
    set_seed,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["pythia-410m", "pythia-1b", "pythia-2.8b"])
    ap.add_argument("--task", required=True, choices=["world_knowledge", "personal_belief", "attributed_belief"])
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--model-root", required=True)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    ap.add_argument("--batch-size", type=int, default=32, help="unused; kept for template symmetry")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", required=True)
    ap.add_argument("--write-gate-summary", default=None,
                    help="if provided, also (over)write behavioral/above_chance_gate.json under this path's parent dir")
    args = ap.parse_args()

    set_seed(args.seed)
    net, tok = load_model_and_tokenizer(args.model_root, args.model, dtype=args.dtype, device=args.device)
    examples = load_task(args.data_root, args.task)
    print(f"[m1] {args.model} × {args.task}: {len(examples)} examples")
    result = evaluate_task_accuracy(net, tok, examples, device=args.device)
    result["model"] = args.model
    result["task"] = args.task
    result["data_root"] = args.data_root
    result["dtype"] = args.dtype
    save_json(args.output, result)
    print(f"[m1] {args.model} × {args.task}: acc={result['acc']:.4f}  ci=[{result['wilson_ci_low']:.4f},"
          f"{result['wilson_ci_high']:.4f}]  → {args.output}")

    if args.write_gate_summary is not None:
        _rebuild_gate_summary(args.write_gate_summary)


def _rebuild_gate_summary(summary_path: str) -> None:
    """Scan the behavioral/ directory and write the above-chance gate summary."""
    base = os.path.dirname(summary_path)
    models = ["pythia-410m", "pythia-1b", "pythia-2.8b"]
    tasks = ["world_knowledge", "personal_belief", "attributed_belief"]
    gate = {"models": models, "tasks": tasks, "gate_threshold": 0.5, "cells": [], "above_chance_pairs": []}
    for m in models:
        for t in tasks:
            p = os.path.join(base, m, f"{t}.json")
            if not os.path.exists(p):
                continue
            r = load_json(p)
            passes = r["acc"] > 0.5 and r["wilson_ci_low"] > 0.5
            gate["cells"].append({
                "model": m, "task": t, "acc": r["acc"], "wilson_ci_low": r["wilson_ci_low"],
                "wilson_ci_high": r["wilson_ci_high"], "above_chance": passes,
            })
            if passes and t in ("personal_belief", "attributed_belief"):
                target = "personal" if t == "personal_belief" else "attributed"
                gate["above_chance_pairs"].append({"model": m, "target": target})
    save_json(summary_path, gate)
    print(f"[m1] gate summary → {summary_path}: {len(gate['above_chance_pairs'])} (model,target) pairs cleared")


if __name__ == "__main__":
    main()
