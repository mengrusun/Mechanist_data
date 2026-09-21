#!/usr/bin/env python3
"""Deploy a milestone as parallel waves across the available GPUs from a fixed
allow-list, respecting per-condition GPU pinning and cost tracking.

Usage:
  python scripts/deploy_wave.py \
      --milestone M2 \
      --task gsm8k \
      --condition_ids "$(cat data/prefixes/prefixes.json | jq -r '.[].condition_id' | paste -sd,)" \
      --n_items 500 \
      --eval_mode cot \
      --gpu_pool 2,6 \
      --max_parallel 2 \
      --out_dir runs/M2 \
      --activations_n 200

For each condition, launches `python scripts/run_prefix_eval.py ...` in a
subprocess with CUDA_VISIBLE_DEVICES pinned. Waits and re-queues.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def gpus_have_memory(gpus: List[int], min_free_mib: int = 40_000) -> List[int]:
    """Return subset of gpus with at least min_free_mib free (MiB)."""
    import re as _re
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader"],
            text=True,
        )
        free_by = {}
        for line in out.strip().splitlines():
            idx_str, free_str = line.split(",")
            idx = int(idx_str.strip())
            free = int(_re.search(r"\d+", free_str).group())
            free_by[idx] = free
        return [g for g in gpus if free_by.get(g, 0) >= min_free_mib]
    except Exception:
        return gpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--condition_ids", required=True, help="Comma-separated")
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--eval_mode", default="cot")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--gpu_pool", required=True, help="Comma-separated allowed GPUs")
    ap.add_argument("--max_parallel", type=int, default=2)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--activations_n", type=int, default=None)
    ap.add_argument("--skip_activations", action="store_true", default=False)
    ap.add_argument("--min_free_mib", type=int, default=40_000)
    ap.add_argument("--script", default="scripts/run_prefix_eval.py")
    ap.add_argument("--skip_existing", action="store_true", default=True)
    args = ap.parse_args()

    cond_ids = args.condition_ids.split(",")
    print(f"[deploy] milestone={args.milestone} task={args.task} n_conditions={len(cond_ids)}")

    # Filter to jobs not yet done
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    todo: List[str] = []
    for cid in cond_ids:
        out_json = f"{args.out_dir}/{args.task}_{cid}.json"
        if args.skip_existing and os.path.exists(out_json):
            try:
                d = json.load(open(out_json))
                if "accuracy" in d and len(d.get("per_item", [])) >= args.n_items - 1:
                    print(f"  [skip] {cid} already done acc={d['accuracy']:.4f}")
                    continue
            except Exception:
                pass
        todo.append(cid)
    print(f"[deploy] {len(todo)} conditions to run")

    gpu_pool = [int(g) for g in args.gpu_pool.split(",")]

    # Simple wave-based dispatch
    in_flight: Dict[int, Tuple[str, subprocess.Popen, float]] = {}  # gpu -> (cid, proc, t0)
    cond_iter = iter(todo)
    total_started = 0

    while True:
        # Check running processes for completion
        finished_gpus = []
        for gpu, (cid, proc, t0) in list(in_flight.items()):
            ret = proc.poll()
            if ret is not None:
                elapsed = time.time() - t0
                print(f"[done] gpu={gpu} cond={cid} ret={ret} elapsed={elapsed:.1f}s")
                finished_gpus.append(gpu)
        for gpu in finished_gpus:
            in_flight.pop(gpu)

        # Refill idle GPUs from the pool
        free_gpus = gpus_have_memory([g for g in gpu_pool if g not in in_flight],
                                     min_free_mib=args.min_free_mib)
        while free_gpus and len(in_flight) < args.max_parallel:
            gpu = free_gpus.pop(0)
            try:
                cid = next(cond_iter)
            except StopIteration:
                break
            out_json = f"{args.out_dir}/{args.task}_{cid}.json"
            act_pt = f"{args.out_dir}/act_{args.task}_{cid}.pt"
            # Choose whether to cache activations for this run
            cmd = [
                "python", args.script,
                "--task", args.task,
                "--condition_id", cid,
                "--n_items", str(args.n_items),
                "--eval_mode", args.eval_mode,
                "--max_new_tokens", str(args.max_new_tokens),
                "--out", out_json,
                "--gpu_ids", str(gpu),
            ]
            if args.skip_activations:
                cmd += ["--skip_activations"]
            else:
                cmd += ["--activations_out", act_pt]
                if args.activations_n:
                    cmd += ["--activations_n", str(args.activations_n)]

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            env["VLLM_LOGGING_LEVEL"] = "WARNING"
            log_path = f"logs/{args.milestone}_{args.task}_{cid}_gpu{gpu}.log"
            Path("logs").mkdir(exist_ok=True)
            log_f = open(log_path, "w")
            proc = subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT)
            in_flight[gpu] = (cid, proc, time.time())
            total_started += 1
            print(f"[launch] gpu={gpu} cond={cid} pid={proc.pid} log={log_path}")

        if not in_flight:
            try:
                cid_next = next(cond_iter)
                # Nothing running and no free GPUs? Wait a bit
                # Actually cond_iter is peekable via StopIteration — we consumed one, put back
                # Simpler: rewind by prepending
                # Instead: sleep and retry
                cond_iter = iter([cid_next] + list(cond_iter))
                print(f"[wait] no free GPU; sleeping 30s")
                time.sleep(30)
            except StopIteration:
                # done — nothing running, no more todo
                break
        else:
            time.sleep(15)

    print(f"[deploy] milestone {args.milestone} complete. total_runs_launched={total_started}")


if __name__ == "__main__":
    main()
