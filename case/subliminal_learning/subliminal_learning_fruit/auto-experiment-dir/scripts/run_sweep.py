"""M0.4 orchestrator: LR sweep for student LoRA (5 LRs × 2 arms × 3 seeds = 30 runs).

Runs 4 jobs in parallel across CUDA_VISIBLE_DEVICES={4,5,6,7}. Each job trains a student
LoRA then evaluates it on eval_pref160.

Idempotent: skips a job whose eval json already exists (RESUME semantics).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from itertools import product


LRS = ["1e-5", "3e-5", "1e-4", "3e-4", "1e-3"]
ARMS = ["teacher", "ctrl"]
SEEDS = [42, 200, 201]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr-list", nargs="+", default=LRS)
    p.add_argument("--arms", nargs="+", default=ARMS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--data-dir", default="data/channel_final")
    p.add_argument("--out-ckpt-dir", default="checkpoints/student_sweep")
    p.add_argument("--out-results-dir", default="results/M0/sweep")
    p.add_argument("--eval-prompts",
                   default="/data/zhenqian/exp/subliminal/multi_modal_B/data/eval_pref160.txt")
    p.add_argument("--gpus", default="4,5,6,7",
                   help="Comma-separated GPU ids to use (one job per GPU at a time).")
    p.add_argument("--train-res", type=int, default=512)
    p.add_argument("--eval-res", type=int, default=512)
    p.add_argument("--eval-steps", type=int, default=15)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--skip-existing", action="store_true", default=True)
    return p.parse_args()


def _tag(arm, lr, seed):
    return f"{arm}_lr{lr}_seed{seed}"


def _train_cmd(args, arm, lr, seed):
    tag = _tag(arm, lr, seed)
    out = f"{args.out_ckpt_dir}/{tag}"
    return [
        "python", "-m", "scripts.train_lora",
        "--data", f"{args.data_dir}/{arm}_channel.jsonl",
        "--data-root", ".",   # jsonl paths are relative to project root (data/gen/...)
        "--lora-rank", str(args.lora_rank),
        "--lr", lr,
        "--epochs", str(args.epochs),
        "--seed", str(seed),
        "--out", out,
        "--resolution", str(args.train_res),
        "--batch", str(args.batch),
        "--grad-accum", str(args.grad_accum),
        "--log-every", "10",
    ]


def _eval_cmd(args, arm, lr, seed):
    tag = _tag(arm, lr, seed)
    ckpt = f"{args.out_ckpt_dir}/{tag}"
    outj = f"{args.out_results_dir}/{tag}.json"
    return [
        "python", "-m", "scripts.eval_student",
        "--lora", ckpt,
        "--prompts", args.eval_prompts,
        "--out", outj,
        "--seed", str(seed),
        "--height", str(args.eval_res),
        "--width", str(args.eval_res),
        "--num-inference-steps", str(args.eval_steps),
        "--tag", tag,
    ]


def _combined_cmd(args, arm, lr, seed):
    """Return a single bash command that trains then evaluates in sequence on ONE GPU."""
    tag = _tag(arm, lr, seed)
    cktgz = f"{args.out_ckpt_dir}/{tag}"
    outj = f"{args.out_results_dir}/{tag}.json"
    # Bash: train ONLY if adapter file missing; eval ONLY if outj missing
    parts = []
    if not Path(f"{cktgz}/pytorch_lora_weights.safetensors").exists():
        parts.append(" ".join(_train_cmd(args, arm, lr, seed)))
    if not Path(outj).exists():
        parts.append(" ".join(_eval_cmd(args, arm, lr, seed)))
    if not parts:
        return None
    return " && ".join(parts)


def main():
    args = parse_args()
    Path(args.out_ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_results_dir).mkdir(parents=True, exist_ok=True)

    jobs = list(product(args.arms, args.lr_list, args.seeds))
    gpus = args.gpus.split(",")
    print(f"[sweep] {len(jobs)} jobs across GPUs {gpus}")

    # Queue: launch max len(gpus) at once; on any completion, launch next; write logs
    pending = list(jobs)
    running = {}  # gpu -> (proc, tag)
    t0 = time.time()
    done = 0
    skipped = 0

    while pending or running:
        # Fill idle GPUs
        for g in gpus:
            if g in running:
                continue
            if not pending:
                break
            arm, lr, seed = pending.pop(0)
            cmd = _combined_cmd(args, arm, lr, seed)
            tag = _tag(arm, lr, seed)
            if cmd is None:
                print(f"[sweep] SKIP {tag} (results already present)")
                skipped += 1
                continue
            log = Path("logs") / f"M0.4_{tag}.log"
            log.parent.mkdir(exist_ok=True)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = g
            proc = subprocess.Popen(f"({cmd}) > {log} 2>&1", shell=True, env=env)
            running[g] = (proc, tag)
            print(f"[sweep] launched {tag} on gpu {g} pid={proc.pid}")

        # Poll
        time.sleep(20)
        for g in list(running):
            proc, tag = running[g]
            if proc.poll() is not None:
                rc = proc.returncode
                print(f"[sweep] finished {tag} rc={rc}  elapsed={time.time()-t0:.0f}s")
                del running[g]
                done += 1
    print(f"[sweep] all done: {done} finished, {skipped} skipped, wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
