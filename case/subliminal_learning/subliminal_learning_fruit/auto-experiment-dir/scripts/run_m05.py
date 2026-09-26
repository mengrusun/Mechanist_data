"""M0.5 orchestrator: 7-seed reproduction at BEST_LR.

Trains 2 arms × 7 seeds = 14 students, evaluates each on eval_pref160, computes verdict.
Reuses seeds 42/200/201 from M0.4 sweep at BEST_LR by symlink (done externally).
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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--best-lr", required=True)
    p.add_argument("--new-seeds", nargs="+", type=int, default=[300, 301, 400, 401])
    p.add_argument("--arms", nargs="+", default=["teacher", "ctrl"])
    p.add_argument("--gpus", default="4,5,6,7")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--data-dir", default="data/channel_final")
    p.add_argument("--out-ckpt-dir", default="checkpoints/student_final")
    p.add_argument("--out-results-dir", default="results/M0/final")
    p.add_argument("--eval-prompts",
                   default="/data/zhenqian/exp/subliminal/multi_modal_B/data/eval_pref160.txt")
    p.add_argument("--train-res", type=int, default=512)
    p.add_argument("--eval-res", type=int, default=512)
    p.add_argument("--eval-steps", type=int, default=15)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=4)
    return p.parse_args()


def _tag(arm, seed):
    return f"{arm}_seed{seed}"


def _combined_cmd(args, arm, seed):
    tag = _tag(arm, seed)
    ckpt = f"{args.out_ckpt_dir}/{tag}"
    outj = f"{args.out_results_dir}/{tag}.json"
    parts = []
    if not Path(f"{ckpt}/pytorch_lora_weights.safetensors").exists() and not os.path.islink(ckpt):
        parts.append(" ".join([
            "python", "-m", "scripts.train_lora",
            "--data", f"{args.data_dir}/{arm}_channel.jsonl",
            "--data-root", ".",
            "--lora-rank", str(args.lora_rank),
            "--lr", args.best_lr,
            "--epochs", str(args.epochs),
            "--seed", str(seed),
            "--out", ckpt,
            "--resolution", str(args.train_res),
            "--batch", str(args.batch),
            "--grad-accum", str(args.grad_accum),
            "--log-every", "10",
        ]))
    if not Path(outj).exists():
        parts.append(" ".join([
            "python", "-m", "scripts.eval_student",
            "--lora", ckpt,
            "--prompts", args.eval_prompts,
            "--out", outj,
            "--seed", str(seed),
            "--height", str(args.eval_res),
            "--width", str(args.eval_res),
            "--num-inference-steps", str(args.eval_steps),
            "--tag", tag,
        ]))
    if not parts:
        return None
    return " && ".join(parts)


def main():
    args = parse_args()
    Path(args.out_ckpt_dir).mkdir(parents=True, exist_ok=True)
    Path(args.out_results_dir).mkdir(parents=True, exist_ok=True)

    jobs = list(product(args.arms, args.new_seeds))
    gpus = args.gpus.split(",")
    print(f"[m0.5] {len(jobs)} jobs across GPUs {gpus}")

    pending = list(jobs)
    running = {}
    t0 = time.time()
    done = 0
    skipped = 0

    while pending or running:
        for g in gpus:
            if g in running:
                continue
            if not pending:
                break
            arm, seed = pending.pop(0)
            cmd = _combined_cmd(args, arm, seed)
            tag = _tag(arm, seed)
            if cmd is None:
                print(f"[m0.5] SKIP {tag} (already present)")
                skipped += 1
                continue
            log = Path("logs") / f"M0.5_{tag}.log"
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = g
            proc = subprocess.Popen(f"({cmd}) > {log} 2>&1", shell=True, env=env)
            running[g] = (proc, tag)
            print(f"[m0.5] launched {tag} on gpu {g} pid={proc.pid}", flush=True)

        time.sleep(20)
        for g in list(running):
            proc, tag = running[g]
            if proc.poll() is not None:
                print(f"[m0.5] finished {tag} rc={proc.returncode} elapsed={time.time()-t0:.0f}s", flush=True)
                del running[g]
                done += 1
    print(f"[m0.5] all done: {done} finished {skipped} skipped wall={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
