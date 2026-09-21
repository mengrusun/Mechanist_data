"""Top-level driver that runs a chosen milestone.

Usage:
  python run_all.py --milestone {sanity|M1_generation|M1_extract|M1.5|M1_probes|M2|M3|M4_M6_M7}

This is a thin orchestrator that shells out to the step_*.py scripts in order.
Each milestone is idempotent (uses --resume where the individual scripts support it).
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parent


def run(cmd, extra_env=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    print(f"[run_all] $ {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd, cwd=str(CODE_DIR.parent), env=env)
    if rc != 0:
        sys.exit(rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--milestone", type=str, required=True,
                    choices=["sanity", "M1_generation", "M1_extract", "M1.5",
                             "M1_probes", "M2", "M3", "M4_M6_M7", "M5_paraphrase"])
    ap.add_argument("--n-total", type=int, default=10000)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-dev", type=int, default=2000)
    ap.add_argument("--n-test", type=int, default=2000)
    ap.add_argument("--n-dev-p", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--n-steer", type=int, default=500)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=1024)
    ap.add_argument("--swap-space", type=int, default=4)
    args = ap.parse_args()

    py = sys.executable
    common_gen = [
        py, "code/step1_generate.py",
        "--n-total", str(args.n_total),
        "--n-train", str(args.n_train), "--n-dev", str(args.n_dev), "--n-test", str(args.n_test),
        "--n-dev-p", str(args.n_dev_p),
        "--tensor-parallel-size", str(args.tp),
        "--gpu-mem-util", str(args.gpu_mem_util),
        "--max-model-len", str(args.max_model_len),
        "--swap-space", str(args.swap_space),
    ]
    if args.resume:
        common_gen.append("--resume")

    if args.milestone == "sanity":
        # Tiny slice: 100 total (60/20/20 split), turn1 only + hidden extract at layer 15 only
        run(common_gen + ["--n-total", "100", "--n-train", "60", "--n-dev", "20", "--n-test", "20",
                          "--which", "turn1,turn2,single"])
        # Extract hidden states for turn1 to sanity-check pipeline
        for mode in ("turn1", "turn2", "single"):
            cmd = [py, "code/step2_extract_hidden.py", "--mode", mode, "--batch-size", "8"]
            if args.resume:
                cmd.append("--resume")
            run(cmd)
        # Score + variance diagnostic
        run([py, "code/step3_variance.py"])
        # Fit probes only at every 4 layers
        run([py, "code/step4_probes.py", "--n-bootstrap", "100", "--layers", "all", "--n-random-null", "20"])
        return

    if args.milestone == "M1_generation":
        run(common_gen + ["--which", "turn1,turn2,single"])
        return

    if args.milestone == "M1_extract":
        for mode in ("turn1", "turn2", "single"):
            cmd = [py, "code/step2_extract_hidden.py", "--mode", mode, "--batch-size", str(args.batch_size)]
            if args.resume:
                cmd.append("--resume")
            run(cmd)
        return

    if args.milestone == "M1.5":
        run([py, "code/step3_variance.py"])
        return

    if args.milestone == "M1_probes":
        run([py, "code/step4_probes.py", "--n-bootstrap", str(args.n_bootstrap)])
        return

    if args.milestone == "M2":
        # Cosine trajectory is computed inside step4_probes.py; nothing to add unless
        # we want a separate M2 pass. Kept for symmetry.
        print("[run_all] M2 metrics already produced by M1_probes step.")
        return

    if args.milestone == "M3":
        run([py, "code/step5_steering.py", "--n-steer", str(args.n_steer),
             "--batch-size", str(args.batch_size)])
        return

    if args.milestone == "M5_paraphrase":
        run(common_gen + ["--which", "turn2_p1,turn2_p2"])
        for mode in ("turn2_p1", "turn2_p2"):
            cmd = [py, "code/step2_extract_hidden.py", "--mode", mode, "--batch-size", str(args.batch_size)]
            if args.resume:
                cmd.append("--resume")
            run(cmd)
        return

    if args.milestone == "M4_M6_M7":
        run([py, "code/step6_analyses.py"])
        return


if __name__ == "__main__":
    main()
