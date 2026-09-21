"""
Dispatch a list of commands across 4 GPUs in waves of 4 (one per card).

- Each command is a bash string; the script pins CUDA_VISIBLE_DEVICES per slot.
- Waits for each wave before launching the next (no over-scheduling on a card).
- Logs stdout+stderr per job into runs/<run_id>/log.txt
- Emits runs/<run_id>/cost.json with gpu_ids used, start/end/duration.
- Exits nonzero if any job failed.

Usage:
    python scripts/dispatch_wave.py --manifest jobs.jsonl [--gpus 0,1,2,3]

Each line of jobs.jsonl:
    {"run_id": "M0.7_treated_lr1e-4_seed42", "cmd": "python scripts/student_sft.py ..."}
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from common import ROOT


def _sh(run_id: str, cmd: str, gpu_id: int, run_root: Path) -> subprocess.Popen:
    run_dir = run_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "log.txt"
    # Write run.sh for reproducibility
    (run_dir / "run.sh").write_text(
        f"#!/bin/bash\nset -e\nexport CUDA_VISIBLE_DEVICES={gpu_id}\n{cmd}\n"
    )
    (run_dir / "run.sh").chmod(0o755)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # Persist start metadata
    meta = {"run_id": run_id, "cmd": cmd, "gpu_id": gpu_id,
            "start_ts": time.time()}
    (run_dir / "start.json").write_text(json.dumps(meta, indent=2))
    print(f"[dispatch] launching {run_id} on GPU {gpu_id}", flush=True)
    proc = subprocess.Popen(
        cmd, shell=True, env=env, cwd=str(ROOT),
        stdout=open(log_path, "wb"), stderr=subprocess.STDOUT,
    )
    return proc


def _finalize(run_id: str, run_root: Path, gpu_id: int,
              exit_code: int, wall_s: float) -> None:
    run_dir = run_root / run_id
    cost = {
        "run_id": run_id,
        "gpu_ids": [gpu_id],
        "start_ts": json.loads((run_dir / "start.json").read_text())["start_ts"],
        "end_ts": time.time(),
        "wall_sec": wall_s,
        "gpu_hours": (wall_s / 3600.0),
        "exit_code": exit_code,
        "status": "done" if exit_code == 0 else "failed",
    }
    (run_dir / "cost.json").write_text(json.dumps(cost, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=str, required=True,
                    help="jsonl of {run_id, cmd} entries")
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument("--runs_root", type=str, default=str(ROOT / "runs"))
    args = ap.parse_args()

    gpus = [int(g) for g in args.gpus.split(",") if g.strip()]
    wave_size = len(gpus)
    jobs = []
    with open(args.manifest) as f:
        for line in f:
            line = line.strip()
            if line:
                jobs.append(json.loads(line))
    if not jobs:
        print("[dispatch] no jobs — nothing to do", flush=True)
        return

    run_root = Path(args.runs_root)
    run_root.mkdir(parents=True, exist_ok=True)
    n_fail = 0
    for wave_i in range(0, len(jobs), wave_size):
        wave = jobs[wave_i:wave_i + wave_size]
        procs = []
        starts = []
        for slot_i, job in enumerate(wave):
            gpu_id = gpus[slot_i]
            starts.append((job["run_id"], gpu_id, time.time()))
            p = _sh(job["run_id"], job["cmd"], gpu_id, run_root)
            procs.append(p)
        # Wait for all
        for slot_i, p in enumerate(procs):
            ec = p.wait()
            run_id, gpu_id, t0 = starts[slot_i]
            wall = time.time() - t0
            _finalize(run_id, run_root, gpu_id, ec, wall)
            if ec != 0:
                n_fail += 1
                print(f"[dispatch] {run_id} FAILED (exit={ec}) after {wall:.0f}s",
                      flush=True)
            else:
                print(f"[dispatch] {run_id} OK ({wall:.0f}s)", flush=True)
    print(f"[dispatch] {len(jobs)} jobs total, {n_fail} failed", flush=True)
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
