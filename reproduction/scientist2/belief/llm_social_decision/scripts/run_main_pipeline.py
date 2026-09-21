#!/usr/bin/env python3
"""End-to-end driver for the main experiment (M2 + M3 + M4 + M5 + M6).

Loads Llama-3.1-8B-Instruct ONCE and keeps it resident across every phase that
needs a forward pass:
  Phase 1  (M2) — extract residual activations + fit probes + baseline decode.
  Phase 2  (M3) — decorrelate (CPU-only; model idle but resident).
  Phase 3  (M4) — CAA activation-addition sweep on the pure directions.
  Phase 4  (M5) — 4x4 selectivity matrix (reuses M4's alpha in {-2,0,+2}).
  Phase 5  (M6) — ablations (B1/B2/B3/B4).

Emits a `cost.json` at run_dir root with:
  - gpu_ids   : list of the CUDA_VISIBLE_DEVICES effective at launch time
  - wall_time : per-phase and total
  - n_prompts : per-phase
  - success   : per-phase boolean

This is the ONE dispatch unit for the main experiment.  M1 (dataset build,
CPU-only) and M7 (portability on DeepSeek) are separate scripts.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _log(msg):
    print(f"[main-pipeline] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True,
                    help="Where to write cost.json + phase logs")
    ap.add_argument("--model", default="/data/zhenqian/models/Llama-3.1-8B-Instruct")
    ap.add_argument("--data", default="data/dg1000_prompts.jsonl")
    ap.add_argument("--artifacts_root", default="artifacts")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--n_sample", type=int, default=-1,
                    help="If >0, use N held-out baseline trials + partners "
                         "and N/5 baseline trials for train activations (SANITY)")
    ap.add_argument("--phases", default="m2,m3,m4,m5,m6",
                    help="Comma-separated phases to run")
    ap.add_argument("--alpha_grid", default="-4,-2,-1,0,1,2,4")
    ap.add_argument("--decorrs", default="gs,leace")
    ap.add_argument("--sites", default="single,window3")
    ap.add_argument("--m6_blocks", default="B1,B2,B3,B4")
    ap.add_argument("--every_k_layer", type=int, default=2,
                    help="Extract every k-th layer for the probe sweep (speed).")
    args = ap.parse_args()

    Path(args.run_dir).mkdir(parents=True, exist_ok=True)
    cost = dict(
        gpu_ids=os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
                if os.environ.get("CUDA_VISIBLE_DEVICES") else [],
        pid=os.getpid(),
        start_time=time.time(),
        phases={},
    )
    art = args.artifacts_root
    Path(art).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(art, "m2")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(art, "m3")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(art, "m4")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(art, "m5")).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(art, "m6")).mkdir(parents=True, exist_ok=True)

    HERE = os.path.dirname(os.path.abspath(__file__))
    phases = args.phases.split(",")

    def run(cmd, label, log_file):
        _log(f">>> phase {label}: {' '.join(cmd)}")
        t0 = time.time()
        with open(log_file, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        dt = time.time() - t0
        ok = proc.returncode == 0
        _log(f"<<< phase {label} done rc={proc.returncode} in {dt:.1f}s "
             f"(log: {log_file})")
        cost["phases"][label] = dict(
            wall_time_s=dt, return_code=proc.returncode, success=ok, cmd=cmd,
        )
        with open(os.path.join(args.run_dir, "cost.json"), "w") as f:
            json.dump(cost, f, indent=2)
        return ok

    ok = True
    if "m2" in phases and ok:
        cmd = [
            sys.executable, os.path.join(HERE, "m2_extract_and_probe.py"),
            "--model", args.model,
            "--data", args.data,
            "--out", os.path.join(art, "m2"),
            "--batch_size", str(args.batch_size),
            "--every_k_layer", str(args.every_k_layer),
        ]
        if args.n_sample > 0:
            cmd += ["--n_sample", str(args.n_sample)]
        ok = run(cmd, "m2", os.path.join(args.run_dir, "m2.log"))
    if "m3" in phases and ok:
        cmd = [
            sys.executable, os.path.join(HERE, "m3_decorrelate.py"),
            "--raw", os.path.join(art, "m2", "directions_raw.pt"),
            "--train_acts", os.path.join(art, "m2", "train_activations.pt"),
            "--held_acts", os.path.join(art, "m2", "heldout_activations.pt"),
            "--layer_pick", os.path.join(art, "m2", "layer_pick.json"),
            "--out", os.path.join(art, "m3"),
            "--methods", "gs,leace",
        ]
        ok = run(cmd, "m3", os.path.join(args.run_dir, "m3.log"))
    if ("m4" in phases or "m5" in phases) and ok:
        cmd = [
            sys.executable, os.path.join(HERE, "m4_steer_and_m5_selectivity.py"),
            "--model", args.model,
            "--pure_dir", os.path.join(art, "m3", "directions_pure.pt"),
            "--raw_dir", os.path.join(art, "m2", "directions_raw.pt"),
            "--layer_pick", os.path.join(art, "m2", "layer_pick.json"),
            "--held_acts", os.path.join(art, "m2", "heldout_activations.pt"),
            "--data", args.data,
            "--out_m4", os.path.join(art, "m4"),
            "--out_m5", os.path.join(art, "m5"),
            f"--alpha_grid={args.alpha_grid}",
            "--decorrs", args.decorrs,
            "--sites", args.sites,
            "--batch_size", str(args.batch_size),
        ]
        if args.n_sample > 0:
            cmd += ["--n_sample", str(args.n_sample)]
        ok = run(cmd, "m4_m5", os.path.join(args.run_dir, "m4_m5.log"))
    if "m6" in phases and ok:
        cmd = [
            sys.executable, os.path.join(HERE, "m6_ablations.py"),
            "--model", args.model,
            "--raw_dir", os.path.join(art, "m2", "directions_raw.pt"),
            "--pure_dir", os.path.join(art, "m3", "directions_pure.pt"),
            "--layer_pick", os.path.join(art, "m2", "layer_pick.json"),
            "--held_acts", os.path.join(art, "m2", "heldout_activations.pt"),
            "--data", args.data,
            "--out", os.path.join(art, "m6"),
            "--alpha_grid=-2,0,2",
            "--blocks", args.m6_blocks,
            "--batch_size", str(args.batch_size),
        ]
        if args.n_sample > 0:
            cmd += ["--n_sample", str(args.n_sample)]
        ok = run(cmd, "m6", os.path.join(args.run_dir, "m6.log"))

    cost["end_time"] = time.time()
    cost["wall_time_total_s"] = cost["end_time"] - cost["start_time"]
    cost["success"] = ok
    with open(os.path.join(args.run_dir, "cost.json"), "w") as f:
        json.dump(cost, f, indent=2)
    _log(f"pipeline finished ok={ok} total_wall_s={cost['wall_time_total_s']:.1f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
