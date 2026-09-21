#!/usr/bin/env python3
"""M7 — portability check on DeepSeek-R1-Distill-Llama-8B (nice-to-have; seed for /auto-verify).

Runs a scaled-down version of M2 + M3 + M4 (single-site LEACE, alpha in {-2,0,+2})
on the swap model.  Reuses the same DG-1000 dataset (400 baseline trials + partners).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",
                    default="/data/zhenqian/models/DeepSeek-R1-Distill-Llama-8B")
    ap.add_argument("--data", default="data/dg1000_prompts.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_trials", type=int, default=400)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    for sub in ["m2", "m3", "m4", "m5"]:
        Path(os.path.join(args.out, sub)).mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    def run(cmd, label):
        log = os.path.join(args.out, f"{label}.log")
        print(f"[m7] >> {label}: {' '.join(cmd)}", flush=True)
        with open(log, "w") as lf:
            proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT)
        return proc.returncode == 0

    ok = True
    if ok:
        cmd = [
            sys.executable, os.path.join(HERE, "m2_extract_and_probe.py"),
            "--model", args.model,
            "--data", args.data,
            "--out", os.path.join(args.out, "m2"),
            "--batch_size", str(args.batch_size),
            "--every_k_layer", "2",
            "--n_sample", str(args.n_trials),
        ]
        ok = run(cmd, "m2")
    if ok:
        cmd = [
            sys.executable, os.path.join(HERE, "m3_decorrelate.py"),
            "--raw", os.path.join(args.out, "m2", "directions_raw.pt"),
            "--train_acts", os.path.join(args.out, "m2", "train_activations.pt"),
            "--held_acts", os.path.join(args.out, "m2", "heldout_activations.pt"),
            "--layer_pick", os.path.join(args.out, "m2", "layer_pick.json"),
            "--out", os.path.join(args.out, "m3"),
            "--methods", "gs,leace",
        ]
        ok = run(cmd, "m3")
    if ok:
        cmd = [
            sys.executable, os.path.join(HERE, "m4_steer_and_m5_selectivity.py"),
            "--model", args.model,
            "--pure_dir", os.path.join(args.out, "m3", "directions_pure.pt"),
            "--raw_dir", os.path.join(args.out, "m2", "directions_raw.pt"),
            "--layer_pick", os.path.join(args.out, "m2", "layer_pick.json"),
            "--held_acts", os.path.join(args.out, "m2", "heldout_activations.pt"),
            "--data", args.data,
            "--out_m4", os.path.join(args.out, "m4"),
            "--out_m5", os.path.join(args.out, "m5"),
            "--alpha_grid=-2,0,2",
            "--decorrs", "leace",
            "--sites", "single",
            "--batch_size", str(args.batch_size),
            "--n_sample", str(args.n_trials // 5),  # 80 held-out for M4
        ]
        ok = run(cmd, "m4_m5")

    report = dict(
        model=args.model,
        n_trials=args.n_trials,
        wall_time_s=time.time() - t0,
        success=ok,
    )
    with open(os.path.join(args.out, "portability_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
