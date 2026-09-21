#!/usr/bin/env python3
"""Stage 0 step 2 — ESMFold baseline forward worker.

Reads a slice of `native_candidates.jsonl` (via --start and --stop indices) and runs
ESMFold on each chain with 1- and 2-recycle passes, applies DSSP, keeps chains whose
baseline_hairpin_rate ≥ 0.7 (i.e. hairpin in both passes).

Writes `data/prepared/esmfold_accepted_worker_<id>.jsonl` and
       `data/prepared/esmfold_rejects_worker_<id>.jsonl`

One worker per GPU; 4 workers dispatched in parallel by the driver.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))

from esmfold_lib import (  # noqa: E402
    PREPARED_DIR, load_esmfold, predict_and_judge_hairpin,
    find_cross_strand_pairs_from_coords, set_seed, write_jsonl, write_json,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker-id", type=int, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--candidates", type=Path,
                   default=PREPARED_DIR / "native_candidates.jsonl")
    p.add_argument("--target-accepted", type=int, default=125,
                   help="Stop after accepting this many chains (default = 500/4).")
    p.add_argument("--out", type=Path, default=PREPARED_DIR)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed + args.worker_id)

    # Read candidates slice
    with open(args.candidates) as f:
        all_c = [json.loads(l) for l in f if l.strip()]
    my_slice = all_c[args.start:args.stop]
    print(f"[worker-{args.worker_id}] {len(my_slice)} candidates in slice "
          f"[{args.start},{args.stop})", flush=True)
    print(f"[worker-{args.worker_id}] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
          flush=True)

    print(f"[worker-{args.worker_id}] loading model...", flush=True)
    model, tok, _ = load_esmfold(device="cuda", dtype=torch.float32)
    print(f"[worker-{args.worker_id}] model loaded", flush=True)

    accepted = []
    rejects = []
    t0 = time.time()
    for i, cand in enumerate(my_slice):
        if i % 10 == 0 and i > 0:
            elapsed = time.time() - t0
            rate = elapsed / i
            eta = rate * (len(my_slice) - i) / 60
            print(f"[worker-{args.worker_id}] {i}/{len(my_slice)}  accepted={len(accepted)}  "
                  f"{rate:.1f}s/chain  eta={eta:.1f}min", flush=True)
        seq = cand["seq"]
        try:
            hits = 0
            plddt_vals = []
            ca_ref = None
            ok = True
            for n_rec in (1, 2):
                res = predict_and_judge_hairpin(model, tok, seq,
                                                cand["target_start"], cand["target_end"],
                                                no_recycles=n_rec, return_extras=(n_rec == 2))
                if not res["dssp_ok"]:
                    rejects.append({"pdb_id": cand["pdb_id"], "chain_id": cand["chain_id"],
                                    "reason": "esmfold_dssp_failed", "recycles": n_rec})
                    ok = False
                    break
                if res["is_hairpin"]:
                    hits += 1
                plddt_vals.append(res["plddt_target_mean"])
                if n_rec == 2:
                    ca_ref = res.get("ca_coords")
            if not ok:
                continue
            baseline_rate = hits / 2.0
            if baseline_rate < 0.7:
                rejects.append({"pdb_id": cand["pdb_id"], "chain_id": cand["chain_id"],
                                "reason": f"baseline_rate={baseline_rate}"})
                continue
            cross_pairs = []
            if ca_ref is not None and len(ca_ref) >= len(seq):
                cross_pairs = find_cross_strand_pairs_from_coords(
                    ca_ref, (cand["strand1"][0], cand["strand1"][1]),
                    (cand["strand2"][0], cand["strand2"][1]),
                )
            accepted.append({
                "pdb_id": cand["pdb_id"], "chain_id": cand["chain_id"],
                "seq": seq, "len": len(seq),
                "target_start": cand["target_start"], "target_end": cand["target_end"],
                "strand1": cand["strand1"], "strand2": cand["strand2"],
                "native_ss": cand["native_ss"],
                "baseline_hairpin_rate": baseline_rate,
                "baseline_plddt_target_mean": float(np.mean(plddt_vals)),
                "cross_strand_pairs": cross_pairs,
                "cath_label": None,
            })
        except torch.cuda.OutOfMemoryError:
            print(f"[worker-{args.worker_id}] OOM on {cand['pdb_id']} L={len(seq)}", flush=True)
            torch.cuda.empty_cache()
            rejects.append({"pdb_id": cand["pdb_id"], "chain_id": cand["chain_id"],
                            "reason": "oom", "len": len(seq)})
            continue
        except Exception as e:
            print(f"[worker-{args.worker_id}] error on {cand['pdb_id']}: {e}", flush=True)
            traceback.print_exc()
            rejects.append({"pdb_id": cand["pdb_id"], "chain_id": cand["chain_id"],
                            "reason": f"exception: {e}"})
            continue
        if len(accepted) >= args.target_accepted:
            print(f"[worker-{args.worker_id}] target {args.target_accepted} reached, stopping", flush=True)
            break

    write_jsonl(args.out / f"esmfold_accepted_worker_{args.worker_id}.jsonl", accepted)
    write_jsonl(args.out / f"esmfold_rejects_worker_{args.worker_id}.jsonl", rejects)
    write_json(args.out / f"esmfold_worker_{args.worker_id}_summary.json", {
        "worker_id": args.worker_id, "slice": [args.start, args.stop],
        "n_accepted": len(accepted), "n_rejected": len(rejects),
        "elapsed_sec": time.time() - t0,
    })
    print(f"[worker-{args.worker_id}] DONE. accepted={len(accepted)} rejected={len(rejects)} "
          f"in {(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
