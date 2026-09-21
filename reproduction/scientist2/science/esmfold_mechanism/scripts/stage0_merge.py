#!/usr/bin/env python3
"""Stage 0 step 3 — merge worker outputs, split into main / calib / donor / heldout.

Reads all `esmfold_accepted_worker_*.jsonl`, shuffles deterministically, splits into
main / calib / donor / heldout_probe, assigns donors, writes manifest.jsonl.
"""
from __future__ import annotations
import argparse
import glob
import json
import random
import sys
from pathlib import Path
from typing import List, Dict

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))

from esmfold_lib import PREPARED_DIR, set_seed, write_jsonl, write_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-main", type=int, default=200)
    p.add_argument("--n-calib", type=int, default=50)
    p.add_argument("--n-donor", type=int, default=100)
    p.add_argument("--n-heldout-probe", type=int, default=250)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=PREPARED_DIR)
    args = p.parse_args()

    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # Load all worker outputs
    worker_files = sorted(glob.glob(str(args.out / "esmfold_accepted_worker_*.jsonl")))
    print(f"[stage0/merge] found {len(worker_files)} worker files", flush=True)
    accepted = []
    for wf in worker_files:
        with open(wf) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                accepted.append(json.loads(line))
    # De-duplicate by (pdb_id, chain_id)
    seen = set()
    unique = []
    for r in accepted:
        key = (r["pdb_id"], r["chain_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(r)
    print(f"[stage0/merge] {len(unique)} unique chains after de-dup", flush=True)
    accepted = unique

    random.shuffle(accepted)
    need = args.n_main + args.n_calib + args.n_donor + args.n_heldout_probe
    if len(accepted) < need:
        s = len(accepted) / need
        n_main = max(20, int(args.n_main * s))
        n_calib = max(5, int(args.n_calib * s))
        n_donor = max(20, int(args.n_donor * s))
        n_hp = len(accepted) - n_main - n_calib - n_donor
        if n_hp < 0:
            n_hp = 0
        print(f"[stage0/merge] WARN: only {len(accepted)} chains, needed {need}. "
              f"Shrunk to main={n_main}, calib={n_calib}, donor={n_donor}, heldout={n_hp}",
              flush=True)
    else:
        n_main, n_calib, n_donor = args.n_main, args.n_calib, args.n_donor
        n_hp = min(args.n_heldout_probe, len(accepted) - n_main - n_calib - n_donor)

    main_set = accepted[:n_main]
    calib_set = accepted[n_main:n_main + n_calib]
    donor_set = accepted[n_main + n_calib:n_main + n_calib + n_donor]
    heldout_set = accepted[n_main + n_calib + n_donor:n_main + n_calib + n_donor + n_hp]

    for r in main_set: r["split"] = "main"
    for r in calib_set: r["split"] = "calibration"
    for r in donor_set: r["split"] = "donor"
    for r in heldout_set: r["split"] = "heldout_probe"

    def pick_donors(target_r: Dict, k: int = 3) -> List[str]:
        tgt_len = target_r["len"]
        cands = [(d["pdb_id"], d["chain_id"], d["len"])
                 for d in donor_set
                 if abs(d["len"] - tgt_len) <= 0.20 * tgt_len
                 and (d["pdb_id"], d["chain_id"]) != (target_r["pdb_id"], target_r["chain_id"])]
        if len(cands) < k:
            cands = [(d["pdb_id"], d["chain_id"], d["len"]) for d in donor_set
                     if (d["pdb_id"], d["chain_id"]) != (target_r["pdb_id"], target_r["chain_id"])]
        random.shuffle(cands)
        return [f"{pid}_{cid}" for (pid, cid, _) in cands[:k]]

    for r in main_set + calib_set:
        r["donor_ids"] = pick_donors(r, k=3)

    all_records = main_set + calib_set + donor_set + heldout_set
    write_jsonl(args.out / "manifest.jsonl", all_records)
    write_json(args.out / "stage0_summary.json", {
        "n_total_unique": len(accepted),
        "n_main": len(main_set), "n_calib": len(calib_set),
        "n_donor": len(donor_set), "n_heldout_probe": len(heldout_set),
        "seed": args.seed, "manifest_path": str(args.out / "manifest.jsonl"),
    })
    print("[stage0/merge] DONE.", flush=True)
    print(json.dumps({
        "n_main": len(main_set), "n_calib": len(calib_set),
        "n_donor": len(donor_set), "n_heldout_probe": len(heldout_set),
    }, indent=2))


if __name__ == "__main__":
    main()
