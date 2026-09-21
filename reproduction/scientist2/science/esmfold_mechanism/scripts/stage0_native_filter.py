#!/usr/bin/env python3
"""Stage 0 step 1 — CPU-only native DSSP filter.

Reads PISCES, applies length filter, runs mkdssp on native structures, keeps only
chains that contain ≥1 candidate hairpin per native DSSP, picks the longest hairpin
as target-region. Writes `data/prepared/native_candidates.jsonl` for downstream
ESMFold workers to consume.

Fast: ~10 min for 12k chains on CPU (DSSP-bound).
"""
from __future__ import annotations
import argparse
import json
import random
import sys
import time
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))

from esmfold_lib import (  # noqa: E402
    PISCES_LIST, PDB_CACHE, PREPARED_DIR,
    run_mkdssp, parse_dssp_ss, ss_string_from_dssp,
    find_hairpins, write_jsonl, write_json, set_seed,
)
from stage0_prepare import (  # noqa: E402
    read_pisces, parse_native_pdb, dssp_on_native_pdb, pick_longest_hairpin,
)


def process_one(row: Tuple[str, str, int, int, int]) -> Tuple[Optional[Dict], Optional[Dict]]:
    """Return (accepted_record, reject_record). Exactly one is non-None."""
    pdb_id, chain_id, L, min_len, max_len = row
    parsed = parse_native_pdb(pdb_id, chain_id)
    if parsed is None:
        return None, {"pdb_id": pdb_id, "chain_id": chain_id, "reason": "pdb_missing"}
    seq_native, ca_native = parsed
    if not (min_len <= len(seq_native) <= max_len):
        return None, {"pdb_id": pdb_id, "chain_id": chain_id, "reason": "len_out",
                      "len": len(seq_native)}
    native_ss = dssp_on_native_pdb(pdb_id, chain_id)
    if native_ss is None:
        return None, {"pdb_id": pdb_id, "chain_id": chain_id, "reason": "native_dssp_failed"}
    if len(native_ss) < len(seq_native):
        native_ss = native_ss + "C" * (len(seq_native) - len(native_ss))
    elif len(native_ss) > len(seq_native):
        native_ss = native_ss[:len(seq_native)]
    pick = pick_longest_hairpin(native_ss)
    if pick is None:
        return None, {"pdb_id": pdb_id, "chain_id": chain_id, "reason": "no_native_hairpin"}
    target_start, target_end, (s1a, s1b, s2a, s2b) = pick
    return {
        "pdb_id": pdb_id, "chain_id": chain_id, "len": len(seq_native),
        "seq": seq_native, "native_ss": native_ss,
        "target_start": target_start, "target_end": target_end,
        "strand1": [s1a, s1b], "strand2": [s2a, s2b],
    }, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--min-len", type=int, default=60)
    p.add_argument("--max-len", type=int, default=300)
    p.add_argument("--max-candidates", type=int, default=1500,
                   help="cap on candidates passed downstream to ESMFold")
    p.add_argument("--nproc", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=PREPARED_DIR)
    args = p.parse_args()

    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[stage0/native] reading PISCES from {PISCES_LIST}", flush=True)
    all_chains = read_pisces()
    print(f"[stage0/native] {len(all_chains)} PISCES chains", flush=True)
    chains_len = [(p, c, L, args.min_len, args.max_len)
                  for p, c, L in all_chains if args.min_len <= L <= args.max_len]
    print(f"[stage0/native] {len(chains_len)} after length filter", flush=True)
    random.shuffle(chains_len)

    accepted = []
    rejects = []
    t0 = time.time()
    with mp.Pool(args.nproc) as pool:
        for i, (a, r) in enumerate(pool.imap_unordered(process_one, chains_len, chunksize=4)):
            if a is not None:
                accepted.append(a)
            if r is not None:
                rejects.append(r)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"[stage0/native] processed {i+1}/{len(chains_len)}  "
                      f"accepted={len(accepted)}  ({elapsed:.0f}s)", flush=True)
            if len(accepted) >= args.max_candidates:
                # Have enough; drop the rest of the pool.
                pool.terminate()
                break

    # Reduce to at most --max-candidates (already the case unless caught mid-batch).
    accepted = accepted[:args.max_candidates]
    print(f"[stage0/native] final accepted={len(accepted)}  rejected={len(rejects)}  "
          f"({time.time()-t0:.0f}s)", flush=True)

    write_jsonl(args.out / "native_candidates.jsonl", accepted)
    write_jsonl(args.out / "native_rejects.jsonl", rejects[:2000])  # cap log size
    write_json(args.out / "native_summary.json", {
        "n_pisces": len(all_chains), "n_len_filtered": len(chains_len),
        "n_native_candidates": len(accepted), "n_rejects_sampled": len(rejects[:2000]),
        "min_len": args.min_len, "max_len": args.max_len,
        "seed": args.seed, "max_candidates": args.max_candidates,
    })


if __name__ == "__main__":
    main()
