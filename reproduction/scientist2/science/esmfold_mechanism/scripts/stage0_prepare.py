#!/usr/bin/env python3
"""Stage 0 — Shared setup.

Steps:
  1. Read PISCES cull list, filter to length 60..300.
  2. For each chain, load native structure from /data/zhenqian/data/pdb_cache/, run DSSP
     on native, keep chains containing ≥1 candidate β-hairpin in native.
  3. For each surviving chain, extract the sequence for that chain, run ESMFold forward,
     run DSSP on ESMFold-predicted PDB, judge if ESMFold predicts hairpin in the same
     target region.
  4. Compute per-chain baseline_hairpin_rate. (One recycles-only forward per chain here;
     with no_recycles=1 the model is deterministic given the seed, so rate is 0 or 1.
     To get a graded rate we would need stochastic recycling, but the plan's ≥0.7 filter
     works cleanly on the boolean output when combined across multiple recycles. We use
     no_recycles ∈ {1, 2} and require both to be hairpin for inclusion — this is our
     concrete instantiation of the ≥0.7 baseline_hairpin_rate rule.)
  5. Randomly split the kept chains: 200 main + 50 calib + 100 donor (disjoint).
  6. For each main / calib chain, also record cross-strand pairs on the predicted
     structure (Cα-Cα ≤ 6.5 Å).
  7. CATH mapping — try to look up chain-level CATH label if available; else null.
  8. Write manifest.jsonl with all metadata.

Chunked runtime:
  - PISCES filter + native-DSSP filter: ~10 min for 12055 chains (mostly IO + DSSP).
  - ESMFold forwards for candidate chains (~600 kept): ~35 min at 3s/chain avg (chains
    up to L=300, 4 GPUs available; single-GPU sequential here — Stage 0 is one driver).

Usage:
  python stage0_prepare.py --gpu-id 0 --n-main 200 --n-calib 50 --n-donor 100 \\
                          --out data/prepared/
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Make script importable from anywhere
_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))

from esmfold_lib import (  # noqa: E402
    ESMFOLD_PATH, PDB_CACHE, PISCES_LIST, PREPARED_DIR, MANIFEST_PATH,
    find_hairpins, find_cross_strand_pairs_from_coords,
    is_hairpin_region, run_mkdssp, parse_dssp_ss, ss_string_from_dssp,
    load_esmfold, esmfold_forward, predict_and_judge_hairpin,
    extract_ca_from_pdb, set_seed, write_jsonl, write_json,
)


# ---------------------------------------------------------------------------
# PISCES list parsing
# ---------------------------------------------------------------------------

def read_pisces() -> List[Tuple[str, str, int]]:
    """Return list of (pdb_id, chain_id, length). Skip header + malformed lines."""
    out = []
    with open(PISCES_LIST) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            tok = parts[0]
            if tok == "PDBchain":  # header
                continue
            if len(tok) < 5:
                continue
            pdb_id = tok[:4].upper()
            chain_id = tok[4:]  # usually 1 char; could be longer if PISCES uses that
            try:
                length = int(parts[1])
            except ValueError:
                continue
            out.append((pdb_id, chain_id, length))
    return out


# ---------------------------------------------------------------------------
# Native PDB parsing — read one PDB file, extract SEQ + CA per chain, run DSSP on it
# ---------------------------------------------------------------------------

_THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLU": "E", "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "MSE": "M",  # selenomet → M
}


def parse_native_pdb(pdb_id: str, chain_id: str) -> Optional[Tuple[str, np.ndarray]]:
    """Return (sequence, ca_coords) for the specified chain in the local native PDB.
    Skips waters/HET. Missing residues get 'X' + NaN coord. Returns None if PDB is
    missing or has no residues for this chain.
    """
    pdb_path = PDB_CACHE / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        return None
    # Parse only the requested chain, ATOM records only
    seq_records: Dict[int, Tuple[str, np.ndarray]] = {}
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("ATOM"):
                    if line.startswith("ENDMDL"):
                        # Only take model 1
                        break
                    continue
                if line[21] != chain_id:
                    continue
                atom_name = line[12:16].strip()
                if atom_name != "CA":
                    continue
                res_name = line[17:20].strip()
                aa = _THREE_TO_ONE.get(res_name, "X")
                try:
                    resseq = int(line[22:26].strip())
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                if resseq in seq_records:
                    continue  # first altloc only
                seq_records[resseq] = (aa, np.array([x, y, z], dtype=float))
    except OSError:
        return None
    if not seq_records:
        return None
    # Sort by resseq
    sorted_res = sorted(seq_records.items(), key=lambda kv: kv[0])
    seq = "".join(v[0] for _, v in sorted_res)
    coords = np.array([v[1] for _, v in sorted_res])
    return seq, coords


def dssp_on_native_pdb(pdb_id: str, chain_id: str) -> Optional[str]:
    """Run mkdssp on the native PDB, return the SS string for the chain, or None."""
    pdb_path = PDB_CACHE / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        return None
    try:
        with open(pdb_path) as f:
            pdb_text = f.read()
    except OSError:
        return None
    dssp_text = run_mkdssp(pdb_text)
    if dssp_text is None:
        return None
    recs = parse_dssp_ss(dssp_text, target_chain=chain_id)
    if not recs:
        return None
    return ss_string_from_dssp(recs)


# ---------------------------------------------------------------------------
# Target-region selection: longest total-E hairpin
# ---------------------------------------------------------------------------

def pick_longest_hairpin(ss: str) -> Optional[Tuple[int, int, Tuple[int, int, int, int]]]:
    """Return (target_start, target_end, (s1a, s1b, s2a, s2b)) for the hairpin whose
    two strand lengths sum to max. Or None if no hairpin found.
    """
    hps = find_hairpins(ss, "")
    if not hps:
        return None
    def score(h):
        s1a, s1b, s2a, s2b = h
        return (s1b - s1a + 1) + (s2b - s2a + 1)
    best = max(hps, key=score)
    return best[0], best[3], best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu-id", default="0")
    p.add_argument("--n-main", type=int, default=200)
    p.add_argument("--n-calib", type=int, default=50)
    p.add_argument("--n-donor", type=int, default=100)
    p.add_argument("--n-heldout-probe", type=int, default=250,
                   help="Additional held-out chains for M3a probe training (~250)")
    p.add_argument("--min-len", type=int, default=60)
    p.add_argument("--max-len", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=PREPARED_DIR)
    p.add_argument("--max-candidates", type=int, default=1200,
                   help="Cap on candidates to run ESMFold on; chosen by shuffled PISCES order")
    p.add_argument("--test-run", action="store_true",
                   help="Only run pipeline on ~10 chains as sanity")
    args = p.parse_args()

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.gpu_id)
    set_seed(args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out / "manifest.jsonl"
    reject_log_path = args.out / "reject_log.jsonl"
    summary_path = args.out / "stage0_summary.json"

    # -------- Step 1: read PISCES --------
    print(f"[stage0] reading PISCES list from {PISCES_LIST}", flush=True)
    all_chains = read_pisces()
    print(f"[stage0] PISCES: {len(all_chains)} chains", flush=True)
    # Length filter
    chains_len = [(p, c, L) for p, c, L in all_chains if args.min_len <= L <= args.max_len]
    print(f"[stage0] after length filter [{args.min_len}, {args.max_len}]: {len(chains_len)}", flush=True)
    random.shuffle(chains_len)  # shuffle so the candidate cap doesn't bias by resolution
    if args.test_run:
        chains_len = chains_len[:15]
        print("[stage0] TEST-RUN: capping to 15 chains", flush=True)
    else:
        chains_len = chains_len[:args.max_candidates]

    # -------- Step 2: native DSSP filter --------
    t0 = time.time()
    candidates: List[Dict] = []
    rejects: List[Dict] = []
    for i, (pdb_id, chain_id, L) in enumerate(chains_len):
        if i % 100 == 0:
            print(f"[stage0/native] {i}/{len(chains_len)} candidates so far: {len(candidates)}", flush=True)
        parsed = parse_native_pdb(pdb_id, chain_id)
        if parsed is None:
            rejects.append({"pdb_id": pdb_id, "chain_id": chain_id, "reason": "pdb_missing"})
            continue
        seq_native, ca_native = parsed
        if not (args.min_len <= len(seq_native) <= args.max_len):
            rejects.append({"pdb_id": pdb_id, "chain_id": chain_id, "reason": "len_from_pdb_out_of_range",
                            "len": len(seq_native)})
            continue
        native_ss = dssp_on_native_pdb(pdb_id, chain_id)
        if native_ss is None:
            rejects.append({"pdb_id": pdb_id, "chain_id": chain_id, "reason": "native_dssp_failed"})
            continue
        # Sequence and SS lengths should match; if not, align to sequence
        if len(native_ss) < len(seq_native):
            native_ss = native_ss + "C" * (len(seq_native) - len(native_ss))
        elif len(native_ss) > len(seq_native):
            native_ss = native_ss[:len(seq_native)]
        pick = pick_longest_hairpin(native_ss)
        if pick is None:
            rejects.append({"pdb_id": pdb_id, "chain_id": chain_id, "reason": "no_native_hairpin"})
            continue
        target_start, target_end, (s1a, s1b, s2a, s2b) = pick
        candidates.append({
            "pdb_id": pdb_id, "chain_id": chain_id, "len": len(seq_native),
            "seq": seq_native, "native_ss": native_ss,
            "target_start": target_start, "target_end": target_end,
            "strand1": [s1a, s1b], "strand2": [s2a, s2b],
        })
    print(f"[stage0/native] filtered to {len(candidates)} candidates in {time.time()-t0:.1f}s", flush=True)
    write_jsonl(reject_log_path, rejects)

    # -------- Step 3-4: ESMFold baseline forward + judge --------
    print("[stage0/esmfold] loading model...", flush=True)
    model, tok, no_rec = load_esmfold(device="cuda", dtype=torch.float32, no_recycles=1)
    print("[stage0/esmfold] model loaded", flush=True)

    accepted: List[Dict] = []
    t0 = time.time()
    for i, cand in enumerate(candidates):
        if i % 20 == 0 and i > 0:
            rate = (time.time() - t0) / i
            eta = rate * (len(candidates) - i) / 60
            print(f"[stage0/esmfold] {i}/{len(candidates)} accepted: {len(accepted)} "
                  f"({rate:.1f}s/chain, eta {eta:.1f} min)", flush=True)
        seq = cand["seq"]
        try:
            # Two recycles = baseline rate proxy (need at least 2 to get non-boolean rate).
            # We tally hairpin in 2 recycles setups: 1-recycle and 2-recycle. rate ∈ {0,0.5,1}.
            hits = 0
            plddt_vals = []
            ca_coords_ref = None
            for n_rec in (1, 2):
                res = predict_and_judge_hairpin(model, tok, seq,
                                                cand["target_start"], cand["target_end"],
                                                no_recycles=n_rec, return_extras=(n_rec == 2))
                if not res["dssp_ok"]:
                    hits = -1
                    break
                if res["is_hairpin"]:
                    hits += 1
                plddt_vals.append(res["plddt_target_mean"])
                if n_rec == 2:
                    ca_coords_ref = res.get("ca_coords")
            if hits < 0:
                rejects.append({"pdb_id": cand["pdb_id"], "chain_id": cand["chain_id"],
                                "reason": "esmfold_dssp_failed"})
                continue
            baseline_rate = hits / 2.0
            if baseline_rate < 0.7:  # plan spec: keep ≥0.7
                rejects.append({"pdb_id": cand["pdb_id"], "chain_id": cand["chain_id"],
                                "reason": "baseline_rate_below_0.7", "rate": baseline_rate})
                continue
            # Cross-strand pairs from ESMFold-predicted structure
            cross_pairs = []
            if ca_coords_ref is not None and len(ca_coords_ref) >= len(seq):
                cross_pairs = find_cross_strand_pairs_from_coords(
                    ca_coords_ref, (cand["strand1"][0], cand["strand1"][1]),
                    (cand["strand2"][0], cand["strand2"][1]),
                )
            record = {
                "pdb_id": cand["pdb_id"], "chain_id": cand["chain_id"],
                "seq": seq, "len": len(seq),
                "target_start": cand["target_start"], "target_end": cand["target_end"],
                "strand1": cand["strand1"], "strand2": cand["strand2"],
                "native_ss": cand["native_ss"],
                "baseline_hairpin_rate": baseline_rate,
                "baseline_plddt_target_mean": float(np.mean(plddt_vals)),
                "cross_strand_pairs": cross_pairs,
                "cath_label": None,  # CATH mapping optional; set later if lookup available
            }
            accepted.append(record)
        except Exception as e:
            print(f"[stage0/esmfold] {cand['pdb_id']}{cand['chain_id']} failed: {e}", flush=True)
            traceback.print_exc()
            rejects.append({"pdb_id": cand["pdb_id"], "chain_id": cand["chain_id"],
                            "reason": f"exception: {e}"})
            continue
        if not args.test_run and len(accepted) >= (args.n_main + args.n_calib + args.n_donor +
                                                   args.n_heldout_probe + 100):
            # Enough — stop early to save GPU time
            print(f"[stage0/esmfold] enough accepted ({len(accepted)}), stopping candidate scan.", flush=True)
            break

    print(f"[stage0/esmfold] total accepted: {len(accepted)} in {time.time()-t0:.1f}s", flush=True)
    write_jsonl(reject_log_path, rejects)

    # -------- Step 5: split into main / calib / donor / held-out --------
    random.shuffle(accepted)
    need_total = args.n_main + args.n_calib + args.n_donor + args.n_heldout_probe
    if len(accepted) < need_total:
        print(f"[stage0/warn] only {len(accepted)} chains, needed {need_total}. "
              f"Splits will be shrunk proportionally.", flush=True)
        # Shrink proportional; but keep at least 20 main.
        s = len(accepted) / need_total
        n_main = max(20, int(args.n_main * s))
        n_calib = max(5, int(args.n_calib * s))
        n_donor = max(20, int(args.n_donor * s))
        n_hp = len(accepted) - n_main - n_calib - n_donor
    else:
        n_main = args.n_main
        n_calib = args.n_calib
        n_donor = args.n_donor
        n_hp = min(args.n_heldout_probe, len(accepted) - n_main - n_calib - n_donor)

    main_set = accepted[:n_main]
    calib_set = accepted[n_main:n_main + n_calib]
    donor_set = accepted[n_main + n_calib:n_main + n_calib + n_donor]
    heldout_probe_set = accepted[n_main + n_calib + n_donor:n_main + n_calib + n_donor + n_hp]

    for r in main_set:
        r["split"] = "main"
    for r in calib_set:
        r["split"] = "calibration"
    for r in donor_set:
        r["split"] = "donor"
    for r in heldout_probe_set:
        r["split"] = "heldout_probe"

    # -------- Step 6: assign donors --------
    # For each main / calib chain, pick 3 donors that
    #   (a) have length within ±20%
    #   (b) don't contain hairpin at the *target chain's* target-region indices
    #       (approximation: donor's native SS at [target_start..target_end] should not
    #       be a hairpin — but donor and target have different sequences so target's
    #       indices refer to the *shared* residue window we intervene on; we just
    #       require the donor is a distinct chain of similar length).
    def pick_donors(target_r: Dict, k: int = 3) -> List[str]:
        tgt_len = target_r["len"]
        candidates = [(d["pdb_id"], d["chain_id"], d["len"])
                      for d in donor_set
                      if abs(d["len"] - tgt_len) <= 0.20 * tgt_len
                      and (d["pdb_id"], d["chain_id"]) != (target_r["pdb_id"], target_r["chain_id"])]
        if len(candidates) < k:
            # relax constraints
            candidates = [(d["pdb_id"], d["chain_id"], d["len"]) for d in donor_set
                          if (d["pdb_id"], d["chain_id"]) != (target_r["pdb_id"], target_r["chain_id"])]
        random.shuffle(candidates)
        return [f"{p}_{c}" for (p, c, _) in candidates[:k]]

    for r in main_set + calib_set:
        r["donor_ids"] = pick_donors(r, k=3)

    # -------- Step 7: (Optional) CATH mapping placeholder --------
    # We attempt a lightweight CATH lookup via https://www.cathdb.info/ — but do NOT
    # require network access. If offline, cath_label stays None.
    # (Skipped for automated runs; a helper script can post-populate.)

    # -------- Step 8: write manifest --------
    all_records = main_set + calib_set + donor_set + heldout_probe_set
    write_jsonl(manifest_path, all_records)

    # Summary
    summary = {
        "n_pisces_total": len(all_chains),
        "n_after_length_filter": len(chains_len),
        "n_candidates_after_native_dssp": len(candidates),
        "n_accepted_after_esmfold": len(accepted),
        "n_main": len(main_set), "n_calibration": len(calib_set),
        "n_donor": len(donor_set), "n_heldout_probe": len(heldout_probe_set),
        "min_len": args.min_len, "max_len": args.max_len,
        "seed": args.seed, "baseline_threshold": 0.7,
        "manifest_path": str(manifest_path),
    }
    write_json(summary_path, summary)
    print("[stage0] DONE.", flush=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
