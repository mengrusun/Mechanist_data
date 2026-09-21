#!/usr/bin/env python3
"""M1 worker — Block-window s-patching + specificity controls.

Each worker processes a slice of the 200 main chains (assigned by --start / --stop).
For every chain in its slice, the worker runs (per chain × per condition):
  - baseline (no intervention) — one forward
  - Step B: s-patching each of 8 block bands × 3 donors = 24 conditions
  - Step C (specificity 1): z-patching in each of the 4 EARLY-CANDIDATE bands × 3 donors = 12
    (we don't do all 8 — that'd double the total; the plan requires "same window as B",
    so we z-patch in each candidate window bearing an early designation)
  - Step D (specificity 2): implicit — the late-window s-patching is already covered
    by Step B (bands 24-31, 32-39, 40-47 are the late windows).
  - Step E (matched-control specificity): for the 3 EARLIEST bands ({[0-3], [4-7], [8-11]})
    × 3 donors = 9, apply s-patching to a non-target matched mask (same length, different
    residues, no hairpin overlap).

Per chain total: 1 (baseline) + 24 (B) + 12 (C, early bands only) + 9 (E) = 46 forwards.
At ~2s/forward on GPU → ~92s/chain. 200 chains / 4 workers = 50 chains/worker × 92s = 4600s ≈ 77 min.

Writes results/M1/effect_by_window_worker_{id}.jsonl. Each record represents one
(chain, condition, donor) triple with the per-run outcome (is_hairpin, plddt).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))

from esmfold_lib import (  # noqa: E402
    PROJECT_ROOT, PREPARED_DIR, MANIFEST_PATH, NUM_TRUNK_BLOCKS,
    load_esmfold, esmfold_forward, predict_and_judge_hairpin,
    patch_s_at_blocks, patch_z_at_blocks, collect_s_z_at_block,
    set_seed, write_jsonl, append_jsonl, write_json,
)


BLOCK_BANDS: List[Tuple[str, List[int]]] = [
    ("b_0_3",   list(range(0, 4))),
    ("b_4_7",   list(range(4, 8))),
    ("b_8_11",  list(range(8, 12))),
    ("b_12_15", list(range(12, 16))),
    ("b_16_23", list(range(16, 24))),
    ("b_24_31", list(range(24, 32))),
    ("b_32_39", list(range(32, 40))),
    ("b_40_47", list(range(40, 48))),
]
EARLY_BANDS_FOR_Z = ["b_0_3", "b_4_7", "b_8_11", "b_12_15"]
EARLY_BANDS_FOR_E = ["b_0_3", "b_4_7", "b_8_11"]  # matched-control non-target mask on early bands
BAND_INDICES = {name: idx for name, idx in BLOCK_BANDS}


def find_matched_control_mask(seq_len: int, target_start: int, target_end: int) -> Tuple[int, int]:
    """Pick a non-overlapping residue mask of the same length as [target_start..target_end]
    from the same chain, avoiding N/C termini (first/last 3 residues).

    Strategy: place immediately AFTER target_end + 5 if it fits; else BEFORE target_start - 5.
    """
    tgt_len = target_end - target_start + 1
    # Try after
    a = target_end + 6
    b = a + tgt_len - 1
    if b < seq_len - 3:
        return a, b
    # Try before
    b2 = target_start - 6
    a2 = b2 - tgt_len + 1
    if a2 >= 3:
        return a2, b2
    # Fallback: use residues 3..3+tgt_len-1 even if it overlaps a bit (should be rare)
    return 3, min(3 + tgt_len - 1, seq_len - 4)


def collect_donor_activations(model, tok, donor_seq: str,
                              block_indices_per_band: List[List[int]]) -> Dict[int, Dict[str, torch.Tensor]]:
    """For one donor sequence, capture the s and z tensors at each block that we might
    later patch. Returns a dict keyed by block_idx: {block_idx: {'s': (1,Ld,1024), 'z': (1,Ld,Ld,128)}}
    """
    all_blocks: List[int] = sorted(set(idx for band in block_indices_per_band for idx in band))
    # We need one pass per unique block to capture — but we can capture multiple blocks in
    # one forward by attaching pre-hooks to each. We'll capture into a dict of lists.
    stores: Dict[int, Dict[str, torch.Tensor]] = {k: {} for k in all_blocks}
    handles = []
    def make_hook(k):
        def pre_hook(module, args, kwargs):
            stores[k]['s'] = args[0].detach().clone()
            stores[k]['z'] = args[1].detach().clone()
            return None
        return pre_hook
    try:
        for k in all_blocks:
            h = model.trunk.blocks[k].register_forward_pre_hook(make_hook(k), with_kwargs=True)
            handles.append(h)
        with torch.no_grad():
            _ = esmfold_forward(model, tok, donor_seq)
    finally:
        for h in handles:
            h.remove()
    return stores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker-id", type=int, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "M1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--only-chains", type=int, default=-1, help="optional limit for smoke test")
    p.add_argument("--resume", action="store_true",
                   help="if set, skip chains that already have a complete record set in the "
                        "existing effect_by_window JSONL and append new results instead of "
                        "truncating the file.")
    args = p.parse_args()

    set_seed(args.seed + args.worker_id)
    args.out.mkdir(parents=True, exist_ok=True)
    accepted_records: List[Dict] = []
    with open(args.manifest) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "main":
                accepted_records.append(r)
    print(f"[m1-{args.worker_id}] {len(accepted_records)} main chains total", flush=True)
    my_chains = accepted_records[args.start:args.stop]
    if args.only_chains > 0:
        my_chains = my_chains[:args.only_chains]
    print(f"[m1-{args.worker_id}] slice [{args.start},{args.stop}), running on {len(my_chains)} chains",
          flush=True)

    # Build donor index (map "PID_C" -> record) from manifest
    donor_index = {f"{r['pdb_id']}_{r['chain_id']}": r for r in
                   [json.loads(l) for l in open(args.manifest)] if r["split"] == "donor"}
    # Iterating twice ok — file is small. We rely on this dict for donor sequences.
    with open(args.manifest) as f:
        donor_index = {}
        for line in f:
            r = json.loads(line)
            if r.get("split") == "donor":
                donor_index[f"{r['pdb_id']}_{r['chain_id']}"] = r

    print(f"[m1-{args.worker_id}] loading model...", flush=True)
    model, tok, _ = load_esmfold(device="cuda", dtype=torch.float32)
    print(f"[m1-{args.worker_id}] model loaded", flush=True)

    all_unique_blocks_for_bands = [band for _, band in BLOCK_BANDS]  # all 48

    out_path = args.out / f"effect_by_window_worker_{args.worker_id}.jsonl"
    reject_path = args.out / f"m1_rejects_worker_{args.worker_id}.jsonl"

    # Chain-level resume: if --resume, scan the existing JSONL for any (pdb_id, chain_id)
    # that already has a baseline row *plus* at least one s_patch row per band (i.e. was
    # meaningfully processed) and skip those. Any partial chains get their rows removed
    # so the fresh run's rows for that chain don't duplicate. Otherwise, truncate.
    done_keys: set = set()
    EXPECTED_BAND_COUNT = len(BLOCK_BANDS)  # 8
    if out_path.exists():
        if args.resume:
            # per-chain: count baseline + s_patch bands touched
            per_chain_baseline: set = set()
            per_chain_bands: Dict[Tuple[str, str], set] = {}
            all_lines: List[Tuple[Tuple[str, str], str]] = []
            with open(out_path) as f:
                for line in f:
                    line_s = line.rstrip("\n")
                    if not line_s.strip():
                        continue
                    try:
                        r = json.loads(line_s)
                    except json.JSONDecodeError:
                        continue
                    key = (r.get("pdb_id"), r.get("chain_id"))
                    cond = r.get("condition")
                    if cond == "baseline":
                        per_chain_baseline.add(key)
                    elif cond == "s_patch":
                        per_chain_bands.setdefault(key, set()).add(r.get("band"))
                    all_lines.append((key, line_s))
            for key in per_chain_baseline:
                bands = per_chain_bands.get(key, set())
                if len(bands) >= EXPECTED_BAND_COUNT:
                    done_keys.add(key)
            partial_keys = per_chain_baseline - done_keys
            if partial_keys:
                # Rewrite JSONL keeping only rows for complete-chain keys
                kept = [ln for (k, ln) in all_lines if k in done_keys]
                tmp = out_path.with_suffix(out_path.suffix + ".tmp")
                with open(tmp, "w") as f:
                    for ln in kept:
                        f.write(ln + "\n")
                tmp.replace(out_path)
                print(f"[m1-{args.worker_id}] resume: removed partial rows for "
                      f"{len(partial_keys)} chain(s): {sorted(partial_keys)[:5]}", flush=True)
            print(f"[m1-{args.worker_id}] resume: {len(done_keys)} chains already complete "
                  f"in existing JSONL; will append", flush=True)
        else:
            # Legacy behavior: truncate on start
            out_path.unlink()

    total_forwards = 0
    t0 = time.time()

    for ci, chain_r in enumerate(my_chains):
        pdb_id = chain_r["pdb_id"]
        chain_id = chain_r["chain_id"]
        seq = chain_r["seq"]
        L = len(seq)
        target_start = chain_r["target_start"]
        target_end = chain_r["target_end"]
        target_res = list(range(target_start, target_end + 1))
        cross_pairs = [tuple(x) for x in chain_r.get("cross_strand_pairs", [])]
        cath_label = chain_r.get("cath_label")
        donor_ids = chain_r.get("donor_ids", [])[:3]

        # Resume skip: chain was fully processed in a prior run.
        if (pdb_id, chain_id) in done_keys:
            if ci % 10 == 0:
                print(f"[m1-{args.worker_id}] {ci}/{len(my_chains)}  {pdb_id}_{chain_id} "
                      f"already complete — skipping", flush=True)
            continue

        # Reject if we don't have enough donors
        donors_recs = [donor_index[d] for d in donor_ids if d in donor_index]
        if len(donors_recs) < 1:
            print(f"[m1-{args.worker_id}] no donors for {pdb_id}_{chain_id}, skip", flush=True)
            continue

        elapsed = time.time() - t0
        rate = elapsed / max(1, ci)
        eta = rate * (len(my_chains) - ci) / 60
        print(f"[m1-{args.worker_id}] {ci}/{len(my_chains)}  {pdb_id}_{chain_id} L={L}  "
              f"donors={len(donors_recs)}  ({rate:.1f}s/chain, eta {eta:.1f}min)", flush=True)

        try:
            # -------- 1. Baseline forward (no intervention) --------
            base = predict_and_judge_hairpin(model, tok, seq, target_start, target_end,
                                             no_recycles=1)
            total_forwards += 1
            if not base["dssp_ok"]:
                continue
            base_hairpin = int(bool(base["is_hairpin"]))
            base_plddt = base["plddt_target_mean"]

            # Record baseline
            rec = {
                "pdb_id": pdb_id, "chain_id": chain_id, "condition": "baseline",
                "band": None, "donor_id": None,
                "is_hairpin": base_hairpin, "plddt_target": base_plddt,
                "target_start": target_start, "target_end": target_end, "L": L,
                "cath_label": cath_label,
            }
            append_jsonl(out_path, [rec])

            # -------- 2. For each donor: collect donor s and z at all blocks needed --------
            for donor_r in donors_recs:
                donor_seq = donor_r["seq"]
                donor_key = f"{donor_r['pdb_id']}_{donor_r['chain_id']}"
                Ld = len(donor_seq)
                # For s-patching we need donor s of shape (Ld, S). Truncate/pad to L target.
                donor_stores = collect_donor_activations(
                    model, tok, donor_seq, all_unique_blocks_for_bands)
                total_forwards += 1

                # Build a per-block donor_s aligned to the target sequence length L.
                # If donor is shorter than L at those positions, we tile/pad by repeating (safe
                # since we only index positions ≤ Ld anyway — for target residues that need
                # positions > Ld we skip that condition).
                # Simplest: if any target_res >= Ld, this donor can't be used for this chain.
                if max(target_res) >= Ld:
                    # length mismatch — skip this donor for this chain
                    write_jsonl(reject_path, [{
                        "pdb_id": pdb_id, "chain_id": chain_id, "reason": "donor_too_short",
                        "donor": donor_key, "Ld": Ld, "target_max": max(target_res),
                    }])
                    continue

                device = next(model.parameters()).device
                # Precompute per-block donor tensors on device — one entry for each of
                # the 48 blocks (needed because different bands intervene on different
                # block indices, and correctness requires block-specific donor tensors).
                donor_s_by_block = {k: donor_stores[k]['s'][0].to(device)
                                    for k in range(NUM_TRUNK_BLOCKS)}
                donor_z_by_block = {k: donor_stores[k]['z'][0].to(device)
                                    for k in range(NUM_TRUNK_BLOCKS)}

                # -------- 2a. Step B: s-patching each of 8 bands --------
                for band_name, band_blocks in BLOCK_BANDS:
                    sub_donor = {k: donor_s_by_block[k] for k in band_blocks}
                    with patch_s_at_blocks(model, band_blocks, target_res, sub_donor):
                        pr = predict_and_judge_hairpin(model, tok, seq, target_start,
                                                       target_end, no_recycles=1)
                        total_forwards += 1
                    rec_base = {
                        "pdb_id": pdb_id, "chain_id": chain_id,
                        "condition": "s_patch",
                        "band": band_name, "donor_id": donor_key,
                        "cath_label": cath_label,
                    }
                    if pr["dssp_ok"] and pr.get("dssp_align_ok", True):
                        rec_base.update({"is_hairpin": int(bool(pr["is_hairpin"])),
                                         "plddt_target": pr["plddt_target_mean"],
                                         "status": "ok"})
                    else:
                        rec_base.update({"is_hairpin": None, "plddt_target": None,
                                         "status": "dssp_failed" if not pr["dssp_ok"] else "dssp_misaligned"})
                    append_jsonl(out_path, [rec_base])

                # -------- 2b. Step C: z-patching in the early bands only --------
                if cross_pairs:
                    for band_name in EARLY_BANDS_FOR_Z:
                        band_blocks = BAND_INDICES[band_name]
                        valid_pairs = [(i, j) for (i, j) in cross_pairs if i < Ld and j < Ld]
                        if not valid_pairs:
                            append_jsonl(out_path, [{
                                "pdb_id": pdb_id, "chain_id": chain_id,
                                "condition": "z_patch_early",
                                "band": band_name, "donor_id": donor_key,
                                "cath_label": cath_label,
                                "is_hairpin": None, "plddt_target": None,
                                "status": "no_valid_pairs",
                            }])
                            continue
                        sub_donor_z = {k: donor_z_by_block[k] for k in band_blocks}
                        with patch_z_at_blocks(model, band_blocks, valid_pairs, sub_donor_z):
                            pr = predict_and_judge_hairpin(model, tok, seq, target_start,
                                                           target_end, no_recycles=1)
                            total_forwards += 1
                        rec = {
                            "pdb_id": pdb_id, "chain_id": chain_id,
                            "condition": "z_patch_early",
                            "band": band_name, "donor_id": donor_key,
                            "cath_label": cath_label,
                        }
                        if pr["dssp_ok"] and pr.get("dssp_align_ok", True):
                            rec.update({"is_hairpin": int(bool(pr["is_hairpin"])),
                                        "plddt_target": pr["plddt_target_mean"],
                                        "status": "ok"})
                        else:
                            rec.update({"is_hairpin": None, "plddt_target": None,
                                        "status": "dssp_failed" if not pr["dssp_ok"] else "dssp_misaligned"})
                        append_jsonl(out_path, [rec])

                # -------- 2c. Step E: matched-control mask on early bands --------
                mask_a, mask_b = find_matched_control_mask(L, target_start, target_end)
                mask_res = list(range(mask_a, mask_b + 1))
                mask_valid = (0 <= mask_a and mask_b < L and set(mask_res) != set(target_res)
                              and max(mask_res) < Ld)
                for band_name in EARLY_BANDS_FOR_E:
                    band_blocks = BAND_INDICES[band_name]
                    rec = {
                        "pdb_id": pdb_id, "chain_id": chain_id,
                        "condition": "s_patch_matched_ctrl",
                        "band": band_name, "donor_id": donor_key,
                        "matched_mask": [mask_a, mask_b],
                        "cath_label": cath_label,
                    }
                    if not mask_valid:
                        rec.update({"is_hairpin": None, "plddt_target": None,
                                    "status": "matched_mask_invalid"})
                        append_jsonl(out_path, [rec])
                        continue
                    sub_donor_s = {k: donor_s_by_block[k] for k in band_blocks}
                    with patch_s_at_blocks(model, band_blocks, mask_res, sub_donor_s):
                        pr = predict_and_judge_hairpin(model, tok, seq, target_start,
                                                       target_end, no_recycles=1)
                        total_forwards += 1
                    if pr["dssp_ok"] and pr.get("dssp_align_ok", True):
                        rec.update({"is_hairpin": int(bool(pr["is_hairpin"])),
                                    "plddt_target": pr["plddt_target_mean"],
                                    "status": "ok"})
                    else:
                        rec.update({"is_hairpin": None, "plddt_target": None,
                                    "status": "dssp_failed" if not pr["dssp_ok"] else "dssp_misaligned"})
                    append_jsonl(out_path, [rec])
                # Free donor_stores memory before next donor
                del donor_stores, donor_s_by_block, donor_z_by_block
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"[m1-{args.worker_id}] OOM at {pdb_id} L={L} — skipping", flush=True)
            torch.cuda.empty_cache()
            write_jsonl(reject_path, [{"pdb_id": pdb_id, "chain_id": chain_id, "reason": "oom", "L": L}])
            continue
        except Exception as e:
            print(f"[m1-{args.worker_id}] error at {pdb_id}: {e}", flush=True)
            traceback.print_exc()
            write_jsonl(reject_path, [{"pdb_id": pdb_id, "chain_id": chain_id,
                                        "reason": f"exception: {e}"}])
            continue

    write_json(args.out / f"m1_worker_{args.worker_id}_summary.json", {
        "worker_id": args.worker_id, "slice": [args.start, args.stop],
        "n_chains_processed": len(my_chains),
        "total_forwards": total_forwards,
        "elapsed_sec": time.time() - t0,
    })
    print(f"[m1-{args.worker_id}] DONE. forwards={total_forwards} "
          f"time={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
