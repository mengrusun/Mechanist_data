#!/usr/bin/env python3
"""M2 worker — seq2pair vs pair2seq matched pathway ablation.

Depends on M1's early-window discovery. Reads `results/M1/early_window.json` to know
which blocks to patch; falls back to `[0..7]` (band b_0_3 + b_4_7) if M1 hasn't
localized.

Per chain, per donor:
  - baseline (1 forward)   — actually inherited from M1; but re-computed for freshness.
  - Step A: seq2pair donor-patch (1)
  - Step B: pair2seq matched-donor-patch (1)
  - Step C: seq2pair zero-ablation robustness sanity (1)
  - Step D: matched-control non-target-pair seq2pair patch (1)

Per chain × 3 donors: 1 + 3 × 4 + 3 (donor stores) = 16 forwards.
200 chains × 16 × 2s = 6400s / 4 workers = 1600s ≈ 27 min (well under 2.5h plan).
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

import torch

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))

from esmfold_lib import (  # noqa: E402
    PROJECT_ROOT, MANIFEST_PATH, load_esmfold, esmfold_forward,
    predict_and_judge_hairpin,
    patch_seq2pair_output, patch_pair2seq_output,
    set_seed, write_jsonl, append_jsonl, write_json,
)


def collect_donor_pathway_outputs(model, tok, donor_seq: str,
                                  block_indices: List[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    """For a donor sequence, capture the seq2pair.output and pair2seq.output at each
    listed block. Returns dict: {block_idx: {'s2p': (1,Ld,Ld,Z), 'p2s': (1,Ld,S)}}.
    """
    stores: Dict[int, Dict[str, torch.Tensor]] = {k: {} for k in block_indices}
    handles = []

    def make_s2p_hook(k):
        def hook(module, inputs, output):
            stores[k]['s2p'] = output.detach().clone()
            return output
        return hook

    def make_p2s_hook(k):
        def hook(module, inputs, output):
            stores[k]['p2s'] = output.detach().clone()
            return output
        return hook

    try:
        for k in block_indices:
            h1 = model.trunk.blocks[k].sequence_to_pair.register_forward_hook(make_s2p_hook(k))
            h2 = model.trunk.blocks[k].pair_to_sequence.register_forward_hook(make_p2s_hook(k))
            handles.append(h1)
            handles.append(h2)
        with torch.no_grad():
            _ = esmfold_forward(model, tok, donor_seq)
    finally:
        for h in handles:
            h.remove()
    return stores


def find_matched_control_pair_mask(L: int, target_pairs: List[Tuple[int, int]],
                                   target_start: int, target_end: int) -> List[Tuple[int, int]]:
    """Return a size-matched non-target pair set outside the hairpin region."""
    n_target = len(target_pairs)
    if n_target == 0:
        return []
    # Simple: place pairs on residues both after target_end + 5. If not enough room,
    # place before target_start - 5.
    tgt_len = target_end - target_start + 1
    offset = target_end + 6
    if offset + tgt_len + 5 > L - 3:
        offset = max(3, target_start - 6 - tgt_len)
    if offset + tgt_len > L - 3:
        # cannot fit — return empty and skip step D for this chain
        return []
    # Mirror the target pair pattern shifted by offset - target_start.
    shift = offset - target_start
    ctrl_pairs = [(i + shift, j + shift) for (i, j) in target_pairs
                  if 0 <= i + shift < L and 0 <= j + shift < L]
    return ctrl_pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker-id", type=int, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--early-window-json", type=Path,
                   default=PROJECT_ROOT / "results" / "M1" / "early_window.json")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "M2")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed + args.worker_id)
    args.out.mkdir(parents=True, exist_ok=True)

    if args.early_window_json.exists():
        ew = json.loads(open(args.early_window_json).read())
        early_blocks: List[int] = list(ew.get("blocks", list(range(0, 8))))
        print(f"[m2-{args.worker_id}] early window from M1: {early_blocks}", flush=True)
    else:
        early_blocks = list(range(0, 8))
        print(f"[m2-{args.worker_id}] early window: fallback to {early_blocks}", flush=True)

    accepted: List[Dict] = []
    donors: Dict[str, Dict] = {}
    with open(args.manifest) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "main":
                accepted.append(r)
            elif r.get("split") == "donor":
                donors[f"{r['pdb_id']}_{r['chain_id']}"] = r

    my_chains = accepted[args.start:args.stop]
    print(f"[m2-{args.worker_id}] {len(my_chains)} chains in slice", flush=True)

    print(f"[m2-{args.worker_id}] loading model...", flush=True)
    model, tok, _ = load_esmfold(device="cuda", dtype=torch.float32)
    print(f"[m2-{args.worker_id}] model loaded", flush=True)

    out_path = args.out / f"pathway_effect_worker_{args.worker_id}.jsonl"
    if out_path.exists():
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
        donor_recs = [donors[d] for d in donor_ids if d in donors]

        if not cross_pairs:
            # No pairs → M2 cannot proceed
            print(f"[m2-{args.worker_id}] no cross_pairs for {pdb_id}, skip", flush=True)
            continue
        if not donor_recs:
            continue

        elapsed = time.time() - t0
        rate = elapsed / max(1, ci)
        eta = rate * (len(my_chains) - ci) / 60
        print(f"[m2-{args.worker_id}] {ci}/{len(my_chains)}  {pdb_id}_{chain_id} L={L}  "
              f"pairs={len(cross_pairs)}  ({rate:.1f}s/chain, eta {eta:.1f}min)", flush=True)

        try:
            # Baseline
            base = predict_and_judge_hairpin(model, tok, seq, target_start, target_end,
                                             no_recycles=1)
            total_forwards += 1
            if not base["dssp_ok"]:
                continue
            base_hairpin = int(bool(base["is_hairpin"]))
            base_plddt = base["plddt_target_mean"]
            append_jsonl(out_path, [{
                "pdb_id": pdb_id, "chain_id": chain_id,
                "condition": "baseline", "band": None, "donor_id": None,
                "is_hairpin": base_hairpin, "plddt_target": base_plddt,
                "target_start": target_start, "target_end": target_end,
                "L": L, "cath_label": cath_label,
            }])

            # Matched-control non-target pair mask (Step D)
            ctrl_pairs = find_matched_control_pair_mask(L, cross_pairs, target_start, target_end)

            device = next(model.parameters()).device

            def _write_run(cond, donor_key, pr, extra=None):
                rec = {
                    "pdb_id": pdb_id, "chain_id": chain_id,
                    "condition": cond,
                    "band": "early_window", "donor_id": donor_key,
                    "cath_label": cath_label,
                }
                if extra:
                    rec.update(extra)
                if pr and pr["dssp_ok"] and pr.get("dssp_align_ok", True):
                    rec.update({"is_hairpin": int(bool(pr["is_hairpin"])),
                                "plddt_target": pr["plddt_target_mean"],
                                "status": "ok"})
                else:
                    reason = "unknown"
                    if pr is None:
                        reason = "not_attempted"
                    elif not pr["dssp_ok"]:
                        reason = "dssp_failed"
                    elif not pr.get("dssp_align_ok", True):
                        reason = "dssp_misaligned"
                    rec.update({"is_hairpin": None, "plddt_target": None, "status": reason})
                append_jsonl(out_path, [rec])

            for donor_r in donor_recs:
                donor_seq = donor_r["seq"]
                donor_key = f"{donor_r['pdb_id']}_{donor_r['chain_id']}"
                Ld = len(donor_seq)
                if max(target_res) >= Ld:
                    # donor too short — write NA rows for every condition so pairing is honest
                    for cond in ("seq2pair_donor", "pair2seq_donor", "seq2pair_zero",
                                 "seq2pair_matched_ctrl"):
                        _write_run(cond, donor_key, None,
                                   extra={"na_reason": "donor_too_short", "Ld": Ld})
                    continue

                # Collect donor's seq2pair and pair2seq outputs at each block in early_blocks
                dstores = collect_donor_pathway_outputs(model, tok, donor_seq, early_blocks)
                total_forwards += 1

                # Precompute per-block donor tensors on device
                s2p_by_block = {k: dstores[k]['s2p'][0].to(device) for k in early_blocks}
                p2s_by_block = {k: dstores[k]['p2s'][0].to(device) for k in early_blocks}

                valid_pairs = [(i, j) for (i, j) in cross_pairs if i < Ld and j < Ld]

                # Step A: seq2pair donor-patch on target pairs (block-specific donor)
                if valid_pairs:
                    # For seq2pair the hook uses a single donor tensor because patch_seq2pair_output
                    # was written to accept a single s2p tensor. We must call it per block
                    # to preserve block-specific donors.
                    # Modify: sequentially attach hooks per block with per-block donor.
                    handles_s2p = []
                    try:
                        pi_idx = torch.as_tensor([p[0] for p in valid_pairs], dtype=torch.long, device=device)
                        pj_idx = torch.as_tensor([p[1] for p in valid_pairs], dtype=torch.long, device=device)
                        for k in early_blocks:
                            donor_z_k = s2p_by_block[k]
                            def make_hook(donor_z_local):
                                def hook(module, inputs, output):
                                    new_out = output.clone()
                                    new_out[0, pi_idx, pj_idx, :] = donor_z_local[pi_idx, pj_idx, :].to(new_out.dtype)
                                    new_out[0, pj_idx, pi_idx, :] = donor_z_local[pj_idx, pi_idx, :].to(new_out.dtype)
                                    return new_out
                                return hook
                            h = model.trunk.blocks[k].sequence_to_pair.register_forward_hook(
                                make_hook(donor_z_k))
                            handles_s2p.append(h)
                        pr = predict_and_judge_hairpin(model, tok, seq, target_start,
                                                       target_end, no_recycles=1)
                        total_forwards += 1
                    finally:
                        for h in handles_s2p:
                            h.remove()
                    _write_run("seq2pair_donor", donor_key, pr)
                else:
                    _write_run("seq2pair_donor", donor_key, None,
                               extra={"na_reason": "no_valid_pairs"})

                # Step B: pair2seq matched donor-patch (target_res mask, per-block donor)
                if max(target_res) < Ld:
                    tgt_idx = torch.as_tensor(target_res, dtype=torch.long, device=device)
                    handles_p2s = []
                    try:
                        for k in early_blocks:
                            donor_p2s_k = p2s_by_block[k]
                            def make_hook(donor_local):
                                def hook(module, inputs, output):
                                    new_out = output.clone()
                                    new_out[0, tgt_idx, :] = donor_local[tgt_idx, :].to(new_out.dtype)
                                    return new_out
                                return hook
                            h = model.trunk.blocks[k].pair_to_sequence.register_forward_hook(
                                make_hook(donor_p2s_k))
                            handles_p2s.append(h)
                        pr = predict_and_judge_hairpin(model, tok, seq, target_start,
                                                       target_end, no_recycles=1)
                        total_forwards += 1
                    finally:
                        for h in handles_p2s:
                            h.remove()
                    _write_run("pair2seq_donor", donor_key, pr)
                else:
                    _write_run("pair2seq_donor", donor_key, None,
                               extra={"na_reason": "donor_too_short_for_targetres"})

                # Step C: seq2pair zero-ablation
                if valid_pairs:
                    with patch_seq2pair_output(model, early_blocks, valid_pairs, s2p_by_block[early_blocks[0]],
                                               zero_ablation=True):
                        pr = predict_and_judge_hairpin(model, tok, seq, target_start,
                                                       target_end, no_recycles=1)
                        total_forwards += 1
                    _write_run("seq2pair_zero", donor_key, pr)
                else:
                    _write_run("seq2pair_zero", donor_key, None, extra={"na_reason": "no_valid_pairs"})

                # Step D: matched-control seq2pair donor-patch on ctrl_pairs (per-block donor)
                valid_ctrl = [(i, j) for (i, j) in ctrl_pairs if i < Ld and j < Ld]
                if valid_ctrl:
                    ci_idx = torch.as_tensor([p[0] for p in valid_ctrl], dtype=torch.long, device=device)
                    cj_idx = torch.as_tensor([p[1] for p in valid_ctrl], dtype=torch.long, device=device)
                    handles_s2p = []
                    try:
                        for k in early_blocks:
                            donor_z_k = s2p_by_block[k]
                            def make_hook(donor_local):
                                def hook(module, inputs, output):
                                    new_out = output.clone()
                                    new_out[0, ci_idx, cj_idx, :] = donor_local[ci_idx, cj_idx, :].to(new_out.dtype)
                                    new_out[0, cj_idx, ci_idx, :] = donor_local[cj_idx, ci_idx, :].to(new_out.dtype)
                                    return new_out
                                return hook
                            h = model.trunk.blocks[k].sequence_to_pair.register_forward_hook(
                                make_hook(donor_z_k))
                            handles_s2p.append(h)
                        pr = predict_and_judge_hairpin(model, tok, seq, target_start,
                                                       target_end, no_recycles=1)
                        total_forwards += 1
                    finally:
                        for h in handles_s2p:
                            h.remove()
                    _write_run("seq2pair_matched_ctrl", donor_key, pr,
                               extra={"matched_pair": valid_ctrl[:5]})
                else:
                    _write_run("seq2pair_matched_ctrl", donor_key, None,
                               extra={"na_reason": "no_ctrl_pairs_fit"})

                del dstores, s2p_by_block, p2s_by_block
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"[m2-{args.worker_id}] OOM at {pdb_id}", flush=True)
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"[m2-{args.worker_id}] error at {pdb_id}: {e}", flush=True)
            traceback.print_exc()
            continue

    write_json(args.out / f"m2_worker_{args.worker_id}_summary.json", {
        "worker_id": args.worker_id, "slice": [args.start, args.stop],
        "total_forwards": total_forwards,
        "elapsed_sec": time.time() - t0,
        "early_blocks": early_blocks,
    })
    print(f"[m2-{args.worker_id}] DONE. forwards={total_forwards} "
          f"time={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
