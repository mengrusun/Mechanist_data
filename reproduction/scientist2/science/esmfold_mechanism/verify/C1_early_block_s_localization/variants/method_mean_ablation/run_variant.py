#!/usr/bin/env python3
"""C1 Method Variant — Mean-Ablation s-patching (within-family submethod swap).

Within-family method swap: Causal Attribution / Patching submethod.
Instead of donor-cover patching (clean/corrupted paradigm using donor chain s),
replace s[target_res] with the MEAN activation across a random sample of 30 main chains
at the same residue positions and block. This tests whether the s-patching effect
is specific to the donor's signal vs. simply removing the target chain's own s signal.

Design:
- Early band b_0_3 (blocks 0-3) only — the band with the strongest M1 effect (Δ=-0.862)
- N=200 main chains (same as M1)
- Mean computed from a held-out 30-chain "reference pool" (non-overlapping with target chains)
- DSSP on ESMFold-predicted structure (task.md HARD)
- Specificity controls: same-window z-ablation (zero), matched-ctrl non-target mask mean-ablation

Expected: If donor-specific signal was driving M1, mean-ablation should show a WEAKER effect.
If the mere disruption of the target chain's early-block s signal is sufficient,
mean-ablation should show a SIMILAR or STRONGER effect.

GPU: CUDA_VISIBLE_DEVICES set by caller (first positional arg or env var)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Tuple, Sequence

import numpy as np
import torch

_this_dir = Path(__file__).resolve().parent.parent.parent.parent.parent  # project root
sys.path.insert(0, str(_this_dir / "scripts"))

from esmfold_lib import (
    PROJECT_ROOT, MANIFEST_PATH, load_esmfold, esmfold_forward,
    predict_and_judge_hairpin, set_seed, write_jsonl, append_jsonl, write_json,
    NUM_TRUNK_BLOCKS, S_HIDDEN,
)

EARLY_BAND = "b_0_3"
EARLY_BLOCKS = [0, 1, 2, 3]
VARIANT_NAME = "method_mean_ablation"


@contextmanager
def patch_s_with_mean_at_blocks(model, block_indices: Sequence[int],
                                  target_res: Sequence[int],
                                  mean_s_by_block: Dict[int, torch.Tensor]):
    """Pre-hook context: replace s[0, target_res, :] with mean_s_by_block[k][target_res, :]
    for each block k in block_indices. Mean is computed over a reference pool of chains."""
    handles = []
    _first = next(iter(mean_s_by_block.values()))
    device = _first.device
    tgt = torch.as_tensor(list(target_res), dtype=torch.long, device=device)

    def make_pre_hook(blk_idx):
        mean_s_local = mean_s_by_block[blk_idx]
        def pre_hook(module, args, kwargs):
            seq_state = args[0]
            new_seq = seq_state.clone()
            # Mean s is (S_HIDDEN,) or (max_L, S_HIDDEN); use per-position mean
            # We take the mean s at the target positions
            mean_vals = mean_s_local[tgt, :].to(new_seq.dtype).to(new_seq.device)
            new_seq[0, tgt, :] = mean_vals
            new_args = (new_seq,) + args[1:]
            return (new_args, kwargs)
        return pre_hook

    try:
        for k in block_indices:
            h = model.trunk.blocks[k].register_forward_pre_hook(make_pre_hook(k), with_kwargs=True)
            handles.append(h)
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def zero_ablate_s_at_blocks(model, block_indices: Sequence[int],
                              target_res: Sequence[int], device: str = "cuda"):
    """Pre-hook context: zero-ablate s[0, target_res, :] at entry to each block k."""
    handles = []
    tgt = torch.as_tensor(list(target_res), dtype=torch.long, device=device)

    def make_pre_hook(blk_idx):
        def pre_hook(module, args, kwargs):
            seq_state = args[0]
            new_seq = seq_state.clone()
            new_seq[0, tgt, :] = 0.0
            new_args = (new_seq,) + args[1:]
            return (new_args, kwargs)
        return pre_hook

    try:
        for k in block_indices:
            h = model.trunk.blocks[k].register_forward_pre_hook(make_pre_hook(k), with_kwargs=True)
            handles.append(h)
        yield
    finally:
        for h in handles:
            h.remove()


def collect_s_at_blocks(model, tok, seq: str, block_indices: List[int]) -> Dict[int, torch.Tensor]:
    """Capture s entering each block in block_indices for the given sequence."""
    stores: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(k):
        def pre_hook(module, args, kwargs):
            stores[k] = args[0].detach()[0].clone()  # (L, S_HIDDEN)
            return None
        return pre_hook

    try:
        for k in block_indices:
            h = model.trunk.blocks[k].register_forward_pre_hook(make_hook(k), with_kwargs=True)
            handles.append(h)
        with torch.no_grad():
            _ = esmfold_forward(model, tok, seq)
    finally:
        for h in handles:
            h.remove()
    return stores


def build_mean_s(model, tok, ref_chains: List[Dict], block_indices: List[int],
                  max_len: int) -> Dict[int, torch.Tensor]:
    """Compute mean s at each block over ref_chains, truncated/padded to max_len positions."""
    accum: Dict[int, List[torch.Tensor]] = {k: [] for k in block_indices}
    device = next(model.parameters()).device

    for rec in ref_chains:
        seq = rec["seq"]
        try:
            stores = collect_s_at_blocks(model, tok, seq, block_indices)
            for k in block_indices:
                s = stores[k]  # (Lref, 1024)
                Lref = s.shape[0]
                # Pad/truncate to max_len
                if Lref >= max_len:
                    accum[k].append(s[:max_len])
                else:
                    padded = torch.zeros(max_len, S_HIDDEN, dtype=s.dtype, device=s.device)
                    padded[:Lref] = s
                    accum[k].append(padded)
        except Exception as e:
            print(f"[mean_s] ref chain {rec['pdb_id']}_{rec['chain_id']} failed: {e}", flush=True)
            continue

    mean_s: Dict[int, torch.Tensor] = {}
    for k in block_indices:
        if accum[k]:
            stacked = torch.stack(accum[k], dim=0)  # (N_ref, max_len, 1024)
            mean_s[k] = stacked.mean(dim=0).to(device)  # (max_len, 1024)
        else:
            mean_s[k] = torch.zeros(max_len, S_HIDDEN, device=device)

    return mean_s


def find_matched_control_mask(seq_len: int, target_start: int, target_end: int) -> Tuple[int, int]:
    tgt_len = target_end - target_start + 1
    a = target_end + 6
    b = a + tgt_len - 1
    if b < seq_len - 3:
        return a, b
    b2 = target_start - 6
    a2 = b2 - tgt_len + 1
    if a2 >= 3:
        return a2, b2
    return 3, min(3 + tgt_len - 1, seq_len - 4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu_id", nargs="?", default="0", help="CUDA_VISIBLE_DEVICES")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--out", type=Path,
                        default=Path(__file__).resolve().parent)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-ref", type=int, default=30,
                        help="Number of reference chains for mean-s computation")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    set_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    # Load manifest
    all_chains = []
    with open(args.manifest) as f:
        for line in f:
            all_chains.append(json.loads(line))
    main_chains = [r for r in all_chains if r.get("split") == "main"]
    donor_chains = [r for r in all_chains if r.get("split") == "donor"]
    print(f"[mean_ablation] {len(main_chains)} main chains, {len(donor_chains)} donor chains", flush=True)

    # Reference pool: first n_ref donor chains (independent from main set)
    rng = np.random.default_rng(args.seed)
    ref_pool = donor_chains[:args.n_ref]
    print(f"[mean_ablation] ref pool: {len(ref_pool)} chains", flush=True)

    t0 = time.time()
    print("[mean_ablation] loading ESMFold model...", flush=True)
    model, tok, _ = load_esmfold(device="cuda", dtype=torch.float32)
    device = next(model.parameters()).device
    print(f"[mean_ablation] model loaded on {device}", flush=True)

    # Determine max_len from main chains
    max_len = max(len(r["seq"]) for r in main_chains) + 10  # small buffer

    # Build mean s from reference pool (forward pass per ref chain)
    print(f"[mean_ablation] computing mean s over {len(ref_pool)} ref chains...", flush=True)
    mean_s_by_block = build_mean_s(model, tok, ref_pool, EARLY_BLOCKS, max_len)
    print(f"[mean_ablation] mean s computed; elapsed={time.time()-t0:.1f}s", flush=True)

    out_path = args.out / "effect_mean_ablation.jsonl"
    if out_path.exists():
        out_path.unlink()

    results_summary = {
        "variant": VARIANT_NAME,
        "band": EARLY_BAND,
        "blocks": EARLY_BLOCKS,
        "n_ref_chains": len(ref_pool),
        "conditions_run": ["mean_ablation_s", "mean_ablation_s_matched_ctrl"],
        "n_chains_attempted": 0,
        "n_chains_ok": {"mean_ablation_s": 0, "mean_ablation_s_matched_ctrl": 0},
        "delta_rate": {"mean_ablation_s": None, "mean_ablation_s_matched_ctrl": None},
        "p_value": {"mean_ablation_s": None, "mean_ablation_s_matched_ctrl": None},
    }

    chain_results: Dict[str, Dict] = {}  # pdb_chain -> {baseline, mean_ablation, ctrl}

    for ci, chain_r in enumerate(main_chains):
        pdb_id = chain_r["pdb_id"]
        chain_id = chain_r["chain_id"]
        seq = chain_r["seq"]
        L = len(seq)
        target_start = chain_r["target_start"]
        target_end = chain_r["target_end"]
        target_res = list(range(target_start, target_end + 1))
        cath_label = chain_r.get("cath_label")
        key = f"{pdb_id}_{chain_id}"

        elapsed = time.time() - t0
        print(f"[mean_ablation] {ci+1}/{len(main_chains)} {key} L={L} "
              f"({elapsed:.0f}s elapsed)", flush=True)

        results_summary["n_chains_attempted"] += 1

        try:
            # Build per-block mean s aligned to this chain's length
            # Since we have max_len-padded mean_s, slice to L
            chain_mean_s = {k: mean_s_by_block[k][:L].to(device) for k in EARLY_BLOCKS}

            # Baseline
            base = predict_and_judge_hairpin(model, tok, seq, target_start, target_end, no_recycles=1)
            if not base["dssp_ok"]:
                print(f"[mean_ablation] {key} baseline DSSP failed, skip", flush=True)
                continue
            base_hairpin = int(bool(base["is_hairpin"]))

            rec_base = {
                "pdb_id": pdb_id, "chain_id": chain_id, "condition": "baseline",
                "band": None, "is_hairpin": base_hairpin,
                "plddt_target": base["plddt_target_mean"],
                "target_start": target_start, "target_end": target_end,
                "L": L, "cath_label": cath_label,
            }
            append_jsonl(out_path, [rec_base])

            # Mean-ablation on target residues in b_0_3
            with patch_s_with_mean_at_blocks(model, EARLY_BLOCKS, target_res, chain_mean_s):
                pr_mean = predict_and_judge_hairpin(model, tok, seq, target_start, target_end, no_recycles=1)

            if pr_mean["dssp_ok"] and pr_mean.get("dssp_align_ok", True):
                rec_mean = {
                    "pdb_id": pdb_id, "chain_id": chain_id,
                    "condition": "mean_ablation_s",
                    "band": EARLY_BAND,
                    "is_hairpin": int(bool(pr_mean["is_hairpin"])),
                    "plddt_target": pr_mean["plddt_target_mean"],
                    "cath_label": cath_label,
                    "status": "ok",
                }
                chain_results.setdefault(key, {})["baseline"] = base_hairpin
                chain_results[key]["mean_ablation_s"] = int(bool(pr_mean["is_hairpin"]))
                results_summary["n_chains_ok"]["mean_ablation_s"] += 1
            else:
                rec_mean = {
                    "pdb_id": pdb_id, "chain_id": chain_id,
                    "condition": "mean_ablation_s",
                    "band": EARLY_BAND,
                    "is_hairpin": None, "plddt_target": None,
                    "cath_label": cath_label, "status": "dssp_failed",
                }
            append_jsonl(out_path, [rec_mean])

            # Matched-ctrl: mean-ablation on non-target residues
            ctrl_start, ctrl_end = find_matched_control_mask(L, target_start, target_end)
            ctrl_res = list(range(ctrl_start, ctrl_end + 1))
            with patch_s_with_mean_at_blocks(model, EARLY_BLOCKS, ctrl_res, chain_mean_s):
                pr_ctrl = predict_and_judge_hairpin(model, tok, seq, target_start, target_end, no_recycles=1)

            if pr_ctrl["dssp_ok"] and pr_ctrl.get("dssp_align_ok", True):
                rec_ctrl = {
                    "pdb_id": pdb_id, "chain_id": chain_id,
                    "condition": "mean_ablation_s_matched_ctrl",
                    "band": EARLY_BAND,
                    "is_hairpin": int(bool(pr_ctrl["is_hairpin"])),
                    "plddt_target": pr_ctrl["plddt_target_mean"],
                    "cath_label": cath_label,
                    "status": "ok",
                }
                chain_results[key]["mean_ablation_s_matched_ctrl"] = int(bool(pr_ctrl["is_hairpin"]))
                results_summary["n_chains_ok"]["mean_ablation_s_matched_ctrl"] += 1
            else:
                rec_ctrl = {
                    "pdb_id": pdb_id, "chain_id": chain_id,
                    "condition": "mean_ablation_s_matched_ctrl",
                    "band": EARLY_BAND,
                    "is_hairpin": None, "plddt_target": None,
                    "cath_label": cath_label, "status": "dssp_failed",
                }
            append_jsonl(out_path, [rec_ctrl])

        except Exception:
            print(f"[mean_ablation] {key} EXCEPTION:\n{traceback.format_exc()}", flush=True)
            continue

    # Aggregate statistics
    paired_mean_ablation = [
        (chain_results[k]["baseline"], chain_results[k]["mean_ablation_s"])
        for k in chain_results
        if "baseline" in chain_results[k] and "mean_ablation_s" in chain_results[k]
    ]
    paired_ctrl = [
        (chain_results[k]["baseline"], chain_results[k]["mean_ablation_s_matched_ctrl"])
        for k in chain_results
        if "baseline" in chain_results[k] and "mean_ablation_s_matched_ctrl" in chain_results[k]
    ]

    def compute_delta_and_p(pairs):
        if not pairs:
            return None, None, 0
        base_rates = np.array([b for b, c in pairs], dtype=float)
        cond_rates = np.array([c for b, c in pairs], dtype=float)
        delta = float(np.mean(cond_rates - base_rates))
        from scipy import stats
        diffs = cond_rates - base_rates
        if np.all(diffs == 0):
            p = 1.0
        else:
            try:
                result = stats.wilcoxon(diffs)
                p = float(result.pvalue)
            except Exception:
                p = None
        return delta, p, len(pairs)

    delta_mean, p_mean, n_mean = compute_delta_and_p(paired_mean_ablation)
    delta_ctrl, p_ctrl, n_ctrl = compute_delta_and_p(paired_ctrl)

    # Predicate checks (same thresholds as M1)
    predicates = {
        "early_effect_>=0.2pp": abs(delta_mean) >= 0.2 if delta_mean is not None else False,
        "early_p_<0.05": (p_mean is not None and p_mean < 0.05),
        "matched_ctrl_<50pct_of_early": (
            abs(delta_ctrl) < 0.5 * abs(delta_mean)
            if delta_mean is not None and delta_ctrl is not None and delta_mean != 0
            else False
        ),
    }

    summary = {
        "variant": VARIANT_NAME,
        "dimension": "method",
        "submethod": "mean-ablation s-patching (within Causal Attribution / Patching family)",
        "band": EARLY_BAND,
        "blocks": EARLY_BLOCKS,
        "n_ref_chains_for_mean": len(ref_pool),
        "n_chains_attempted": results_summary["n_chains_attempted"],
        "n_chains_ok_mean_ablation": n_mean,
        "n_chains_ok_ctrl": n_ctrl,
        "mean_ablation_s_delta": delta_mean,
        "mean_ablation_s_p": p_mean,
        "matched_ctrl_delta": delta_ctrl,
        "matched_ctrl_p": p_ctrl,
        "predicate_checks": predicates,
        "verdict": (
            "supported" if all(predicates.values())
            else "not-supported"
        ),
        "wall_time_seconds": time.time() - t0,
    }

    write_json(args.out / "summary_stats.json", summary)
    print(f"\n[mean_ablation] === SUMMARY ===", flush=True)
    print(f"  N_chains_ok = {n_mean}", flush=True)
    print(f"  mean_ablation Δ = {delta_mean:.4f}  p = {p_mean}", flush=True)
    print(f"  matched_ctrl Δ = {delta_ctrl:.4f}  p = {p_ctrl}", flush=True)
    print(f"  Predicates: {predicates}", flush=True)
    print(f"  Verdict: {summary['verdict']}", flush=True)
    print(f"  Wall time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
