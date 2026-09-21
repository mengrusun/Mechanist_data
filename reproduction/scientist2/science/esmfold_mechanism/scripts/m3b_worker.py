#!/usr/bin/env python3
"""M3b worker — Charge steering dose-response + same/opposite paired diff.

Reads M3a's `v_charge.npy` and `best_block.json` to know direction + block.
Reads calibration split to compute σ_proj = std(s_block^T v_charge) — coefficients
are expressed as β · σ_proj (the steering-coefficient-tuning tip's convention).

Per chain: for each target-pair (i, j) in cross_strand_pairs (use FIRST cross-pair;
if none, skip chain) plus a matched-control pair (i', j') outside target region:
  - baseline (1)
  - For α ∈ {-3, -1, 0, +1, +3} σ (5 points):
    - config same: +α to both residues (2 residues per intervention)  → 5 forwards
    - config opposite: +α to res_i, -α to res_j                        → 5 forwards
    - Random-direction control at α=±3σ, same config only              → 2 forwards
    - Matched-control (non-target pair), same config, α=±3σ only       → 2 forwards
Per chain: 1 + 10 + 2 + 2 = 15 forwards. 200 chains × 15 = 3000 forwards / 4 workers.
At ~2s/forward → ~1500s per worker = 25 min. Fits in 1.5-h plan estimate.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))

from esmfold_lib import (  # noqa: E402
    PROJECT_ROOT, MANIFEST_PATH, load_esmfold, esmfold_forward,
    predict_and_judge_hairpin, additive_steering_s, collect_s_z_at_block,
    S_HIDDEN, set_seed, write_jsonl, append_jsonl, write_json,
)


def compute_sigma_proj(model, tok, calib_chains: List[Dict], v_charge: torch.Tensor,
                      block_idx: int) -> float:
    """Compute σ_proj = std(s[block, all_residues, :] @ v_charge) over calibration chains."""
    device = next(model.parameters()).device
    v = v_charge.to(device)
    projections = []
    for i, c in enumerate(calib_chains):
        try:
            with collect_s_z_at_block(model, block_idx) as store:
                _ = esmfold_forward(model, tok, c["seq"])
            s = store['s'][0]  # (L, 1024)
            proj = (s.to(device) @ v).cpu().numpy()  # (L,)
            projections.append(proj)
        except (torch.cuda.OutOfMemoryError, Exception) as e:
            print(f"[m3b/sigma] error on {c['pdb_id']}: {e}", flush=True)
            torch.cuda.empty_cache()
            continue
        if i >= 30:  # first 30 chains enough for σ estimate
            break
    all_proj = np.concatenate(projections)
    return float(all_proj.std())


def find_matched_control_pair(L: int, target_start: int, target_end: int,
                             target_pair: Tuple[int, int]) -> Optional[Tuple[int, int]]:
    """Return a residue pair outside [target_start..target_end] mirroring the target pair's
    intra-strand distance. Returns None if not fittable.
    """
    i, j = target_pair
    d = j - i
    # Try after
    a = target_end + 6
    b = a + d
    if b < L - 3:
        return a, b
    # Try before
    b2 = target_start - 6
    a2 = b2 - d
    if a2 >= 3:
        return a2, b2
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worker-id", type=int, required=True)
    p.add_argument("--start", type=int, required=True)
    p.add_argument("--stop", type=int, required=True)
    p.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    p.add_argument("--v-charge", type=Path,
                   default=PROJECT_ROOT / "results" / "M3a" / "v_charge.npy")
    p.add_argument("--best-block-json", type=Path,
                   default=PROJECT_ROOT / "results" / "M3a" / "best_block.json")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "results" / "M3b")
    p.add_argument("--betas", type=str, default="-3,-1,0,1,3",
                   help="β multipliers on σ_proj; 0 is identity control")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed + args.worker_id)
    args.out.mkdir(parents=True, exist_ok=True)

    # Load v_charge + best block
    v_np = np.load(args.v_charge)
    v_charge = torch.tensor(v_np, dtype=torch.float32)
    best_block_info = json.loads(open(args.best_block_json).read())
    block_idx = int(best_block_info["best_block"])
    print(f"[m3b-{args.worker_id}] best block from M3a: {block_idx}", flush=True)
    print(f"[m3b-{args.worker_id}] v_charge norm: {np.linalg.norm(v_np):.4f}", flush=True)

    # Load chains
    main_chains = []
    calib_chains = []
    with open(args.manifest) as f:
        for line in f:
            r = json.loads(line)
            if r.get("split") == "main":
                main_chains.append(r)
            elif r.get("split") == "calibration":
                calib_chains.append(r)
    my_chains = main_chains[args.start:args.stop]
    print(f"[m3b-{args.worker_id}] {len(my_chains)} main chains in slice", flush=True)
    print(f"[m3b-{args.worker_id}] {len(calib_chains)} calibration chains available", flush=True)

    print(f"[m3b-{args.worker_id}] loading model...", flush=True)
    model, tok, _ = load_esmfold(device="cuda", dtype=torch.float32)
    print(f"[m3b-{args.worker_id}] model loaded", flush=True)

    # Compute σ_proj using calibration (only worker 0 computes; workers 1-3 read from disk)
    sigma_path = args.out / f"sigma_proj_block_{block_idx}.json"
    if args.worker_id == 0 or not sigma_path.exists():
        # If worker 0, compute freshly. Others: try to reuse — if worker 0 hasn't finished
        # yet, fall back to computing locally (small overhead, ~30 chains).
        if calib_chains:
            print(f"[m3b-{args.worker_id}] computing σ_proj on calibration set...", flush=True)
            sigma = compute_sigma_proj(model, tok, calib_chains, v_charge, block_idx)
            print(f"[m3b-{args.worker_id}] σ_proj = {sigma:.4f}", flush=True)
        else:
            sigma = 1.0  # fallback, no scaling
        if args.worker_id == 0:
            write_json(sigma_path, {"sigma_proj": sigma, "block": block_idx})
    else:
        sigma = float(json.loads(open(sigma_path).read())["sigma_proj"])
        print(f"[m3b-{args.worker_id}] σ_proj (from disk) = {sigma:.4f}", flush=True)

    # Parse betas
    betas = [float(b) for b in args.betas.split(",")]

    # Build a random control direction with the same L2 norm as v_charge — held CONST
    # across the entire run for reproducibility.
    rng = np.random.default_rng(args.seed)
    v_rand = rng.normal(size=v_np.shape).astype(np.float32)
    v_rand *= (np.linalg.norm(v_np) / (np.linalg.norm(v_rand) + 1e-8))
    v_rand_t = torch.tensor(v_rand, dtype=torch.float32)

    device = next(model.parameters()).device
    v_charge_dev = v_charge.to(device)
    v_rand_dev = v_rand_t.to(device)

    out_path = args.out / f"steering_effect_worker_{args.worker_id}.jsonl"
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
        cross_pairs = [tuple(x) for x in chain_r.get("cross_strand_pairs", [])]
        cath_label = chain_r.get("cath_label")

        if not cross_pairs:
            continue
        # Use the FIRST cross-strand pair as the target for steering
        target_pair = cross_pairs[0]
        i_tgt, j_tgt = target_pair
        # Matched-control pair
        mctrl = find_matched_control_pair(L, target_start, target_end, target_pair)

        elapsed = time.time() - t0
        rate = elapsed / max(1, ci)
        eta = rate * (len(my_chains) - ci) / 60
        print(f"[m3b-{args.worker_id}] {ci}/{len(my_chains)}  {pdb_id}_{chain_id} L={L}  "
              f"target_pair={target_pair}  mctrl={mctrl}  ({rate:.1f}s/chain, eta {eta:.1f}min)",
              flush=True)

        try:
            def _write_steer_early(cond, beta, config, direction, pair, pr, extra=None):
                # Bootstrap helper — will be overwritten by nested defn below,
                # but define here so baseline can use it via the same code path.
                # (Kept for clarity — the nested closure below is the canonical one.)
                pass
            # Baseline
            base = predict_and_judge_hairpin(model, tok, seq, target_start, target_end,
                                             no_recycles=1)
            total_forwards += 1
            if not base["dssp_ok"]:
                continue
            append_jsonl(out_path, [{
                "pdb_id": pdb_id, "chain_id": chain_id, "condition": "baseline",
                "beta": None, "alpha_scaled": None,
                "config": None, "direction": None, "pair": target_pair,
                "is_hairpin": int(bool(base["is_hairpin"])),
                "plddt_target": base["plddt_target_mean"],
                "target_start": target_start, "target_end": target_end,
                "L": L, "cath_label": cath_label,
                "sigma_proj": sigma, "block": block_idx,
                "status": "ok",
            }])

            # For each β, run same and opposite config on target pair
            for beta in betas:
                alpha = beta * sigma
                for config in ("same", "opposite"):
                    if config == "same":
                        coeffs = [alpha, alpha]
                    else:
                        coeffs = [alpha, -alpha]
                    with additive_steering_s(model, block_idx, [i_tgt, j_tgt],
                                              v_charge_dev, coeffs):
                        pr = predict_and_judge_hairpin(model, tok, seq, target_start,
                                                        target_end, no_recycles=1)
                        total_forwards += 1
                    rec = {
                        "pdb_id": pdb_id, "chain_id": chain_id, "condition": "steer_target",
                        "beta": beta, "alpha_scaled": alpha,
                        "config": config, "direction": "v_charge",
                        "pair": target_pair,
                        "cath_label": cath_label,
                        "sigma_proj": sigma, "block": block_idx,
                    }
                    if pr["dssp_ok"] and pr.get("dssp_align_ok", True):
                        rec.update({"is_hairpin": int(bool(pr["is_hairpin"])),
                                    "plddt_target": pr["plddt_target_mean"],
                                    "status": "ok"})
                    else:
                        reason = "dssp_failed" if not pr["dssp_ok"] else "dssp_misaligned"
                        rec.update({"is_hairpin": None, "plddt_target": None,
                                    "status": reason})
                    append_jsonl(out_path, [rec])

            def _write_steer(cond, beta, config, direction, pair, pr, extra=None):
                rec = {
                    "pdb_id": pdb_id, "chain_id": chain_id, "condition": cond,
                    "beta": beta, "alpha_scaled": beta * sigma,
                    "config": config, "direction": direction, "pair": pair,
                    "cath_label": cath_label,
                    "sigma_proj": sigma, "block": block_idx,
                }
                if extra:
                    rec.update(extra)
                if pr and pr["dssp_ok"] and pr.get("dssp_align_ok", True):
                    rec.update({"is_hairpin": int(bool(pr["is_hairpin"])),
                                "plddt_target": pr["plddt_target_mean"],
                                "status": "ok"})
                else:
                    reason = "not_attempted" if pr is None else \
                             ("dssp_failed" if not pr["dssp_ok"] else "dssp_misaligned")
                    rec.update({"is_hairpin": None, "plddt_target": None, "status": reason})
                append_jsonl(out_path, [rec])

            # Random-direction control at ALL β (same config, target pair)
            for beta in betas:
                alpha = beta * sigma
                coeffs = [alpha, alpha]
                with additive_steering_s(model, block_idx, [i_tgt, j_tgt],
                                          v_rand_dev, coeffs):
                    pr = predict_and_judge_hairpin(model, tok, seq, target_start,
                                                    target_end, no_recycles=1)
                    total_forwards += 1
                _write_steer("steer_random", beta, "same", "v_random", target_pair, pr)

            # Matched-control pair steering at ALL β × both configs
            if mctrl is not None:
                for beta in betas:
                    alpha = beta * sigma
                    for config in ("same", "opposite"):
                        coeffs = [alpha, alpha] if config == "same" else [alpha, -alpha]
                        with additive_steering_s(model, block_idx, [mctrl[0], mctrl[1]],
                                                  v_charge_dev, coeffs):
                            pr = predict_and_judge_hairpin(model, tok, seq, target_start,
                                                            target_end, no_recycles=1)
                            total_forwards += 1
                        _write_steer("steer_matched_ctrl", beta, config, "v_charge", mctrl, pr)
            else:
                # Write NA rows so pairing analysis is honest
                for beta in betas:
                    for config in ("same", "opposite"):
                        _write_steer("steer_matched_ctrl", beta, config, "v_charge", None, None,
                                     extra={"na_reason": "no_mctrl_fit"})
        except torch.cuda.OutOfMemoryError:
            print(f"[m3b-{args.worker_id}] OOM at {pdb_id}", flush=True)
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"[m3b-{args.worker_id}] error at {pdb_id}: {e}", flush=True)
            traceback.print_exc()
            continue

    write_json(args.out / f"m3b_worker_{args.worker_id}_summary.json", {
        "worker_id": args.worker_id, "slice": [args.start, args.stop],
        "total_forwards": total_forwards, "elapsed_sec": time.time() - t0,
        "sigma_proj": sigma, "block": block_idx, "betas": betas,
    })
    print(f"[m3b-{args.worker_id}] DONE. forwards={total_forwards} "
          f"time={(time.time()-t0)/60:.1f}min", flush=True)


if __name__ == "__main__":
    main()
