#!/usr/bin/env python3
"""M6 — baselines & ablations (all four claims, robustness).

B1: raw v_hat_V (no decorrelation) — steering with the raw direction. 4V x 3alpha = 12 runs.
B2: random unit direction, norm-matched. 4V x 3alpha x 3 seeds = 36 runs.
B3: mean-centred v_hat_V only. 4V x 3alpha = 12 runs.
B4: directional ablation h <- (I - u u^T) h. 4V = 4 runs.

All at site = single-layer S = {ell_V*}, alpha ∈ {-2 sigma, 0, +2 sigma} (except B4
which has no alpha).  Reuses the CAA hook infrastructure from m4_steer_and_m5.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import numpy as np
import torch

# Reuse everything from the m4 script by importing.
import sys
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from m4_steer_and_m5_selectivity import (  # type: ignore
    LLAMA31_INSTRUCT_TEMPLATE,
    parse_transfer, format_prompt, load_model_and_tokenizer,
    SteeringHook, DirectionalAblationHook,
    register_steering, register_ablation,
    decode_batch, coherence_check, compute_sigma_proj,
)


def run_one(V, hook_kind, decorr_key, alpha_mult, seed, unit_dir, sigma_proj,
             ell_star, model, tokenizer, held_base_rows, held_partner_rows,
             batch_size, max_new_tokens, coherence_k, pos_side_map, out_dir):
    """Return dict. Writes JSON."""
    fname = os.path.join(
        out_dir,
        f"{hook_kind}_{V}_{decorr_key}_a{alpha_mult:+d}_s{seed}.json",
    )
    alpha = alpha_mult * sigma_proj if sigma_proj is not None else 0.0
    layer_hs = ell_star if ell_star > 0 else 1
    layers_to_hook = [layer_hs - 1]
    handles = []
    if hook_kind == "steer" and alpha_mult != 0:
        handles = register_steering(model, layers_to_hook, unit_dir, alpha)
    elif hook_kind == "ablate":
        handles = register_ablation(model, layers_to_hook, unit_dir)
    try:
        t0 = time.time()
        combined = held_base_rows + held_partner_rows
        gens = decode_batch(combined, tokenizer, model, batch_size, max_new_tokens)
        n_base = len(held_base_rows)
        base_gens = gens[:n_base]
        base_taus = [parse_transfer(g) for g in base_gens]
        valid = [t for t in base_taus if t is not None]
        parse_fail = sum(1 for t in base_taus if t is None) / max(1, n_base)
        pos_val = pos_side_map[V]
        pos = [t for r, t in zip(held_base_rows, base_taus)
               if t is not None and r[V] == pos_val]
        neg = [t for r, t in zip(held_base_rows, base_taus)
               if t is not None and r[V] != pos_val]
        v_effect = (float(np.mean(pos)) - float(np.mean(neg))
                    if pos and neg else 0.0)
        w_effects = {}
        for W in ["G", "A", "I", "M"]:
            pvw = pos_side_map[W]
            pw = [t for r, t in zip(held_base_rows, base_taus)
                  if t is not None and r[W] == pvw]
            nw = [t for r, t in zip(held_base_rows, base_taus)
                  if t is not None and r[W] != pvw]
            w_effects[W] = (float(np.mean(pw)) - float(np.mean(nw))
                            if pw and nw else 0.0)
        coh = coherence_check(base_gens[:coherence_k])
        res = dict(
            hook_kind=hook_kind, V=V, decorrelator=decorr_key,
            alpha_mult=alpha_mult, alpha=alpha, seed=seed,
            ell_star=ell_star, layers_hooked=layers_to_hook,
            n_baseline=n_base,
            n_baseline_parsed=len(valid),
            parse_failure_rate=float(parse_fail),
            mean_transfer=float(np.mean(valid)) if valid else float("nan"),
            std_transfer=float(np.std(valid)) if valid else float("nan"),
            v_effect=v_effect,
            w_effects=w_effects,
            coherence=coh,
            sample_generations=base_gens[:3],
            wall_time_s=time.time() - t0,
        )
        with open(fname, "w") as f:
            json.dump(res, f, indent=2)
        return res
    finally:
        for h in handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--pure_dir", required=True)
    ap.add_argument("--layer_pick", required=True)
    ap.add_argument("--held_acts", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--alpha_grid", default="-2,0,2")
    ap.add_argument("--random_seeds", default="0,1,2")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--coherence_k", type=int, default=10)
    ap.add_argument("--n_sample", type=int, default=-1)
    ap.add_argument("--blocks", default="B1,B2,B3,B4")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)
    with open(args.data) as f:
        rows = [json.loads(l) for l in f]
    held_rows = [r for r in rows if r["split"] == "held"]
    held_base_rows = [r for r in held_rows if r["is_paired_partner_of"] is None]
    held_partner_rows = [r for r in held_rows if r["is_paired_partner_of"] is not None]
    if args.n_sample > 0:
        held_base_rows = held_base_rows[: args.n_sample]
        keep = {r["trial_id"] for r in held_base_rows}
        held_partner_rows = [r for r in held_partner_rows
                             if r["is_paired_partner_of"] in keep]

    with open(args.layer_pick) as f:
        pick = json.load(f)
    raw_pack = torch.load(args.raw_dir, map_location="cpu", weights_only=False)
    pure_pack = torch.load(args.pure_dir, map_location="cpu", weights_only=False)
    held_pack = torch.load(args.held_acts, map_location="cpu", weights_only=False)
    layer_ids = raw_pack["layer_ids"]
    layer_to_idx = {li: i for i, li in enumerate(layer_ids)}

    tokenizer, model = load_model_and_tokenizer(args.model)

    pos_side_map = {"G": "male", "A": "young", "I": "take-frame", "M": "no-meet"}

    alpha_grid = [int(x) for x in args.alpha_grid.split(",")]
    seeds = [int(s) for s in args.random_seeds.split(",")]
    blocks = args.blocks.split(",")

    all_res = []
    for V in ["G", "A", "I", "M"]:
        if V not in pick:
            continue
        ell_star = pick[V]["ell_star"]
        li = layer_to_idx[ell_star]
        v_raw = raw_pack["raw"][V][li].float()
        v_mc = raw_pack["mean_centered"][V][li].float()
        v_leace = pure_pack["pure"][V].get("leace")
        v_gs = pure_pack["pure"][V].get("gs")

        # -- B1: raw --------------------------------------------------------- #
        if "B1" in blocks:
            unit = v_raw / (v_raw.norm() + 1e-9)
            sigma = compute_sigma_proj(held_pack["acts"], li, unit)
            for am in alpha_grid:
                r = run_one(V, "steer", "raw", am, 0, unit, sigma, ell_star,
                            model, tokenizer, held_base_rows, held_partner_rows,
                            args.batch_size, args.max_new_tokens, args.coherence_k,
                            pos_side_map, args.out)
                all_res.append(r)
        # -- B2: random unit direction, norm-matched to raw ------------------- #
        if "B2" in blocks:
            for seed in seeds:
                gen = torch.Generator().manual_seed(seed * 10 + hash(V) % 100)
                rand = torch.randn(v_raw.shape[0], generator=gen)
                unit_rand = rand / rand.norm()
                sigma = compute_sigma_proj(held_pack["acts"], li, unit_rand)
                for am in alpha_grid:
                    r = run_one(V, "steer", "random", am, seed, unit_rand, sigma,
                                ell_star, model, tokenizer,
                                held_base_rows, held_partner_rows,
                                args.batch_size, args.max_new_tokens,
                                args.coherence_k, pos_side_map, args.out)
                    all_res.append(r)
        # -- B3: mean-centred ------------------------------------------------- #
        if "B3" in blocks:
            unit = v_mc / (v_mc.norm() + 1e-9)
            sigma = compute_sigma_proj(held_pack["acts"], li, unit)
            for am in alpha_grid:
                r = run_one(V, "steer", "mean_centered", am, 0, unit, sigma,
                            ell_star, model, tokenizer,
                            held_base_rows, held_partner_rows,
                            args.batch_size, args.max_new_tokens,
                            args.coherence_k, pos_side_map, args.out)
                all_res.append(r)
        # -- B4: directional ablation on LEACE pure --------------------------- #
        if "B4" in blocks:
            if v_leace is not None and v_leace.norm() > 1e-4:
                unit = v_leace / (v_leace.norm() + 1e-9)
                r = run_one(V, "ablate", "leace", 0, 0, unit, None, ell_star,
                            model, tokenizer, held_base_rows, held_partner_rows,
                            args.batch_size, args.max_new_tokens,
                            args.coherence_k, pos_side_map, args.out)
                all_res.append(r)

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"[m6] done: {len(all_res)} runs")


if __name__ == "__main__":
    main()
