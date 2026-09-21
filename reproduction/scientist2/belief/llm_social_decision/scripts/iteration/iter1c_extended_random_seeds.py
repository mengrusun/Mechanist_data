#!/usr/bin/env python3
"""Iteration 2 fix (⓪+① combo): extend random-direction seeds from 3 to 10 for
the 4 showcase L=16 settings the paper actually cites:
  - V=M at alpha=+2 (sign inversion, headline C3 finding)
  - V=A at alpha=+2 (5.5x amplification from near-null baseline)
  - V=G at alpha=-2 (2.5x amplification)
  - V=I at alpha=-2 (46% amplification)

Reviewer's Suspicion C (Iteration 2): 3 random seeds may be too few. This test
adds 7 more seeds (4..10) at these 4 settings only, so the total null grows to
10 seeds — enough to characterize whether the extracted direction sits outside
the random-direction null with sub-2σ noise.

Compute: 4 settings × 7 new seeds × ~15s/point ≈ 7 min wall.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_HERE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from m4_steer_and_m5_selectivity import (  # type: ignore
    parse_transfer, format_prompt, load_model_and_tokenizer,
    register_steering, decode_batch, coherence_check, compute_sigma_proj,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--held_acts", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ell_abs", type=int, default=16)
    ap.add_argument("--extra_seeds", default="3,4,5,6,7,8,9")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    with open(args.data) as f:
        rows = [json.loads(l) for l in f]
    held_rows = [r for r in rows if r["split"] == "held"]
    held_base_rows = [r for r in held_rows if r["is_paired_partner_of"] is None]
    held_partner_rows = [r for r in held_rows if r["is_paired_partner_of"] is not None]

    raw_pack = torch.load(args.raw_dir, map_location="cpu", weights_only=False)
    held_pack = torch.load(args.held_acts, map_location="cpu", weights_only=False)
    layer_ids = raw_pack["layer_ids"]
    layer_to_idx = {li: i for i, li in enumerate(layer_ids)}
    li = layer_to_idx[args.ell_abs]

    tokenizer, model = load_model_and_tokenizer(args.model)
    device = model.device
    hidden_size = raw_pack["raw"]["G"][li].numel()
    pos_side_map = {"G": "male", "A": "young", "I": "take-frame", "M": "no-meet"}

    # showcase settings: (V, am)
    settings = [("M", 2), ("A", 2), ("G", -2), ("I", -2)]
    extra_seeds = [int(x) for x in args.extra_seeds.split(",")]

    # Precompute sigma_ruler per V (matching iter1 protocol)
    v_hat_sigmas = {}
    for V in ["G", "A", "I", "M"]:
        v = raw_pack["raw"][V][li].float()
        unit = v / (v.norm() + 1e-9)
        h_cpu = held_pack["acts"][:, li, :].float()
        v_hat_sigmas[V] = float((h_cpu @ unit).std().item())
    print(f"[iter1c] sigma_ruler = v_hat sigma_proj at L={args.ell_abs}: {v_hat_sigmas}",
          flush=True)

    all_res = []
    for V, am in settings:
        sigma_ruler = v_hat_sigmas[V]
        for seed in extra_seeds:
            g = torch.Generator(device="cpu").manual_seed(seed)
            v = torch.randn(hidden_size, generator=g, dtype=torch.float32).to(device).to(torch.bfloat16)
            unit_rand = v / (v.norm() + 1e-9)
            alpha = am * sigma_ruler
            layers_to_hook = [args.ell_abs - 1 if args.ell_abs > 0 else 0]
            handles = register_steering(model, layers_to_hook, unit_rand, alpha)
            try:
                t0 = time.time()
                combined = held_base_rows + held_partner_rows
                gens = decode_batch(combined, tokenizer, model,
                                    batch_size=args.batch_size,
                                    max_new_tokens=args.max_new_tokens)
                n_base = len(held_base_rows)
                base_gens = gens[:n_base]
                base_taus = [parse_transfer(g) for g in base_gens]
                valid = [t for t in base_taus if t is not None]
                pos_val = pos_side_map[V]
                pos = [t for r, t in zip(held_base_rows, base_taus)
                       if t is not None and r[V] == pos_val]
                neg = [t for r, t in zip(held_base_rows, base_taus)
                       if t is not None and r[V] != pos_val]
                v_eff = (float(np.mean(pos)) - float(np.mean(neg))
                         if pos and neg else 0.0)
                res = dict(
                    V=V, seed=seed, alpha_mult=am, alpha=float(alpha),
                    sigma_ruler=sigma_ruler,
                    n_baseline=n_base,
                    parse_failure_rate=(sum(1 for t in base_taus if t is None) / n_base
                                        if n_base else 0.0),
                    mean_transfer=float(np.mean(valid)) if valid else float("nan"),
                    v_effect=v_eff,
                    layer_abs=args.ell_abs,
                    wall_time_s=time.time() - t0,
                    kind="extended_random_control",
                )
                fname = os.path.join(args.out,
                                     f"{V}_L{args.ell_abs}_seed{seed}_a{am:+d}.json")
                with open(fname, "w") as f:
                    json.dump(res, f, indent=2)
                all_res.append(res)
                print(f"[iter1c] V={V} seed={seed} a={am:+d}: mean={res['mean_transfer']:.2f} "
                      f"v_eff={v_eff:+.3f}", flush=True)
            finally:
                for h in handles:
                    h.remove()

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(all_res, f, indent=2)


if __name__ == "__main__":
    main()
