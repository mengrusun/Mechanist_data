#!/usr/bin/env python3
"""Supplementary M4 run at DEEPER (mid-network) layers.

The main M4 picked ell_V^* by max probe cv_acc — which was 1.0 at very early layers
(layer 2/4/6) due to token-embedding-level linearity of the paired-partner design.
The projection-transfer regression on the held-out showed that layer-2 also has large
|beta| for I, M, but the sigma_proj at those layers is very small (~0.01), so
`alpha * sigma_proj * direction` at ±4 sigma injects a magnitude of only ~0.04 on
the residual — insufficient to produce a measurable transfer shift.

This supplementary run tests C3 / C4 at MID-NETWORK layers (per experiment-tips
general rule "mid layers usually work best"), re-extracting v_hat_V at layer L_MID
and using an alpha grid expressed both in sigma_proj units AND absolute units to
compare with the main M4.
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
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from m4_steer_and_m5_selectivity import (  # type: ignore
    parse_transfer, format_prompt, load_model_and_tokenizer,
    SteeringHook, DirectionalAblationHook,
    register_steering, decode_batch, coherence_check, compute_sigma_proj,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--train_acts", required=True)
    ap.add_argument("--held_acts", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_m4", required=True)
    ap.add_argument("--out_m5", required=True)
    ap.add_argument("--layers", default="12,16,20",
                    help="Comma-separated ABSOLUTE layer indices (0..32) to sweep.")
    ap.add_argument("--alpha_grid", default="-4,-2,-1,0,1,2,4",
                    help="Sigma-multiplier grid.")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--n_sample", type=int, default=-1)
    args = ap.parse_args()

    Path(args.out_m4).mkdir(parents=True, exist_ok=True)
    Path(args.out_m5).mkdir(parents=True, exist_ok=True)

    # Load data
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

    raw_pack = torch.load(args.raw_dir, map_location="cpu", weights_only=False)
    train_pack = torch.load(args.train_acts, map_location="cpu", weights_only=False)
    held_pack = torch.load(args.held_acts, map_location="cpu", weights_only=False)
    layer_ids = raw_pack["layer_ids"]
    layer_to_idx = {li: i for i, li in enumerate(layer_ids)}

    tokenizer, model = load_model_and_tokenizer(args.model)
    pos_side_map = {"G": "male", "A": "young", "I": "take-frame", "M": "no-meet"}
    layers_to_test = [int(x) for x in args.layers.split(",")]
    alpha_grid = [int(x) for x in args.alpha_grid.split(",")]

    all_res = []
    for V in ["G", "A", "I", "M"]:
        for L in layers_to_test:
            if L not in layer_to_idx:
                continue
            li = layer_to_idx[L]
            # v_hat_V at THIS layer (not the ell_V* from M2's pick)
            v = raw_pack["raw"][V][li].float()
            if v.norm() < 1e-6:
                continue
            unit = v / v.norm()
            sigma = compute_sigma_proj(held_pack["acts"], li, unit)
            print(f"[m4-sup] V={V} L={L} |v|={float(v.norm()):.4f} sigma_proj={sigma:.4f}", flush=True)
            for am in alpha_grid:
                alpha = am * sigma
                fname = os.path.join(args.out_m4, f"{V}_L{L}_a{am:+d}.json")
                handles = []
                if am != 0:
                    handles = register_steering(model, [L - 1 if L > 0 else 0],
                                                 unit, alpha)
                try:
                    t0 = time.time()
                    combined = held_base_rows + held_partner_rows
                    gens = decode_batch(combined, tokenizer, model,
                                        args.batch_size, args.max_new_tokens)
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
                    v_eff = (float(np.mean(pos)) - float(np.mean(neg))
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
                    coh = coherence_check(base_gens[:10])
                    res = dict(
                        V=V, decorrelator="raw", site="single", layer_abs=L,
                        alpha_mult=am, alpha=alpha, sigma_proj=sigma,
                        n_baseline=n_base,
                        parse_failure_rate=float(parse_fail),
                        mean_transfer=float(np.mean(valid)) if valid else float("nan"),
                        std_transfer=float(np.std(valid)) if valid else float("nan"),
                        v_effect=v_eff, w_effects=w_effects,
                        coherence=coh,
                        sample_generations=base_gens[:3],
                        wall_time_s=time.time() - t0,
                    )
                    with open(fname, "w") as f:
                        json.dump(res, f, indent=2)
                    all_res.append(res)
                    print(f"  L={L} a={am:+d}: mean={res['mean_transfer']:.2f} "
                          f"v_eff={v_eff:+.3f} parse={parse_fail:.2f}", flush=True)
                finally:
                    for h in handles:
                        h.remove()
    with open(os.path.join(args.out_m4, "supp_summary.json"), "w") as f:
        json.dump(all_res, f, indent=2)


if __name__ == "__main__":
    main()
