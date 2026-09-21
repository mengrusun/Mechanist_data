#!/usr/bin/env python3
"""Iteration 1 fix: L=16 random-direction control (C3) + full C4 4x4 selectivity at L=16.

Addresses reviewer's two top-priority ② fixes:
  (1) Missing L=16 random-direction control for C3 — is the M sign-inversion at
      alpha=+2 driven by the extracted direction, or by any large-magnitude
      perturbation in any direction? Norm-matched random unit directions
      (3 seeds) with the same alpha grid at L=16 should NOT reproduce the M
      sign-inversion or the V=A 5.5x amplification if the effect is truly
      direction-specific.
  (2) C4 4x4 selectivity at L=16 — the main M5 ran at picked shallow layers
      where sigma_proj was ~10x too small; C3 works at L=16 but C4 has never
      been swept there. Reuses raw v_hat_V (matching M4-supp).

Reuses runs/M_main_v1/artifacts/m2/{directions_raw.pt, heldout_activations.pt,
train_activations.pt} and the DG-1000 held-out prompts.

Outputs:
  runs/iteration_round_1/L16_random_control/{V}_L16_seed{s}_a{am}.json
  runs/iteration_round_1/L16_random_control/summary.json
  runs/iteration_round_1/L16_c4_selectivity/{V}_L16_a{am}.json
  runs/iteration_round_1/L16_c4_selectivity/selectivity_L16.json
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
# Reuse M4-supp/M4 primitives
from m4_steer_and_m5_selectivity import (  # type: ignore
    parse_transfer, format_prompt, load_model_and_tokenizer,
    register_steering, decode_batch, coherence_check, compute_sigma_proj,
)


def rand_unit_norm_matched(sigma_target, hidden_size, seed, device, dtype,
                           held_acts_layer):
    """Sample a random unit direction whose projection-std on the held-out set
    is normalized to unity, then rescale to match the target extracted direction's
    sigma_proj via the alpha grid multiplier. We return a *unit* vector and
    let the caller multiply by (am * sigma_target) — this matches the
    extraction convention used for v_hat_V.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    v = torch.randn(hidden_size, generator=g, dtype=torch.float32).to(device).to(dtype)
    # unit norm
    v = v / (v.norm() + 1e-9)
    return v


def compute_sigma_for_dir(held_pack, layer_idx, unit_dir):
    acts = held_pack["acts"]  # [N, L, H] on CPU (from torch.load)
    # Ensure unit_dir on same device/dtype as acts for the matmul.
    ud = unit_dir.detach().to(acts.device).to(torch.float32)
    return compute_sigma_proj(acts, layer_idx, ud)


def do_grid_point(V, unit_dir, sigma, am, ell_abs, model, tokenizer,
                  held_base_rows, held_partner_rows, batch_size,
                  max_new_tokens, pos_side_map):
    """Run one grid point at absolute layer ell_abs. Returns metrics dict."""
    alpha = am * sigma
    layer_hs = ell_abs if ell_abs > 0 else 1
    layers_to_hook = [layer_hs - 1]
    handles = []
    if am != 0:
        handles = register_steering(model, layers_to_hook, unit_dir, alpha)
    try:
        t0 = time.time()
        combined = held_base_rows + held_partner_rows
        gens = decode_batch(combined, tokenizer, model,
                            batch_size=batch_size, max_new_tokens=max_new_tokens)
        n_base = len(held_base_rows)
        base_gens = gens[:n_base]
        base_taus = [parse_transfer(g) for g in base_gens]
        valid = [t for t in base_taus if t is not None]
        n_parse_fail = sum(1 for t in base_taus if t is None)
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
        return dict(
            V=V, layer_abs=ell_abs, alpha_mult=am, alpha=float(alpha),
            sigma_proj=float(sigma),
            n_baseline=n_base,
            parse_failure_rate=(n_parse_fail / n_base if n_base else 0.0),
            mean_transfer=float(np.mean(valid)) if valid else float("nan"),
            std_transfer=float(np.std(valid)) if valid else float("nan"),
            v_effect=v_eff,
            w_effects=w_effects,
            coherence=coh,
            sample_generations=base_gens[:3],
            wall_time_s=time.time() - t0,
        )
    finally:
        for h in handles:
            h.remove()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--raw_dir", required=True)
    ap.add_argument("--held_acts", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_random", required=True,
                    help="Output dir for L=16 random-direction control results.")
    ap.add_argument("--out_c4", required=True,
                    help="Output dir for L=16 C4 selectivity results.")
    ap.add_argument("--ell_abs", type=int, default=16)
    ap.add_argument("--alpha_grid_random", default="-2,-1,0,1,2",
                    help="Sigma-mult grid for random-dir control (matches supp L=16).")
    ap.add_argument("--alpha_grid_c4", default="-2,0,2",
                    help="Sigma-mult grid for C4 selectivity at L=16 (matches main M5).")
    ap.add_argument("--seeds_random", default="0,1,2",
                    help="Random-direction seeds for the L=16 control (3 seeds).")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--n_sample", type=int, default=-1)
    ap.add_argument("--skip_random", action="store_true",
                    help="Skip the L=16 random-direction control block.")
    ap.add_argument("--skip_c4", action="store_true",
                    help="Skip the L=16 C4 selectivity block.")
    args = ap.parse_args()

    Path(args.out_random).mkdir(parents=True, exist_ok=True)
    Path(args.out_c4).mkdir(parents=True, exist_ok=True)

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
    held_pack = torch.load(args.held_acts, map_location="cpu", weights_only=False)
    layer_ids = raw_pack["layer_ids"]
    layer_to_idx = {li: i for i, li in enumerate(layer_ids)}
    if args.ell_abs not in layer_to_idx:
        raise RuntimeError(f"ell_abs={args.ell_abs} not in raw_pack layer_ids={layer_ids}")
    li = layer_to_idx[args.ell_abs]

    tokenizer, model = load_model_and_tokenizer(args.model)
    device = model.device
    dtype = torch.bfloat16
    hidden_size = raw_pack["raw"]["G"][li].numel()
    pos_side_map = {"G": "male", "A": "young", "I": "take-frame", "M": "no-meet"}

    alpha_grid_random = [int(x) for x in args.alpha_grid_random.split(",")]
    alpha_grid_c4 = [int(x) for x in args.alpha_grid_c4.split(",")]
    seeds_random = [int(x) for x in args.seeds_random.split(",")]

    # ---------- BLOCK 1: L=16 random-direction control (C3 paired null) ----------
    # For each V we sample a random unit direction per seed, but rescale via the
    # SAME sigma_proj magnitude the extracted v_hat_V uses at L=16. This makes
    # alpha*sigma the same expected residual perturbation size — the key
    # apples-to-apples comparison.
    random_summary = []
    if not args.skip_random:
        # Precompute the extracted-direction sigma_proj so random directions match magnitude.
        v_hat_sigmas = {}
        for V in ["G", "A", "I", "M"]:
            v = raw_pack["raw"][V][li].float()
            unit = v / (v.norm() + 1e-9)
            v_hat_sigmas[V] = compute_sigma_for_dir(held_pack, li, unit)
        print(f"[iter1-random] extracted-dir sigma_proj at L={args.ell_abs}: {v_hat_sigmas}", flush=True)

        for V in ["G", "A", "I", "M"]:
            # For each seed sample a fresh random direction; use v_hat_V's sigma as the magnitude ruler.
            sigma_ruler = v_hat_sigmas[V]
            for seed in seeds_random:
                unit_rand = rand_unit_norm_matched(sigma_ruler, hidden_size, seed,
                                                    device, dtype, held_pack["acts"][:, li, :])
                # For the random unit, compute its OWN sigma_proj (for the log; but
                # we scale alpha by sigma_ruler so alpha*sigma_ruler == alpha*sigma of v_hat).
                sigma_rand_actual = compute_sigma_for_dir(held_pack, li, unit_rand)
                print(f"[iter1-random] V={V} seed={seed} sigma_rand_actual={sigma_rand_actual:.4f} "
                      f"(ruler=sigma_v_hat={sigma_ruler:.4f})", flush=True)
                for am in alpha_grid_random:
                    res = do_grid_point(V, unit_rand, sigma_ruler, am, args.ell_abs,
                                        model, tokenizer, held_base_rows, held_partner_rows,
                                        args.batch_size, args.max_new_tokens, pos_side_map)
                    res.update(dict(kind="random_control", seed=seed,
                                    sigma_rand_actual=sigma_rand_actual,
                                    sigma_ruler=float(sigma_ruler),
                                    layer_abs=args.ell_abs))
                    fname = os.path.join(args.out_random,
                                          f"{V}_L{args.ell_abs}_seed{seed}_a{am:+d}.json")
                    with open(fname, "w") as f:
                        json.dump(res, f, indent=2)
                    random_summary.append(res)
                    print(f"  V={V} seed={seed} a={am:+d}: mean={res['mean_transfer']:.2f} "
                          f"v_eff={res['v_effect']:+.3f} parse={res['parse_failure_rate']:.2f}",
                          flush=True)
        with open(os.path.join(args.out_random, "summary.json"), "w") as f:
            json.dump(random_summary, f, indent=2)

    # ---------- BLOCK 2: L=16 C4 4x4 selectivity (raw v_hat, three alpha rows) ----------
    # Use raw v_hat_V (same as M4-supp) at L=16. For each V, we sweep alpha in
    # {-2sigma, 0, +2sigma} and record all four W's baseline effect shifts, so
    # we can assemble M[V,W] shift matrices and the c4-ratio for the L=16 setting.
    c4_summary = []
    if not args.skip_c4:
        v_hats = {}
        for V in ["G", "A", "I", "M"]:
            v = raw_pack["raw"][V][li].float()
            unit = v / (v.norm() + 1e-9)
            v_hats[V] = (unit, compute_sigma_for_dir(held_pack, li, unit))

        # baseline (alpha=0) can be shared across V; we still run once per V for coherence.
        for V in ["G", "A", "I", "M"]:
            unit, sigma = v_hats[V]
            for am in alpha_grid_c4:
                res = do_grid_point(V, unit, sigma, am, args.ell_abs, model, tokenizer,
                                    held_base_rows, held_partner_rows,
                                    args.batch_size, args.max_new_tokens, pos_side_map)
                res.update(dict(kind="c4_selectivity_raw", layer_abs=args.ell_abs))
                fname = os.path.join(args.out_c4, f"{V}_L{args.ell_abs}_a{am:+d}.json")
                with open(fname, "w") as f:
                    json.dump(res, f, indent=2)
                c4_summary.append(res)
                print(f"[iter1-c4] V={V} L={args.ell_abs} a={am:+d}: "
                      f"mean={res['mean_transfer']:.2f} v_eff={res['v_effect']:+.3f} "
                      f"w_effects={res['w_effects']}", flush=True)

        # Build the selectivity matrix M[V, W] = w_effects[W](+2sigma) - w_effects[W](0)
        # (mirrors the main M5 convention). Also build the -2sigma shift matrix.
        def _get(V, am):
            for r in c4_summary:
                if r["V"] == V and r["alpha_mult"] == am:
                    return r
            return None
        mat_plus = {V: {} for V in ["G","A","I","M"]}
        mat_minus = {V: {} for V in ["G","A","I","M"]}
        for V in ["G","A","I","M"]:
            r0 = _get(V, 0)
            rp = _get(V, +2)
            rm = _get(V, -2)
            for W in ["G","A","I","M"]:
                mat_plus[V][W] = rp["w_effects"][W] - r0["w_effects"][W] if r0 and rp else None
                mat_minus[V][W] = rm["w_effects"][W] - r0["w_effects"][W] if r0 and rm else None

        # c4 ratio = max_off / min_diag (absolute values, +2sigma matrix)
        diag = [abs(mat_plus[V][V]) for V in ["G","A","I","M"] if mat_plus[V][V] is not None]
        off  = [abs(mat_plus[V][W]) for V in ["G","A","I","M"]
                                  for W in ["G","A","I","M"]
                                  if V != W and mat_plus[V][W] is not None]
        min_diag = min(diag) if diag else float("nan")
        max_off  = max(off) if off else float("nan")
        c4_ratio = (max_off / min_diag) if (diag and min_diag > 0) else float("inf")

        # simple permutation test on off-diagonal >= diagonal (mirroring main M5)
        rng = np.random.default_rng(0)
        flat = np.array([[mat_plus[V][W] for W in ["G","A","I","M"]]
                          for V in ["G","A","I","M"]], dtype=float)
        # observed statistic: mean_diag - mean_off in |.|
        obs = float(np.mean(np.abs(np.diag(flat))) - np.mean(
            np.abs(flat[~np.eye(4, dtype=bool)])))
        n_perm = 2000
        greater = 0
        for _ in range(n_perm):
            perm = flat.flatten().copy()
            rng.shuffle(perm)
            perm = perm.reshape(4, 4)
            stat = float(np.mean(np.abs(np.diag(perm))) - np.mean(
                np.abs(perm[~np.eye(4, dtype=bool)])))
            if stat >= obs:
                greater += 1
        p_perm = (greater + 1) / (n_perm + 1)

        selectivity_out = dict(
            layer_abs=args.ell_abs,
            alpha_grid_c4=alpha_grid_c4,
            v_hat_sigmas={V: v_hats[V][1] for V in ["G","A","I","M"]},
            shift_matrix_alpha_plus2=mat_plus,
            shift_matrix_alpha_minus2=mat_minus,
            c4_ratio_max_off_over_min_diag=c4_ratio,
            min_abs_diagonal=min_diag,
            max_abs_off_diagonal=max_off,
            permutation_test=dict(
                obs_statistic=obs,
                n_permutations=n_perm,
                p_value=p_perm,
                null_hypothesis="mean_abs_diag <= mean_abs_off",
            ),
        )
        with open(os.path.join(args.out_c4, "selectivity_L16.json"), "w") as f:
            json.dump(selectivity_out, f, indent=2)
        print(f"[iter1-c4] c4_ratio={c4_ratio:.4f}  perm_p={p_perm:.4f}", flush=True)


if __name__ == "__main__":
    main()
