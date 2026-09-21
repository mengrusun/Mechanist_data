#!/usr/bin/env python3
"""M4 + M5 monolithic: CAA-style activation-addition steering with signed alpha,
plus the 4x4 selectivity matrix on top.

Loads Llama-3.1-8B-Instruct once, keeps it resident, and iterates the
V x alpha_mult x decorrelator x site grid from EXPERIMENT_PLAN.md.  For each grid
point it decodes greedy transfers on the 200-held-out set, parses the integer,
and logs mean/median/std, dose-response signals, coherence-gate signals.

At the end it aggregates the 4x4 selectivity matrix M[V, W] = shift_of_W_effect(alpha)
at the pre-registered (alpha in {-2 sigma, 0, +2 sigma}, LEACE, single-site) grid,
plus a permutation test on "off-diagonal = diagonal".

Outputs per grid point: artifacts/m4/{V}_{decorr}_{site}_a{alpha_mult:+d}.json
Aggregate: artifacts/m4/summary.json, artifacts/m5/selectivity_matrix.json,
           artifacts/m5/permutation_test.json.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch


LLAMA31_INSTRUCT_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
)


ANSWER_INT_RE = re.compile(r"(-?\d{1,3})")


def parse_transfer(text):
    if text is None:
        return None
    first_line = text.strip().split("\n", 1)[0]
    m = ANSWER_INT_RE.search(first_line)
    if m:
        try:
            v = int(m.group(1))
            if 0 <= v <= 20:
                return v
        except ValueError:
            pass
    head = text.strip()[:60]
    m = ANSWER_INT_RE.search(head)
    if m:
        try:
            v = int(m.group(1))
            if 0 <= v <= 20:
                return v
        except ValueError:
            pass
    return None


def format_prompt(user):
    return LLAMA31_INSTRUCT_TEMPLATE.format(user=user)


def load_model_and_tokenizer(model_path):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.eval()
    return tok, model


class SteeringHook:
    """Add `alpha * unit_direction` to every position of the residual output of a
    layer.  This matches the Steering Vectors demo `apply_steering_hook`: the
    intervention is applied to all sequence positions (CAA convention).

    Note: `model.model.layers[l].register_forward_hook` receives the layer's OUTPUT,
    which is a tuple whose first entry is [B, T, H] and represents the post-residual
    hidden state at "layer index l+1" in `hidden_states` (i.e. after residual + FFN
    of block l).  This aligns with our extraction convention.
    """

    def __init__(self, unit_dir, alpha):
        # unit_dir: torch.Tensor of shape [H] on the model's dtype/device
        self.unit_dir = unit_dir
        self.alpha = alpha

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            hidden = out[0]
            hidden = hidden + self.alpha * self.unit_dir.to(hidden.dtype).to(hidden.device)
            return (hidden,) + out[1:]
        else:
            return out + self.alpha * self.unit_dir.to(out.dtype).to(out.device)


class DirectionalAblationHook:
    """h <- (I - u u^T) h at each position of the residual output of a layer."""
    def __init__(self, unit_dir):
        self.unit_dir = unit_dir

    def __call__(self, module, inp, out):
        if isinstance(out, tuple):
            hidden = out[0]
        else:
            hidden = out
        u = self.unit_dir.to(hidden.dtype).to(hidden.device)
        # proj = (h . u) * u
        proj = (hidden @ u).unsqueeze(-1) * u
        hidden = hidden - proj
        if isinstance(out, tuple):
            return (hidden,) + out[1:]
        return hidden


def register_steering(model, layers, unit_dir, alpha):
    handles = []
    for li in layers:
        # `layers` here refers to *decoder block index* (0..n_layers-1).
        hook = SteeringHook(unit_dir, alpha)
        h = model.model.layers[li].register_forward_hook(hook)
        handles.append(h)
    return handles


def register_ablation(model, layers, unit_dir):
    handles = []
    for li in layers:
        hook = DirectionalAblationHook(unit_dir)
        h = model.model.layers[li].register_forward_hook(hook)
        handles.append(h)
    return handles


@torch.no_grad()
def decode_batch(rows, tokenizer, model, batch_size, max_new_tokens=8):
    outs = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        formatted = [format_prompt(r["prompt"]) for r in batch]
        enc = tokenizer(formatted, return_tensors="pt", padding=True,
                        truncation=True, max_length=1024).to(model.device)
        in_len = enc["input_ids"].shape[1]
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None, top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        for row in gen[:, in_len:]:
            outs.append(tokenizer.decode(row, skip_special_tokens=True))
    return outs


def coherence_check(texts):
    """Report format_ok_rate + 5-gram repetition rate."""
    n = len(texts)
    if n == 0:
        return dict(format_ok_rate=0.0, mean_5gram_rep=0.0, n_samples=0)
    format_ok = sum(1 for t in texts if parse_transfer(t) is not None) / n
    reps = []
    for t in texts:
        toks = t.split()
        if len(toks) < 5:
            reps.append(0.0)
            continue
        grams = [tuple(toks[i:i+5]) for i in range(len(toks) - 4)]
        if not grams:
            reps.append(0.0)
        else:
            uniq = len(set(grams))
            reps.append(1.0 - uniq / len(grams))
    return dict(format_ok_rate=float(format_ok),
                mean_5gram_rep=float(np.mean(reps)),
                n_samples=n)


def _pick_unit_dir(directions_pure, V, decorr, layer_idx_in_extracted,
                    raw_pack, layer_pick_json, layer_to_idx, per_layer=False):
    """Return the unit direction to steer with for (V, decorr) at V's chosen layer."""
    if not per_layer:
        v = directions_pure[V].get(decorr)
        if v is None:
            return None
        return v / (v.norm() + 1e-9)
    else:
        # Some decorrelators are per-layer (raw / mean_centered were per-layer in M2);
        # for the pure directions we only stored at ell_V*.  If called with per_layer,
        # fall back to the ell_V* choice.
        return _pick_unit_dir(directions_pure, V, decorr, layer_idx_in_extracted,
                              raw_pack, layer_pick_json, layer_to_idx, per_layer=False)


def compute_sigma_proj(held_acts, layer_idx, unit_dir):
    """sigma_proj = std of (h_l . u) on the held-out set at the extraction layer."""
    h = held_acts[:, layer_idx, :].float()
    proj = h @ unit_dir
    return float(proj.std().item())


def run_one_grid_point(V, decorr, alpha_mult, site, sigma_proj, ell_star,
                       unit_dir_extract, unit_dir_steer, model, tokenizer,
                       held_base_rows, held_partner_rows, batch_size,
                       max_new_tokens, coherence_k, pos_side_map, out_dir):
    """Return dict with metrics. Also writes JSON to disk."""
    fname = os.path.join(out_dir, f"{V}_{decorr}_{site}_a{alpha_mult:+d}.json")
    alpha = alpha_mult * sigma_proj
    # Site = "single" -> {ell_star}; "window3" -> {ell_star-1, ell_star, ell_star+1}
    # We steer decoder blocks 0..n_layers-1.  ell_star is the ABSOLUTE layer index
    # (0..32); block indices 0..31.  Convention: layer index k in hidden_states
    # corresponds to residual output *after* block k-1.  So to intervene on the
    # residual at "layer k" we hook block k-1.  Layer 0 (embedding) is not a
    # decoder block; if ell_star == 0, promote to 1.
    layers_to_hook = []
    if site == "single":
        layer_hs = ell_star if ell_star > 0 else 1
        layers_to_hook = [layer_hs - 1]
    elif site == "window3":
        for k in (-1, 0, 1):
            hs = ell_star + k
            if hs <= 0 or hs > 32:
                continue
            layers_to_hook.append(hs - 1)
    layers_to_hook = sorted(set(layers_to_hook))

    handles = []
    if alpha_mult != 0:
        handles = register_steering(model, layers_to_hook, unit_dir_steer, alpha)
    try:
        t0 = time.time()
        # Baseline transfers on all held-out baseline rows (for C3 mean shift +
        # for building M[V, W] we also need the *partner* rows so we can measure
        # each W's effect under the intervention).
        combined = held_base_rows + held_partner_rows
        gens = decode_batch(combined, tokenizer, model, batch_size=batch_size,
                            max_new_tokens=max_new_tokens)
        n_base = len(held_base_rows)
        base_gens = gens[:n_base]
        partner_gens = gens[n_base:]
        base_taus = [parse_transfer(g) for g in base_gens]
        partner_taus = [parse_transfer(g) for g in partner_gens]
        n_parse_fail_base = sum(1 for t in base_taus if t is None)

        # C3 headline: mean transfer on baseline.
        valid_taus = [t for t in base_taus if t is not None]
        mean_transfer = float(np.mean(valid_taus)) if valid_taus else float("nan")
        median_transfer = float(np.median(valid_taus)) if valid_taus else float("nan")
        std_transfer = float(np.std(valid_taus)) if valid_taus else float("nan")

        # Per-V effect on baseline: mean_{V=pos} tau - mean_{V=neg} tau (baseline).
        # This is the "shift" C3 targets.
        pos_val = pos_side_map[V]
        pos = [t for r, t in zip(held_base_rows, base_taus)
               if t is not None and r[V] == pos_val]
        neg = [t for r, t in zip(held_base_rows, base_taus)
               if t is not None and r[V] != pos_val]
        v_effect = (float(np.mean(pos)) - float(np.mean(neg))
                    if pos and neg else 0.0)

        # For selectivity matrix (M5) we also need each W's effect on baseline rows.
        # Compute all four V,W effects on THIS (V, alpha) run's baseline transfers.
        w_effects = {}
        for W in ["G", "A", "I", "M"]:
            pos_val_w = pos_side_map[W]
            pos_w = [t for r, t in zip(held_base_rows, base_taus)
                     if t is not None and r[W] == pos_val_w]
            neg_w = [t for r, t in zip(held_base_rows, base_taus)
                     if t is not None and r[W] != pos_val_w]
            w_effects[W] = (float(np.mean(pos_w)) - float(np.mean(neg_w))
                            if pos_w and neg_w else 0.0)

        # Coherence: sample first K (or fewer) generations & re-check.
        coh = coherence_check(base_gens[:coherence_k])

        # Save
        result = dict(
            V=V, decorrelator=decorr, site=site,
            alpha_mult=alpha_mult, alpha=alpha, sigma_proj=sigma_proj,
            ell_star=ell_star, layers_hooked=layers_to_hook,
            n_baseline=n_base,
            n_baseline_parsed=len(valid_taus),
            parse_failure_rate=(n_parse_fail_base / n_base if n_base else 0.0),
            mean_transfer=mean_transfer,
            median_transfer=median_transfer,
            std_transfer=std_transfer,
            v_effect=v_effect,
            w_effects=w_effects,
            coherence=coh,
            sample_generations=base_gens[:3],
            wall_time_s=time.time() - t0,
        )
        with open(fname, "w") as f:
            json.dump(result, f, indent=2)
        return result
    finally:
        for h in handles:
            h.remove()


def permutation_test_offdiag_vs_diag(M, n_perm=5000, seed=42):
    """Test: are off-diagonal magnitudes significantly smaller than diagonal?
    Returns p-value under null 'diag and offdiag drawn from same distribution'.
    """
    diag = np.abs(np.array([M[V][V] for V in ["G", "A", "I", "M"]]))
    off = []
    for V in ["G", "A", "I", "M"]:
        for W in ["G", "A", "I", "M"]:
            if V != W:
                off.append(abs(M[V][W]))
    off = np.array(off)
    obs = diag.mean() - off.mean()
    combined = np.concatenate([diag, off])
    n_diag = len(diag)
    rng = np.random.default_rng(seed)
    ge = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        stat = combined[:n_diag].mean() - combined[n_diag:].mean()
        if stat >= obs:
            ge += 1
    p = (ge + 1) / (n_perm + 1)
    return float(obs), float(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--pure_dir", required=True,
                    help="artifacts/m3/directions_pure.pt")
    ap.add_argument("--raw_dir", required=True,
                    help="artifacts/m2/directions_raw.pt")
    ap.add_argument("--layer_pick", required=True)
    ap.add_argument("--held_acts", required=True,
                    help="artifacts/m2/heldout_activations.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out_m4", required=True)
    ap.add_argument("--out_m5", required=True)
    ap.add_argument("--variables", default="G,A,I,M")
    ap.add_argument("--alpha_grid", default="-4,-2,-1,0,1,2,4")
    ap.add_argument("--decorrs", default="gs,leace")
    ap.add_argument("--sites", default="single,window3")
    ap.add_argument("--m5_alpha_grid", default="-2,0,2")
    ap.add_argument("--m5_decorr", default="leace")
    ap.add_argument("--m5_site", default="single")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_new_tokens", type=int, default=8)
    ap.add_argument("--coherence_k", type=int, default=10)
    ap.add_argument("--n_sample", type=int, default=-1,
                    help="If >0, use only the first n_sample held-out baseline "
                         "trials (smoke-test mode).")
    args = ap.parse_args()

    Path(args.out_m4).mkdir(parents=True, exist_ok=True)
    Path(args.out_m5).mkdir(parents=True, exist_ok=True)

    # ------ Load data & artifacts -------------------------------------------- #
    with open(args.data) as f:
        rows = [json.loads(l) for l in f]
    held_rows = [r for r in rows if r["split"] == "held"]
    held_base_rows = [r for r in held_rows if r["is_paired_partner_of"] is None]
    held_partner_rows = [r for r in held_rows if r["is_paired_partner_of"] is not None]
    if args.n_sample > 0:
        held_base_rows = held_base_rows[: args.n_sample]
        keep_ids = {r["trial_id"] for r in held_base_rows}
        held_partner_rows = [r for r in held_partner_rows
                             if r["is_paired_partner_of"] in keep_ids]
        print(f"[m4] SANITY: reduced held-out to {len(held_base_rows)} baseline "
              f"+ {len(held_partner_rows)} partners")

    with open(args.layer_pick) as f:
        pick = json.load(f)
    pure_pack = torch.load(args.pure_dir, map_location="cpu", weights_only=False)
    raw_pack = torch.load(args.raw_dir, map_location="cpu", weights_only=False)
    held_pack = torch.load(args.held_acts, map_location="cpu", weights_only=False)
    layer_ids = raw_pack["layer_ids"]
    layer_to_idx = {li: i for i, li in enumerate(layer_ids)}

    # ------ Load model ------------------------------------------------------- #
    tokenizer, model = load_model_and_tokenizer(args.model)

    pos_side_map = {"G": "male", "A": "young", "I": "take-frame", "M": "no-meet"}

    variables = args.variables.split(",")
    alpha_grid = [int(x) for x in args.alpha_grid.split(",")]
    decorrs = args.decorrs.split(",")
    sites = args.sites.split(",")

    # ------ M4 grid --------------------------------------------------------- #
    summary = []
    for V in variables:
        if V not in pick or V not in pure_pack["pure"]:
            print(f"[m4] skip V={V}: no picked layer / no direction")
            continue
        ell_star = pick[V]["ell_star"]
        layer_local = layer_to_idx[ell_star]
        for decorr in decorrs:
            v_pure = pure_pack["pure"][V].get(decorr)
            if v_pure is None or float(v_pure.norm()) < 1e-4:
                print(f"[m4] skip V={V} decorr={decorr}: null / degenerate direction")
                continue
            unit_dir = (v_pure / (v_pure.norm() + 1e-9)).float()
            sigma_proj = compute_sigma_proj(held_pack["acts"], layer_local, unit_dir)
            print(f"[m4] V={V} decorr={decorr} ell*={ell_star} "
                  f"|v|={float(v_pure.norm()):.3f} sigma_proj={sigma_proj:.4f}")
            for site in sites:
                for alpha_mult in alpha_grid:
                    print(f"  [m4] alpha_mult={alpha_mult:+d} site={site}", flush=True)
                    res = run_one_grid_point(
                        V=V, decorr=decorr, alpha_mult=alpha_mult, site=site,
                        sigma_proj=sigma_proj, ell_star=ell_star,
                        unit_dir_extract=unit_dir,
                        unit_dir_steer=unit_dir,
                        model=model, tokenizer=tokenizer,
                        held_base_rows=held_base_rows,
                        held_partner_rows=held_partner_rows,
                        batch_size=args.batch_size,
                        max_new_tokens=args.max_new_tokens,
                        coherence_k=args.coherence_k,
                        pos_side_map=pos_side_map,
                        out_dir=args.out_m4,
                    )
                    summary.append(res)

    with open(os.path.join(args.out_m4, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ------ M5 selectivity matrix ------------------------------------------- #
    # For each V in {G,A,I,M}, at alpha in {-2,0,+2}, LEACE, single site:
    # M[V, W] = w_effects[W] under (V, alpha) - w_effects[W] under (V, alpha=0).
    def extract_run(V, decorr, site, alpha_mult):
        for r in summary:
            if (r["V"] == V and r["decorrelator"] == decorr
                    and r["site"] == site and r["alpha_mult"] == alpha_mult):
                return r
        return None
    m5_matrix = {}
    m5_missing = []
    for V in variables:
        m5_matrix[V] = {W: 0.0 for W in ["G", "A", "I", "M"]}
        r_zero = extract_run(V, args.m5_decorr, args.m5_site, 0)
        for alpha_mult in [int(x) for x in args.m5_alpha_grid.split(",")]:
            if alpha_mult == 0:
                continue
            r = extract_run(V, args.m5_decorr, args.m5_site, alpha_mult)
            if r is None or r_zero is None:
                m5_missing.append(dict(V=V, alpha_mult=alpha_mult))
                continue
            for W in ["G", "A", "I", "M"]:
                shift = r["w_effects"][W] - r_zero["w_effects"][W]
                # Store the +2 sigma shift as the canonical selectivity entry
                # (per the plan's C4 headline).
                if alpha_mult == 2:
                    m5_matrix[V][W] = shift
    diag = [abs(m5_matrix[V][V]) for V in ["G", "A", "I", "M"]]
    off = [abs(m5_matrix[V][W]) for V in ["G", "A", "I", "M"]
           for W in ["G", "A", "I", "M"] if V != W]
    max_off = max(off) if off else 0.0
    min_diag = min(diag) if diag else 0.0
    c4_ratio = max_off / max(min_diag, 1e-9)
    obs, p_perm = permutation_test_offdiag_vs_diag(m5_matrix)

    with open(os.path.join(args.out_m5, "selectivity_matrix.json"), "w") as f:
        json.dump(dict(
            matrix=m5_matrix,
            alpha_used=2,
            decorr=args.m5_decorr,
            site=args.m5_site,
            diagonal=dict(zip(["G", "A", "I", "M"], diag)),
            max_offdiag=max_off,
            min_diag=min_diag,
            c4_ratio_max_off_over_min_diag=c4_ratio,
            n_missing_grid_points=len(m5_missing),
            missing=m5_missing,
        ), f, indent=2)
    with open(os.path.join(args.out_m5, "permutation_test.json"), "w") as f:
        json.dump(dict(observed_mean_diff=obs, p_value=p_perm, n_perm=5000), f, indent=2)

    print("[m4+m5] done")


if __name__ == "__main__":
    main()
