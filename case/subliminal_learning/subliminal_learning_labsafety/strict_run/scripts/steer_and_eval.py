"""M2.2b — Steering on base student (7-point α × 3-seed dose-response).

For each (α, seed): install add-hook on top-K language-tower layers of the base
student, injecting α · v where v is the L2-normalized d_diff direction (unit
vector). α is expressed in units of σ_proj: the per-layer std of h·û on the
base cache. To keep α comparable across layers, we jointly normalize the
top-K vectors so that the aggregate σ_proj (RMS across layers) is 1.

Direction sign convention: v_l points from Ctrl-B → treated. A POSITIVE α
should push the base student's behavior TOWARD the treated arm (i.e., lower
QA_I accuracy). This is what the plan's ρ ≤ −0.5 threshold expects: as α
increases, accuracy drops (negative Spearman).
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from pathlib import Path

from common import (
    PROJECT_ROOT,
    load_processor, load_student_multimodal,
    JudgeCache, call_judge, set_seed,
)
from eval_qa_i import JUDGE_PROMPT_TMPL, parse_judge_verdict, load_qa_i_items, render_multimodal_prompt
from ablate_and_eval import make_add_hook, install_hooks, eval_with_hooks


def compute_sigma_proj(cache_path, l_hs, u_l):
    """σ_proj = std over base-cache activations of h_l · û_l."""
    d = np.load(cache_path, allow_pickle=True)
    h = d["hidden_states"].astype(np.float32)   # [N, L+1, d]
    proj = h[:, l_hs, :] @ u_l                 # [N]
    return float(np.std(proj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="Direction seed (which per-seed u_diff to use).")
    ap.add_argument("--l_core", required=True)
    ap.add_argument("--cache_root", default=str(PROJECT_ROOT / "cache/residuals"),
                    help="For σ_proj calibration.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge_cache",
                    default=str(PROJECT_ROOT / "cache" / "qa_i_judge.jsonl"))
    ap.add_argument("--random_direction", action="store_true",
                    help="Replace u_diff with a matched-random unit vector "
                         "(used for M2.2c matched-control specificity).")
    args = ap.parse_args()

    set_seed(args.seed)

    print(f"[steer alpha={args.alpha} seed={args.seed} random={args.random_direction}] "
          f"loading BASE student", flush=True)
    model = load_student_multimodal()   # NO adapter — base
    model.eval()
    processor = load_processor()

    l_core = json.load(open(args.l_core))
    top_layers = l_core["top_k_layers"]
    seed_dir = l_core["directions_per_seed"][str(args.seed)]
    u_diff_all = np.array(seed_dir["u_diff"])   # [L+1, d]
    v_diff_all = np.array(seed_dir["v_diff"])

    # Base-cache path for σ_proj estimation. If not available (base was not
    # cached), estimate σ from treated cache — this is a mild approximation.
    base_cache = Path(args.cache_root) / "Ctrl-B" / f"seed{args.seed}" / "activations.npz"
    if not base_cache.exists():
        base_cache = Path(args.cache_root) / "treated" / f"seed{args.seed}" / "activations.npz"
    print(f"[steer] σ_proj source = {base_cache}", flush=True)

    layer_hook_indices = []
    u_by_layer_torch = {}
    sigmas = []
    for l_hs in top_layers:
        if l_hs == 0:
            continue
        li = l_hs - 1
        vec = u_diff_all[l_hs]
        if args.random_direction:
            rng = np.random.default_rng(args.seed + li * 7)
            r = rng.standard_normal(size=vec.shape).astype(np.float32)
            r /= np.linalg.norm(r) + 1e-8
            vec = r
        # σ_proj on base cache for this layer.
        sigma = compute_sigma_proj(str(base_cache), l_hs, vec.astype(np.float32))
        sigmas.append(sigma)
        u_by_layer_torch[li] = torch.tensor(vec, dtype=torch.float32)
        layer_hook_indices.append(li)

    # Aggregate σ across top-K layers (RMS).
    agg_sigma = float(np.sqrt(np.mean(np.square(sigmas)))) if sigmas else 1.0
    # α_actual = alpha (in σ_proj units) * agg_sigma
    alpha_actual = args.alpha * agg_sigma
    print(f"[steer] per-layer σ_proj = {sigmas}, agg_sigma = {agg_sigma:.4f}, "
          f"alpha_actual = {alpha_actual:.4f}", flush=True)

    # Reuse install_hooks with mode="steer" and per-layer alpha_actual.
    handles = []
    for li in layer_hook_indices:
        u = u_by_layer_torch[li]
        hook = make_add_hook(u, alpha_actual)
        # find layer container
        # navigate to model.language_model.layers[li]
        lang = None
        for name, mod in model.named_modules():
            if name.endswith(".language_model") and name.count(".") == 1:
                lang = mod
                break
        if lang is None:
            base = getattr(model, "base_model", model)
            base = getattr(base, "model", base)
            lang = getattr(base, "language_model", None)
        h = lang.layers[li].register_forward_hook(hook)
        handles.append(h)

    items = load_qa_i_items()
    judge_cache = JudgeCache(args.judge_cache)
    try:
        recs, acc, n_c = eval_with_hooks(model, processor, items, judge_cache)
    finally:
        for h in handles:
            h.remove()

    print(f"[steer alpha={args.alpha} seed={args.seed}] acc={acc:.4f} "
          f"(n_correct={n_c}/{len(items)})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "alpha": args.alpha,
            "alpha_actual": alpha_actual,
            "agg_sigma_proj": agg_sigma,
            "seed": args.seed,
            "random_direction": args.random_direction,
            "top_layers_hidden_states_index": top_layers,
            "layer_hook_indices": layer_hook_indices,
            "n": len(items),
            "acc_steered": acc,
            "n_correct": n_c,
            "per_item": recs,
        }, f, indent=2)
    print(f"[steer alpha={args.alpha} seed={args.seed}] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
