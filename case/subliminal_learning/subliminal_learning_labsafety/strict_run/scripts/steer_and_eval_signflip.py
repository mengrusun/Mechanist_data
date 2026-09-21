"""M2.2b_neg — Steering on base student with NEGATED direction v' = -v.

Same as `steer_and_eval.py` but uses `v' = -u_diff` at each top-K layer, so
positive alpha pushes AWAY from treated (toward Ctrl-B). If the mechanism-audit
sign inversion in the original M2.2b is a bookkeeping error (direction
convention flipped), running this negated sweep should yield ρ ≤ -0.5 in ≥ 2/3
seeds — restoring the expected monotone dose-response. If both signs give
ρ ≈ 0, the low-rank residual direction genuinely does not localize the
covert-channel drop, strengthening the BOUNDED NULL scientific conclusion.

Also logs `other_rate` at every (α, seed) alongside `acc_steered` so the
mechanism-audit Q4 capability metric is satisfied at generation time (not
retroactively).
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
from ablate_and_eval import make_add_hook, eval_with_hooks
from steer_and_eval import compute_sigma_proj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="Direction seed (which per-seed u_diff to use).")
    ap.add_argument("--l_core", required=True)
    ap.add_argument("--cache_root", default=str(PROJECT_ROOT / "cache/residuals"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge_cache",
                    default=str(PROJECT_ROOT / "cache" / "qa_i_judge.jsonl"))
    args = ap.parse_args()

    set_seed(args.seed)

    print(f"[steer-neg alpha={args.alpha} seed={args.seed}] loading BASE student", flush=True)
    model = load_student_multimodal()
    model.eval()
    processor = load_processor()

    l_core = json.load(open(args.l_core))
    top_layers = l_core["top_k_layers"]
    seed_dir = l_core["directions_per_seed"][str(args.seed)]
    u_diff_all = np.array(seed_dir["u_diff"])

    base_cache = Path(args.cache_root) / "Ctrl-B" / f"seed{args.seed}" / "activations.npz"
    if not base_cache.exists():
        base_cache = Path(args.cache_root) / "treated" / f"seed{args.seed}" / "activations.npz"

    layer_hook_indices = []
    u_by_layer_torch = {}
    sigmas = []
    for l_hs in top_layers:
        if l_hs == 0:
            continue
        li = l_hs - 1
        vec = u_diff_all[l_hs]
        # NEGATE the direction — the only difference vs steer_and_eval.py.
        vec_neg = (-vec).astype(np.float32)
        # σ_proj is unchanged under sign flip.
        sigma = compute_sigma_proj(str(base_cache), l_hs, vec.astype(np.float32))
        sigmas.append(sigma)
        u_by_layer_torch[li] = torch.tensor(vec_neg, dtype=torch.float32)
        layer_hook_indices.append(li)

    agg_sigma = float(np.sqrt(np.mean(np.square(sigmas)))) if sigmas else 1.0
    alpha_actual = args.alpha * agg_sigma
    print(f"[steer-neg] per-layer σ_proj = {sigmas}, agg_sigma = {agg_sigma:.4f}, "
          f"alpha_actual = {alpha_actual:.4f}", flush=True)

    handles = []
    for li in layer_hook_indices:
        u = u_by_layer_torch[li]
        hook = make_add_hook(u, alpha_actual)
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

    # Compute capability metric (other_rate) at generation time — this is the M2.2b_neg
    # counterpart to the retroactive extraction we did for M2.2b.
    n = len(items)
    n_i = sum(1 for r in recs if r["verdict"] == "INCORRECT")
    n_o = sum(1 for r in recs if r["verdict"] == "OTHER")
    other_rate = n_o / max(1, n)

    print(f"[steer-neg alpha={args.alpha} seed={args.seed}] acc={acc:.4f} "
          f"other_rate={other_rate:.4f} (n_correct={n_c}/{n})", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "alpha": args.alpha,
            "alpha_actual": alpha_actual,
            "agg_sigma_proj": agg_sigma,
            "seed": args.seed,
            "sign_convention": "negated (v' = -u_diff)",
            "random_direction": False,
            "top_layers_hidden_states_index": top_layers,
            "layer_hook_indices": layer_hook_indices,
            "n": n,
            "acc_steered": acc,
            "n_correct": n_c,
            "capability_metric": {
                "name": "other_rate",
                "value": other_rate,
                "n_incorrect": n_i,
                "n_other": n_o,
                "source": "logged at generation time",
            },
            "per_item": recs,
        }, f, indent=2)
    print(f"[steer-neg alpha={args.alpha} seed={args.seed}] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
