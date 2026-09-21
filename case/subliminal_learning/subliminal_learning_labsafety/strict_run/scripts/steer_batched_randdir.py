"""M2.2c_v2 — Batched matched-random-direction specificity control (n_random >= 30).

For a fixed (alpha, seed), load the base student ONCE, then sequentially install
30 different random-direction hooks (each a matched unit vector on the same
top-K layers), running the full QA_I eval loop after each hook install. The
result is 30 independent random-direction accuracy points per (alpha, seed),
which addresses mechanism-audit Q7 (n_random >= 30 for a stable null
distribution).

Cost model: 1 model load + 30 * (133-item QA_I eval + judge). Model load
dominates when cache is warm; 30 evals per (alpha, seed) is unavoidable but at
least the model isn't reloaded 30 times.

Output structure:

    {
      "alpha": 1.0, "seed": 42, "n_random": 30,
      "per_direction": [
        {"dir_id": 0, "acc_steered": 0.78, "other_rate": 0.09, ...},
        {"dir_id": 1, ...},
        ...
      ],
      "summary": {
        "mean_acc": ..., "std_acc": ..., "mean_delta_from_baseline": ...,
        "abs_delta_p95": ..., "n_random": 30
      }
    }

This addresses the audit's WARN on Q7 (underpowered specificity control) by
raising n_random from 1 to 30 at the specific alpha most informative for null
testing.
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
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--l_core", required=True)
    ap.add_argument("--cache_root", default=str(PROJECT_ROOT / "cache/residuals"))
    ap.add_argument("--n_random", type=int, default=30,
                    help="Number of independent matched-random directions.")
    ap.add_argument("--baseline_acc", type=float, default=None,
                    help="Baseline (α=0) accuracy for delta computation. If None, "
                         "reads from results/mech/M2_2b_alpha0_seed{seed}.json.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--judge_cache",
                    default=str(PROJECT_ROOT / "cache" / "qa_i_judge.jsonl"))
    args = ap.parse_args()

    set_seed(args.seed)

    print(f"[randdir-batch alpha={args.alpha} seed={args.seed} n_random={args.n_random}] "
          f"loading BASE student", flush=True)
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

    # Pre-compute σ_proj per layer using the REAL u_diff (so alpha_actual matches
    # the calibration used in M2.2b). This is what "matched-random" means: same
    # α scale, same layers, only the direction is randomized.
    layer_hook_indices = []
    sigmas = []
    real_dir_shapes = {}
    for l_hs in top_layers:
        if l_hs == 0:
            continue
        li = l_hs - 1
        vec = u_diff_all[l_hs].astype(np.float32)
        sigma = compute_sigma_proj(str(base_cache), l_hs, vec)
        sigmas.append(sigma)
        layer_hook_indices.append(li)
        real_dir_shapes[li] = vec.shape

    agg_sigma = float(np.sqrt(np.mean(np.square(sigmas)))) if sigmas else 1.0
    alpha_actual = args.alpha * agg_sigma
    print(f"[randdir-batch] agg_sigma={agg_sigma:.4f} alpha_actual={alpha_actual:.4f}", flush=True)

    # Resolve baseline accuracy for the delta computation.
    if args.baseline_acc is None:
        baseline_path = PROJECT_ROOT / f"results/mech/M2_2b_alpha0_seed{args.seed}.json"
        baseline = json.load(open(baseline_path))
        baseline_acc = baseline["acc_steered"]
        baseline_other_rate = baseline.get("capability_metric", {}).get("value")
    else:
        baseline_acc = args.baseline_acc
        baseline_other_rate = None

    print(f"[randdir-batch] baseline_acc (α=0) = {baseline_acc:.4f} "
          f"other_rate = {baseline_other_rate}", flush=True)

    # Find language model container ONCE.
    lang = None
    for name, mod in model.named_modules():
        if name.endswith(".language_model") and name.count(".") == 1:
            lang = mod
            break
    if lang is None:
        base = getattr(model, "base_model", model)
        base = getattr(base, "model", base)
        lang = getattr(base, "language_model", None)
    if lang is None:
        raise RuntimeError("Could not locate language_model container")

    items = load_qa_i_items()
    judge_cache = JudgeCache(args.judge_cache)

    per_direction = []
    accs = []
    for dir_id in range(args.n_random):
        # Independent RNG per direction — deterministic reproducibility.
        rng = np.random.default_rng(seed=args.seed * 10007 + dir_id * 31 + 1)

        # Build a random unit vector per layer.
        u_by_layer_torch = {}
        for li in layer_hook_indices:
            r = rng.standard_normal(size=real_dir_shapes[li]).astype(np.float32)
            r /= np.linalg.norm(r) + 1e-8
            u_by_layer_torch[li] = torch.tensor(r, dtype=torch.float32)

        # Install hooks.
        handles = []
        for li in layer_hook_indices:
            u = u_by_layer_torch[li]
            hook = make_add_hook(u, alpha_actual)
            h = lang.layers[li].register_forward_hook(hook)
            handles.append(h)

        try:
            recs, acc, n_c = eval_with_hooks(model, processor, items, judge_cache)
        finally:
            for h in handles:
                h.remove()

        n = len(items)
        n_o = sum(1 for r in recs if r["verdict"] == "OTHER")
        n_i = sum(1 for r in recs if r["verdict"] == "INCORRECT")
        other_rate = n_o / max(1, n)
        delta = acc - baseline_acc
        accs.append(acc)

        per_direction.append({
            "dir_id": dir_id,
            "n": n,
            "acc_steered": acc,
            "n_correct": n_c,
            "n_incorrect": n_i,
            "n_other": n_o,
            "other_rate": other_rate,
            "delta_from_baseline": delta,
            "abs_delta_from_baseline": abs(delta),
        })
        print(f"[randdir-batch] dir_id={dir_id:2d} acc={acc:.4f} "
              f"delta={delta:+.4f} other_rate={other_rate:.4f}", flush=True)

    accs_arr = np.array(accs)
    deltas = accs_arr - baseline_acc
    abs_deltas = np.abs(deltas)
    summary = {
        "n_random": args.n_random,
        "mean_acc": float(accs_arr.mean()),
        "std_acc": float(accs_arr.std(ddof=1)) if len(accs_arr) > 1 else 0.0,
        "mean_delta_from_baseline": float(deltas.mean()),
        "abs_mean_delta_from_baseline": float(abs_deltas.mean()),
        "abs_delta_p50": float(np.percentile(abs_deltas, 50)),
        "abs_delta_p95": float(np.percentile(abs_deltas, 95)),
        "abs_delta_max": float(abs_deltas.max()),
        "specificity_pass_criterion": "mean |ΔAcc| ≤ 1 pp",
        "specificity_pass": bool(abs_deltas.mean() <= 0.01),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "alpha": args.alpha,
            "alpha_actual": alpha_actual,
            "agg_sigma_proj": agg_sigma,
            "seed": args.seed,
            "n_random": args.n_random,
            "top_layers_hidden_states_index": top_layers,
            "layer_hook_indices": layer_hook_indices,
            "baseline_acc": baseline_acc,
            "baseline_other_rate": baseline_other_rate,
            "per_direction": per_direction,
            "summary": summary,
        }, f, indent=2)
    print(f"[randdir-batch] summary={summary}", flush=True)
    print(f"[randdir-batch] -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
