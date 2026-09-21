#!/usr/bin/env python3
"""
Launcher: run scripts/m5_steer_v2.py for multiple seeds and direction methods
BUT reuse the loaded Gemma-3-27B across seeds (avoids paying the 5-minute
load cost twice).  Writes one output JSON per (seed, direction_method).

Grid:  seeds × direction_methods
Layout matches the plain M5 outputs so the aggregator can pick it up:
    results/m5_v2/m5v2_{direction_method}_seed{seed}.json
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# We inline the run by importing the m5_steer_v2 module's helpers and reusing
# a single loaded model across seeds.
import importlib.util
spec = importlib.util.spec_from_file_location("m5v2_mod",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "m5_steer_v2.py"))
m5v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m5v2)

from vc_common import MODEL_PATH, load_model_and_tokenizer, set_all_seeds, parse_confidence, MAX_CONF_TOKENS
import h5py


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,123")
    p.add_argument("--methods", default="diff_of_means")
    p.add_argument("--n_items", type=int, default=60)
    p.add_argument("--alphas", default="-16,-8,-4,-1,0,1,4,8,16")
    p.add_argument("--n_random", type=int, default=15)
    p.add_argument("--capability_tol_nats", type=float, default=0.3)
    p.add_argument("--n_sample_texts", type=int, default=5)
    p.add_argument("--m1_cache", default="results/m1")
    p.add_argument("--m2_dir",   default="results/m2")
    p.add_argument("--template", default="T0")
    p.add_argument("--out_dir",  default="results/m5_v2")
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    methods = args.methods.split(",")

    # Load model once
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[runner] Loading model (once) ...", flush=True)
    t_load0 = time.time()
    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype=dtype)
    device = next(model.parameters()).device
    print(f"[runner] Model on {device} in {(time.time()-t_load0)/60:.1f} m", flush=True)

    for seed in seeds:
        for method in methods:
            out_path = Path(args.out_dir) / f"m5v2_{method}_seed{seed}.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"\n\n=========================== seed={seed} method={method} ===========================", flush=True)
            # Build an argparse.Namespace matching m5_steer_v2's expectations
            ns = argparse.Namespace(
                model=MODEL_PATH,
                dataset="",  # unused inside the module
                n_items=args.n_items,
                site="AUTO",
                direction_method=method,
                direction_train_split=0.5,
                alphas=args.alphas,
                n_random=args.n_random,
                capability_tol_nats=args.capability_tol_nats,
                n_sample_texts=args.n_sample_texts,
                sample_gen_tokens=20,
                seed=seed,
                m1_cache=args.m1_cache,
                m2_dir=args.m2_dir,
                template=args.template,
                out=str(out_path),
                dtype=args.dtype,
            )
            run_one(model, tok, device, ns)


def run_one(model, tok, device, args):
    """Copy of m5_steer_v2.main() but using the pre-loaded (model, tok)."""
    set_all_seeds(args.seed)

    alphas = [float(x) for x in args.alphas.split(",")]

    with (Path(args.m2_dir) / "top_k_sites.json").open() as f:
        top_k = json.load(f)["top_k"]
    if args.site == "AUTO":
        site_pos, site_L = top_k[0]["position"], top_k[0]["layer"]
    else:
        site_pos, site_L = m5v2.parse_site(args.site)
    print(f"[m5v2/seed{args.seed}] site={site_pos}L{site_L} method={args.direction_method} alphas={alphas} n_random={args.n_random}", flush=True)

    # digit vocab
    digit_token_ids = set()
    for d in "0123456789":
        for tid in tok(d, add_special_tokens=False).input_ids:
            digit_token_ids.add(tid)

    unrelated_ids = torch.tensor(
        tok(m5v2.UNRELATED_CONTINUATION, add_special_tokens=False).input_ids,
        dtype=torch.long, device=device,
    )

    items, acts_path = m5v2.load_m1_seed(args.m1_cache, args.seed, args.template)
    items = [it for it in items if it.get("verbal_conf") is not None]
    if args.n_items > 0:
        items = items[: max(args.n_items * 3, 500)]
    with h5py.File(acts_path, "r") as h5:
        X_site = h5[site_pos][str(site_L)][()]

    idxs = [it["idx"] for it in items]
    X_site = X_site[idxs]
    y = np.array([it["verbal_conf"] for it in items], dtype=np.float32)

    n = len(items)
    rng = np.random.RandomState(args.seed + 17)
    perm = rng.permutation(n)
    n_train = int(round(n * args.direction_train_split))
    train_idx = perm[:n_train]
    eval_idx  = perm[n_train:][: args.n_items]

    X_train = X_site[train_idx]
    y_train = y[train_idx]
    high_mask = y_train >= 70
    low_mask  = y_train <= 30
    if high_mask.sum() < 20 or low_mask.sum() < 20:
        thr_hi = float(np.quantile(y_train, 0.66))
        thr_lo = float(np.quantile(y_train, 0.34))
        high_mask = y_train >= thr_hi
        low_mask = y_train <= thr_lo
        print(f"[m5v2/seed{args.seed}] adaptive thresholds: high≥{thr_hi} low≤{thr_lo}", flush=True)
    print(f"[m5v2/seed{args.seed}] direction from n_high={int(high_mask.sum())} n_low={int(low_mask.sum())}", flush=True)

    d_trained = m5v2.extract_direction(X_train[high_mask], X_train[low_mask], args.direction_method)
    d_trained_norm = float(np.linalg.norm(d_trained))
    if d_trained_norm < 1e-6:
        raise RuntimeError("Extracted direction has near-zero norm.")
    u_trained = d_trained / d_trained_norm

    projections_trained = X_train @ u_trained
    sigma_proj = float(np.std(projections_trained))
    print(f"[m5v2/seed{args.seed}] σ_proj(trained,train)={sigma_proj:.4f}  ||d||={d_trained_norm:.4f}", flush=True)

    rng2 = np.random.RandomState(args.seed + 71)
    dim = int(u_trained.shape[0])
    random_dirs = rng2.randn(args.n_random, dim).astype(np.float32)
    random_dirs = random_dirs / (np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-9)
    projs_rand = X_train @ random_dirs.T
    sigma_proj_random = projs_rand.std(axis=0)
    print(f"[m5v2/seed{args.seed}] σ_proj(random) mean={sigma_proj_random.mean():.4f} min={sigma_proj_random.min():.4f} max={sigma_proj_random.max():.4f}", flush=True)

    hook = m5v2.SteeringHook(model, site_L, torch.tensor(u_trained, device=device))
    hook.install()

    eval_items = [items[i] for i in eval_idx]
    n_eval = len(eval_items)

    per_item_ctx = []
    for it in eval_items:
        input_ids = m5v2.build_per_item_context(it, tok, device)
        if site_pos.startswith("E"):
            patch_pos = it["post_answer_positions"].get(site_pos)
        else:
            patch_pos = input_ids.shape[1] - 1
        if patch_pos is None or patch_pos >= input_ids.shape[1]:
            continue
        per_item_ctx.append({
            "input_ids": input_ids,
            "patch_pos": int(patch_pos),
        })
    print(f"[m5v2/seed{args.seed}] usable eval items: {len(per_item_ctx)} / {n_eval}", flush=True)

    from typing import Dict, List
    trained_per_alpha: Dict[float, Dict] = {}
    random_per_alpha:  Dict[float, List[Dict]] = {a: [] for a in alphas}
    records: List[Dict] = []
    t0 = time.time()

    def eval_direction(dir_id: int, u: np.ndarray, sigma: float, is_trained: bool):
        hook.set_direction(torch.tensor(u, device=device))
        per_alpha_this_dir: Dict[float, Dict] = {}
        for alpha in alphas:
            alpha_scaled = alpha * sigma
            confs = []
            off_digit_count = 0
            cont_nlls = []
            sample_texts = []
            for i, ctx in enumerate(per_item_ctx):
                input_ids = ctx["input_ids"]
                patch_pos = ctx["patch_pos"]
                hook.set(patch_pos, alpha_scaled)
                conf_val, gen_ids, off_digit = m5v2.greedy_confidence(model, tok, input_ids, digit_token_ids)
                if off_digit:
                    off_digit_count += 1
                if conf_val is not None:
                    confs.append(conf_val)
                if i % 3 == 0:
                    cont_nll = m5v2.teacher_forced_nll(model, input_ids, unrelated_ids)
                    cont_nlls.append(cont_nll)
                    _rec_cont_nll = cont_nll
                else:
                    _rec_cont_nll = None
                if is_trained and i < args.n_sample_texts:
                    txt = m5v2.greedy_text(model, tok, input_ids, max_tokens=args.sample_gen_tokens)
                    sample_texts.append({"alpha": alpha, "text": txt})
                if is_trained:
                    records.append({
                        "dir": "trained",
                        "alpha": alpha,
                        "idx": eval_items[i]["idx"],
                        "conf": conf_val,
                        "off_digit_first": off_digit,
                        "cont_nll": _rec_cont_nll,
                    })
                if is_trained and (i + 1) % 20 == 0:
                    print(f"[m5v2/seed{args.seed}] trained α={alpha:+.1f} item {i+1}/{len(per_item_ctx)} conf={np.mean(confs):.1f} nll={np.mean(cont_nlls) if cont_nlls else 0:.3f} elapsed={(time.time()-t0)/60:.1f}m", flush=True)
            per_alpha_this_dir[alpha] = {
                "conf_mean": float(np.mean(confs)) if confs else 0.0,
                "conf_std":  float(np.std(confs)) if confs else 0.0,
                "n":         len(confs),
                "off_digit_rate": off_digit_count / max(1, len(per_item_ctx)),
                "cont_nll_mean":  float(np.mean(cont_nlls)) if cont_nlls else 0.0,
                "cont_nll_n":     len(cont_nlls),
                "sample_texts":   sample_texts if is_trained else [],
            }
        return per_alpha_this_dir

    print(f"[m5v2/seed{args.seed}] === trained direction ===", flush=True)
    trained_per_alpha = eval_direction(0, u_trained, sigma_proj, is_trained=True)
    for r_idx in range(args.n_random):
        u_r = random_dirs[r_idx]
        s_r = float(sigma_proj_random[r_idx])
        print(f"[m5v2/seed{args.seed}] === random {r_idx+1}/{args.n_random} σ={s_r:.4f} elapsed={(time.time()-t0)/60:.1f}m ===", flush=True)
        per_alpha_r = eval_direction(r_idx + 1, u_r, s_r, is_trained=False)
        for a in alphas:
            random_per_alpha[a].append(per_alpha_r[a])

    hook.remove()

    random_per_alpha_agg = {}
    random_per_alpha_per_dir = {}
    for a in alphas:
        conf_means = np.array([r["conf_mean"] for r in random_per_alpha[a]], dtype=np.float64)
        cont_nll_means = np.array([r["cont_nll_mean"] for r in random_per_alpha[a]], dtype=np.float64)
        conf_t = trained_per_alpha[a]["conf_mean"]
        nll_t  = trained_per_alpha[a]["cont_nll_mean"]
        pct_conf = float((conf_means < conf_t).mean()) if len(conf_means) else 0.5
        pct_nll  = float((cont_nll_means < nll_t).mean()) if len(cont_nll_means) else 0.5
        random_per_alpha_agg[str(a)] = {
            "conf_mean_trained": conf_t,
            "conf_mean_random_mean": float(conf_means.mean()) if len(conf_means) else 0.0,
            "conf_mean_random_std":  float(conf_means.std()) if len(conf_means) else 0.0,
            "conf_mean_random_percentile_of_trained": pct_conf,
            "cont_nll_trained": nll_t,
            "cont_nll_random_mean": float(cont_nll_means.mean()) if len(cont_nll_means) else 0.0,
            "cont_nll_random_std":  float(cont_nll_means.std()) if len(cont_nll_means) else 0.0,
            "cont_nll_random_percentile_of_trained": pct_nll,
        }
        random_per_alpha_per_dir[str(a)] = conf_means.tolist()

    baseline_nll = trained_per_alpha.get(0.0, {}).get("cont_nll_mean", 0.0)
    tol = args.capability_tol_nats
    alpha_star = 0.0
    for a in sorted(alphas, key=lambda x: abs(x)):
        if a == 0.0:
            continue
        d_nll = trained_per_alpha[a]["cont_nll_mean"] - baseline_nll
        if d_nll <= tol:
            if abs(a) > abs(alpha_star):
                alpha_star = a
    a_pos = abs(alpha_star)
    a_neg = -abs(alpha_star)
    conf_pos = trained_per_alpha.get(a_pos, {}).get("conf_mean", 0.0)
    conf_neg = trained_per_alpha.get(a_neg, {}).get("conf_mean", 0.0)
    conf_effect_at_alpha_star = float((conf_pos or 0.0) - (conf_neg or 0.0))

    at_star_key = str(a_pos)
    at_star = random_per_alpha_agg.get(at_star_key, {})
    conf_pct_at_star = at_star.get("conf_mean_random_percentile_of_trained", 0.5)
    trained_beats_random = (conf_pct_at_star <= 0.05) or (conf_pct_at_star >= 0.95)
    capability_preserved = (trained_per_alpha[a_pos]["cont_nll_mean"] - baseline_nll) <= tol if a_pos in trained_per_alpha else True

    summary = {
        "site":              f"{site_pos}L{site_L}",
        "direction_method":  args.direction_method,
        "seed":              args.seed,
        "n_eval":            len(per_item_ctx),
        "n_random":          args.n_random,
        "sigma_proj_train":  sigma_proj,
        "direction_norm":    d_trained_norm,
        "capability_tol":    tol,
        "alphas":            alphas,
        "unrelated_continuation": m5v2.UNRELATED_CONTINUATION,
        "trained_per_alpha": {str(a): trained_per_alpha[a] for a in alphas},
        "random_per_alpha_agg":     random_per_alpha_agg,
        "random_per_alpha_per_dir": random_per_alpha_per_dir,
        "locked_alpha": {
            "capability_tol_nats_per_token": tol,
            "alpha_star":            a_pos,
            "cont_nll_at_alpha_star": trained_per_alpha.get(a_pos, {}).get("cont_nll_mean"),
            "cont_nll_baseline":     baseline_nll,
            "conf_effect_at_alpha_star": conf_effect_at_alpha_star,
        },
        "verdict_v2": {
            "trained_beats_random_at_alpha_star": bool(trained_beats_random),
            "trained_direction_conf_percentile_at_alpha_star": conf_pct_at_star,
            "capability_preserved_at_alpha_star": bool(capability_preserved),
        },
    }
    print(f"[m5v2/seed{args.seed}] SUMMARY: {json.dumps({k: summary[k] for k in ('site','direction_method','seed','locked_alpha','verdict_v2')}, indent=2)}", flush=True)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"[m5v2/seed{args.seed}] wrote {out_path}  total_elapsed={(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
