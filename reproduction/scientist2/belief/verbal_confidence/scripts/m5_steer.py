#!/usr/bin/env python3
"""
M5 — Direction steering (Representation and Parameter Analysis / Steering Vectors).

At the top-1 cache site (position, layer) identified by M2:
  1. Extract a "confidence direction" u_L on a train split of M1 items:
       - diff_of_means:  u = mean(h_high) − mean(h_low)  where high = verbal_conf ≥ 70,
                          low = verbal_conf ≤ 30
       - lda:            Fisher LDA direction between high vs low
     Normalize to unit vector u_hat.
  2. Compute per-item projection std σ_L = std(h_L^T u_hat) on the train split.
  3. For each α ∈ {-4, -2, -1, 0, 1, 2, 4}:
       apply steering h_L[pos] ← h_L[pos] + α · σ_L · u_hat
       greedy-decode confidence at C0
  4. Report:
       - verb_conf_mean(α)                    — target metric
       - monotone_r_squared                   — R² of a linear fit through (α, mean verb_conf)
       - answer_acc_preserved                 — general-ability metric (trivially preserved
                                                since answer is baked in prompt; used as
                                                fluency-collapse detector via off-vocab argmax)
       - collapse_flag: True if the argmax at C0 falls OUTSIDE the digit vocabulary
                        at any |α| ≥ 2 (marker for off-distribution collapse per
                        `experiment-tips/steering-coefficient-tuning`).

Grid = α × direction_method × seed, per plan.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vc_common import (
    CONF_GEN_LABEL,
    MAX_CONF_TOKENS,
    MODEL_PATH,
    load_model_and_tokenizer,
    parse_confidence,
    set_all_seeds,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--dataset", default="/data/zhenqian/data/trivia_qa")
    p.add_argument("--n_items", type=int, default=300)
    p.add_argument("--site", default="AUTO",
                   help="'AUTO' → use M2's #1 top site; or explicit 'E0L30'.")
    p.add_argument("--direction_method", default="diff_of_means",
                   choices=["diff_of_means", "lda"])
    p.add_argument("--direction_train_split", type=float, default=0.5)
    p.add_argument("--direction_eval_split", type=float, default=0.5)
    p.add_argument("--alphas", default="-4,-2,-1,0,1,2,4")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--m1_cache", required=True)
    p.add_argument("--m2_dir", required=True)
    p.add_argument("--template", default="T0")
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    return p.parse_args()


def load_m1_seed(m1_cache: str, seed: int, template: str):
    seed_dir = Path(m1_cache) / f"seed{seed}" / template
    items = []
    with (seed_dir / "items.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items, str(seed_dir / "activations.h5")


def parse_site(s: str) -> Tuple[str, int]:
    m = re.match(r"^([EC]\d+)L(\d+)$", s)
    if not m:
        raise ValueError(f"Bad site spec: {s!r}")
    return m.group(1), int(m.group(2))


def extract_direction(X_high: np.ndarray, X_low: np.ndarray, method: str) -> np.ndarray:
    """
    diff_of_means: mean(high) - mean(low)
    lda: Fisher LDA direction = Σ⁻¹(μ₁ − μ₀) with pooled within-class covariance (regularized)
    """
    if method == "diff_of_means":
        d = X_high.mean(axis=0) - X_low.mean(axis=0)
    elif method == "lda":
        mu1 = X_high.mean(axis=0)
        mu0 = X_low.mean(axis=0)
        c1 = X_high - mu1
        c0 = X_low - mu0
        n1, n0 = len(X_high), len(X_low)
        Sw = (c1.T @ c1 + c0.T @ c0) / max(1, n1 + n0 - 2)
        # Regularize
        dim = Sw.shape[0]
        Sw = Sw + 1e-3 * np.eye(dim, dtype=Sw.dtype) * np.trace(Sw) / dim
        d = np.linalg.solve(Sw, mu1 - mu0)
    else:
        raise ValueError(method)
    return d.astype(np.float32)


@torch.no_grad()
def steered_decode_conf(model, tok, input_ids: torch.Tensor, layer: int, patch_pos: int,
                        direction_unit: torch.Tensor, alpha_scaled: float,
                        digit_token_ids: set,
                        max_conf_tokens: int = MAX_CONF_TOKENS
                        ) -> Tuple[Optional[int], List[int], bool]:
    """
    Add alpha_scaled · direction_unit to hidden_states[L][:, patch_pos, :] via a forward hook,
    then greedy-decode confidence. Returns (parsed_int, gen_ids, off_digit_at_first_step).
    """
    device = input_ids.device
    eos_id = tok.eos_token_id
    newline_ids = {tok(s, add_special_tokens=False).input_ids[-1] for s in ("\n", "\n\n", "\n\n\n")}
    L_module = model.model.layers[layer - 1]

    def hook(module, inputs, output):
        if isinstance(output, tuple):
            hs = output[0]
            if hs.shape[1] > patch_pos:
                hs = hs.clone()
                add = (direction_unit.to(hs.dtype).to(hs.device) * alpha_scaled)
                hs[0, patch_pos, :] = hs[0, patch_pos, :] + add
                return (hs,) + output[1:]
            return output
        else:
            if output.shape[1] > patch_pos:
                output = output.clone()
                add = (direction_unit.to(output.dtype).to(output.device) * alpha_scaled)
                output[0, patch_pos, :] = output[0, patch_pos, :] + add
            return output

    handle = L_module.register_forward_hook(hook)
    off_digit = False
    try:
        cur = input_ids
        gen_ids: List[int] = []
        for step in range(max_conf_tokens):
            out = model(cur, use_cache=False)
            logits = out.logits[0, -1, :]
            next_id = int(logits.argmax().item())
            gen_ids.append(next_id)
            if step == 0 and next_id not in digit_token_ids:
                off_digit = True
            cur = torch.cat([cur, torch.tensor([[next_id]], device=device)], dim=1)
            if next_id == eos_id or next_id in newline_ids:
                break
        txt = tok.decode(gen_ids, skip_special_tokens=True)
        return parse_confidence(txt), gen_ids, off_digit
    finally:
        handle.remove()


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    alphas = [float(x) for x in args.alphas.split(",")]

    # ---- Load M2 top-1 site --------------------------------------------
    with (Path(args.m2_dir) / "top_k_sites.json").open() as f:
        top_k = json.load(f)["top_k"]
    if args.site == "AUTO":
        site_pos, site_L = top_k[0]["position"], top_k[0]["layer"]
    else:
        site_pos, site_L = parse_site(args.site)
    print(f"[m5] steering at site {site_pos}L{site_L} method={args.direction_method}", flush=True)

    # ---- Load model + tokenizer -----------------------------------------
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[m5] Loading model...", flush=True)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(args.model, dtype=dtype)
    device = next(model.parameters()).device
    print(f"[m5] Model loaded in {time.time()-t0:.1f}s on {device}", flush=True)

    # Digit vocab: token ids for '0'..'9' (single-token; Gemma3's tokenizer
    # splits digits as separate tokens in most cases).
    digit_token_ids = set()
    for d in "0123456789":
        ids = tok(d, add_special_tokens=False).input_ids
        for tid in ids:
            digit_token_ids.add(tid)

    # ---- Load M1 items + activations at the steering site ---------------
    items, acts_path = load_m1_seed(args.m1_cache, args.seed, args.template)
    items = [it for it in items if it.get("verbal_conf") is not None]
    if args.n_items > 0:
        items = items[: args.n_items]
    with h5py.File(acts_path, "r") as h5:
        X_site = h5[site_pos][str(site_L)][()]  # (N_total, hidden)
    # We only need activations for the used subset (matched by idx)
    idxs = [it["idx"] for it in items]
    X_site = X_site[idxs]

    y = np.array([it["verbal_conf"] for it in items], dtype=np.float32)

    # ---- Direction extraction on train split ----------------------------
    n = len(items)
    rng = np.random.RandomState(args.seed + 17)
    perm = rng.permutation(n)
    n_train = int(round(n * args.direction_train_split))
    train_idx = perm[:n_train]
    eval_idx = perm[n_train:]

    X_train = X_site[train_idx]
    y_train = y[train_idx]
    high_mask = y_train >= 70
    low_mask = y_train <= 30
    if high_mask.sum() < 20 or low_mask.sum() < 20:
        # Fall back to top/bottom terciles
        thr_hi = float(np.quantile(y_train, 0.66))
        thr_lo = float(np.quantile(y_train, 0.34))
        high_mask = y_train >= thr_hi
        low_mask = y_train <= thr_lo
        print(f"[m5] adaptive thresholds: high≥{thr_hi} low≤{thr_lo}", flush=True)
    print(f"[m5] direction from n_high={int(high_mask.sum())} n_low={int(low_mask.sum())}", flush=True)

    d = extract_direction(X_train[high_mask], X_train[low_mask], args.direction_method)
    d_norm = float(np.linalg.norm(d))
    if d_norm < 1e-6:
        raise RuntimeError("Extracted direction has near-zero norm.")
    u = d / d_norm  # unit direction
    u_t = torch.tensor(u, device=device)

    # σ_proj on train split
    projections = X_train @ u
    sigma_proj = float(np.std(projections))
    print(f"[m5] σ_proj (train, unit u) = {sigma_proj:.4f}  ||d||={d_norm:.4f}", flush=True)

    # ---- Sweep α on eval split -----------------------------------------
    eval_items = [items[i] for i in eval_idx]
    print(f"[m5] evaluating on {len(eval_items)} held-out items over α={alphas}", flush=True)

    per_alpha: Dict[float, Dict] = {}
    per_item_records: List[Dict] = []
    for alpha in alphas:
        alpha_scaled = alpha * sigma_proj
        conf_out = []
        off_digit_count = 0
        for i, it in enumerate(eval_items):
            input_ids = torch.tensor([it["full_ids"][: it["conf_gen_position"]]], device=device)
            # Position label → absolute position in the input
            if site_pos.startswith("E"):
                patch_pos = it["post_answer_positions"].get(site_pos)
            else:
                patch_pos = input_ids.shape[1] - 1
            if patch_pos is None or patch_pos >= input_ids.shape[1]:
                continue
            conf_val, gen_ids, off_digit = steered_decode_conf(
                model, tok, input_ids, site_L, patch_pos,
                u_t, alpha_scaled, digit_token_ids,
            )
            if off_digit:
                off_digit_count += 1
            if conf_val is None:
                continue
            conf_out.append(conf_val)
            per_item_records.append({
                "alpha": alpha,
                "idx": it["idx"],
                "conf": conf_val,
                "off_digit_first": off_digit,
            })
            if (i + 1) % 25 == 0:
                print(f"[m5] α={alpha:+.1f} item {i+1}/{len(eval_items)} conf_mean_so_far={np.mean(conf_out):.1f} "
                      f"off_digit_rate={off_digit_count/(i+1):.3f} elapsed={(time.time()-t0)/60:.1f}m", flush=True)
        per_alpha[alpha] = {
            "mean": float(np.mean(conf_out)) if conf_out else 0.0,
            "std": float(np.std(conf_out)) if conf_out else 0.0,
            "n": len(conf_out),
            "off_digit_rate": off_digit_count / max(1, len(eval_items)),
        }
        print(f"[m5] α={alpha:+.1f} conf_mean={per_alpha[alpha]['mean']:.1f} (N={per_alpha[alpha]['n']}) "
              f"off_digit_rate={per_alpha[alpha]['off_digit_rate']:.3f}", flush=True)

    # ---- Compute monotone_r_squared ------------------------------------
    a_arr = np.array(sorted(per_alpha.keys()), dtype=np.float64)
    m_arr = np.array([per_alpha[a]["mean"] for a in a_arr], dtype=np.float64)
    # Linear fit through (α, mean_conf)
    if len(a_arr) >= 2 and np.std(a_arr) > 0:
        slope, intercept = np.polyfit(a_arr, m_arr, 1)
        pred = slope * a_arr + intercept
        ss_res = np.sum((m_arr - pred) ** 2)
        ss_tot = np.sum((m_arr - np.mean(m_arr)) ** 2)
        monotone_r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        slope_sign = float(np.sign(slope))
    else:
        monotone_r2 = 0.0
        slope_sign = 0.0

    # ---- Collapse detection --------------------------------------------
    collapse_flag = any(per_alpha[a]["off_digit_rate"] > 0.10 for a in per_alpha if abs(a) >= 2)

    summary = {
        "site": f"{site_pos}L{site_L}",
        "direction_method": args.direction_method,
        "seed": args.seed,
        "n_eval": len(eval_items),
        "sigma_proj_train": sigma_proj,
        "direction_norm": d_norm,
        "per_alpha": {str(a): per_alpha[a] for a in a_arr},
        "monotone_r_squared": float(monotone_r2),
        "monotone_slope_sign": slope_sign,
        "verb_conf_mean_at_pos4": per_alpha.get(4.0, {}).get("mean", 0.0),
        "verb_conf_mean_at_neg4": per_alpha.get(-4.0, {}).get("mean", 0.0),
        "collapse_flag": bool(collapse_flag),
    }
    print(f"[m5] SUMMARY: {json.dumps(summary, indent=2)}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"summary": summary, "records": per_item_records}, f, indent=2)
    print(f"[m5] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
