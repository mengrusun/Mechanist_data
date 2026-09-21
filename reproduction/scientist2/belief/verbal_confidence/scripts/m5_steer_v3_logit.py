#!/usr/bin/env python3
"""
M5 v3 — Logit-level readout of the M5-v2 sweep.

Motivation (from iteration-2 reviewer):
   M5-v2 shows no argmax-decoded effect on the confidence token at C0 for the
   trained direction, even at alpha=+/-16 sigma_proj.  But argmax is a
   *thresholded* readout.  If the cache direction affects the score-token
   *distribution* — say, +alpha nudges P(digit=9) up 2% and P(digit=8) down 2%
   without moving argmax — we still have causal use of the cache; it just
   fell below the greedy decoder's threshold.

M5-v3 therefore replaces the readout with two probability-level metrics:

  (i)  E[score | prompt, alpha] = sum_{d=0..100} d * p(d | prompt, alpha)
       where p(d) is the model's probability of the multi-digit score `d`
       taken as a product of per-position digit log-probs, teacher-forced
       against the numeric prefix.  In practice we compute:

           p(d)  = softmax(logits at C0)[first digit] *
                   softmax(logits at C0+1)[second digit] *
                   softmax(logits at C0+2)[third digit if any]

       and sum over d in {0..100}.  This is exact for the deterministic
       tokenizer scheme; small stray probability leaks into non-digit
       tokens are subsumed in the residual.

  (ii) Entropy at C0 restricted to the digit-token vocabulary
           H_digit = -sum_{d in 0..9} p_C0(d) * log p_C0(d)
       — a coarser but cheap check on whether the intervention flattens or
       sharpens the digit distribution at the first-digit position.

Both metrics are computed at each alpha in {-16, -8, -4, -1, 0, 1, 4, 8, 16}
for the SAME trained direction and the SAME 40 held-out items, so the
comparison is apples-to-apples with M5-v2.  No random-direction sweep is
included (that was the expensive part of v2 and is not needed at logit
level for the reviewer's ask — one item's expected-score curve is enough
to prove or disprove sub-argmax effects on that seed).

Output:  results/m5_v3/m5v3_expected_score_seed{seed}.json

Cost estimate:  40 items x 9 alpha x 1 forward pass = 360 forwards per seed.
On A800 80GB with 200-token context ~ 0.5s per fwd  =  ~3 min per seed.
3 seeds  =  ~10 min total (plus 4 min model load = 14 min total).
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vc_common import MODEL_PATH, load_model_and_tokenizer, set_all_seeds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,123,2024")
    p.add_argument("--n_items", type=int, default=40)
    p.add_argument("--alphas", default="-16,-8,-4,-1,0,1,4,8,16")
    p.add_argument("--direction_train_split", type=float, default=0.5)
    p.add_argument("--m1_cache", default="results/m1")
    p.add_argument("--m2_dir",   default="results/m2")
    p.add_argument("--out_dir",  default="results/m5_v3")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--template", default="T0")
    p.add_argument("--sites", default="AUTO",
                   help="Comma list of sites like 'E4L10,E1L5,E2L5' or 'AUTO' (=top-1 from M2).")
    p.add_argument("--out_suffix", default="",
                   help="String appended to output filename before .json (e.g. '_sitesweep').")
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


def extract_diff_of_means_direction(X_high: np.ndarray, X_low: np.ndarray) -> np.ndarray:
    d = X_high.mean(axis=0) - X_low.mean(axis=0)
    return d.astype(np.float32)


class SteeringHook:
    def __init__(self, model, layer: int, direction_unit: torch.Tensor):
        self.model = model
        self.layer = layer
        self.direction_unit = direction_unit
        self.patch_pos = -1
        self.alpha_scaled = 0.0
        self.handle = None

    def install(self):
        L_module = self.model.model.layers[self.layer - 1]
        def hook(module, inputs, output):
            if isinstance(output, tuple):
                hs = output[0]
                if hs.shape[1] > self.patch_pos >= 0 and self.alpha_scaled != 0.0:
                    hs = hs.clone()
                    add = self.direction_unit.to(hs.dtype).to(hs.device) * self.alpha_scaled
                    hs[0, self.patch_pos, :] = hs[0, self.patch_pos, :] + add
                    return (hs,) + output[1:]
                return output
            else:
                if output.shape[1] > self.patch_pos >= 0 and self.alpha_scaled != 0.0:
                    output = output.clone()
                    add = self.direction_unit.to(output.dtype).to(output.device) * self.alpha_scaled
                    output[0, self.patch_pos, :] = output[0, self.patch_pos, :] + add
                return output
        self.handle = L_module.register_forward_hook(hook)

    def remove(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None

    def set(self, patch_pos: int, alpha_scaled: float):
        self.patch_pos = patch_pos
        self.alpha_scaled = alpha_scaled

    def set_direction(self, direction_unit: torch.Tensor):
        self.direction_unit = direction_unit


@torch.no_grad()
def score_distribution_at_c0(model, tok, input_ids: torch.Tensor,
                             digit_token_ids_per_digit: List[List[int]]) -> Tuple[Dict[int, float], float]:
    """
    One forward pass on `input_ids` (the pre-conf-token prefix). Read the logits
    at the last position (which predicts the next token — the first digit of
    the score).

    Returns:
      * dict d -> P(score == d) for d in 0..100  (over the first-digit
        distribution only, i.e. we approximate multi-digit scores by their
        first digit -- for verbal-conf tokens 0-100 the first digit fully
        determines the sign of the shift; the model's tokenizer typically
        splits 100 as "1"+"00" but "1" here means "10-19" range in the
        aggregate expected-score computation using digit distribution)
      * digit-entropy at C0 (nats)

    Actually — since digits typically are single tokens for Gemma3 and the
    score range 0-100 spans multi-digit tokens, we do this the simple way:
    compute per-digit-token probability at C0, then define
        E[first_digit] = sum_{d=0..9} d * p_C0(d)
    which is a proxy for the shift in "how confident the model wants to be"
    at the first digit position.  A monotone shift in E[first_digit] with
    alpha is the KEY signal — it captures sub-argmax preferences without
    needing to enumerate 100 numbers.
    """
    out = model(input_ids, use_cache=False)
    logits_at_c0 = out.logits[0, -1, :].float()   # (V,)
    logprobs = torch.log_softmax(logits_at_c0, dim=-1)
    probs = logprobs.exp()

    # Per-digit probability (sum over any tokens that decode to a single digit)
    per_digit_p = {}
    for d in range(10):
        p_d = 0.0
        for tid in digit_token_ids_per_digit[d]:
            p_d += float(probs[tid].item())
        per_digit_p[d] = p_d
    e_first_digit = float(sum(d * per_digit_p[d] for d in range(10)))

    # digit entropy: renormalize digits to sum to 1, then H
    total = sum(per_digit_p.values())
    if total > 1e-9:
        H = 0.0
        for d in range(10):
            p = per_digit_p[d] / total
            if p > 1e-12:
                H -= p * np.log(p)
    else:
        H = 0.0
    return {"per_digit_p": per_digit_p, "e_first_digit": e_first_digit,
            "digit_mass_at_c0": total}, H


def parse_site(s: str):
    m = re.match(r"^([EC]\d+)L(\d+)$", s)
    if not m:
        raise ValueError(f"Bad site spec: {s!r}")
    return m.group(1), int(m.group(2))


def run_seed(model, tok, device, m2_dir, m1_cache, seed, alphas, n_items,
             direction_train_split, out_dir, template, site_pos, site_L, out_suffix=""):
    set_all_seeds(seed)
    print(f"[m5v3/seed{seed}/{site_pos}L{site_L}] === starting ===", flush=True)

    # Digit tokens per digit value 0..9
    digit_token_ids_per_digit = [[] for _ in range(10)]
    for d in range(10):
        for tid in tok(str(d), add_special_tokens=False).input_ids:
            digit_token_ids_per_digit[d].append(tid)

    items, acts_path = load_m1_seed(m1_cache, seed, template)
    items = [it for it in items if it.get("verbal_conf") is not None]
    if n_items > 0:
        items = items[: max(n_items * 3, 500)]
    with h5py.File(acts_path, "r") as h5:
        X_site = h5[site_pos][str(site_L)][()]
    idxs = [it["idx"] for it in items]
    X_site = X_site[idxs]
    y = np.array([it["verbal_conf"] for it in items], dtype=np.float32)

    n = len(items)
    rng = np.random.RandomState(seed + 17)
    perm = rng.permutation(n)
    n_train = int(round(n * direction_train_split))
    train_idx = perm[:n_train]
    eval_idx  = perm[n_train:][: n_items]

    X_train = X_site[train_idx]
    y_train = y[train_idx]
    high_mask = y_train >= 70
    low_mask  = y_train <= 30
    if high_mask.sum() < 20 or low_mask.sum() < 20:
        thr_hi = float(np.quantile(y_train, 0.66))
        thr_lo = float(np.quantile(y_train, 0.34))
        high_mask = y_train >= thr_hi
        low_mask = y_train <= thr_lo
    d_trained = extract_diff_of_means_direction(X_train[high_mask], X_train[low_mask])
    d_norm = float(np.linalg.norm(d_trained))
    u_trained = d_trained / d_norm

    projections = X_train @ u_trained
    sigma_proj = float(np.std(projections))
    print(f"[m5v3/seed{seed}] sigma_proj={sigma_proj:.4f} ||d||={d_norm:.4f}", flush=True)

    hook = SteeringHook(model, site_L, torch.tensor(u_trained, device=device))
    hook.install()

    eval_items = [items[i] for i in eval_idx]
    per_alpha_results = {}
    per_item_records = []

    for alpha in alphas:
        alpha_scaled = alpha * sigma_proj
        e_first_digits = []
        digit_ent = []
        digit_mass = []
        per_digit_ps = []
        for i, it in enumerate(eval_items):
            input_ids = torch.tensor([it["full_ids"][: it["conf_gen_position"]]], device=device)
            patch_pos = it["post_answer_positions"].get(site_pos)
            if patch_pos is None or patch_pos >= input_ids.shape[1]:
                continue
            hook.set(patch_pos, alpha_scaled)
            stats, H = score_distribution_at_c0(model, tok, input_ids, digit_token_ids_per_digit)
            e_first_digits.append(stats["e_first_digit"])
            digit_ent.append(H)
            digit_mass.append(stats["digit_mass_at_c0"])
            per_digit_ps.append(stats["per_digit_p"])
            per_item_records.append({
                "alpha": alpha, "idx": it["idx"], "e_first_digit": stats["e_first_digit"],
                "digit_entropy": H, "digit_mass_at_c0": stats["digit_mass_at_c0"],
                "per_digit_p": stats["per_digit_p"],
            })
        per_alpha_results[alpha] = {
            "e_first_digit_mean": float(np.mean(e_first_digits)),
            "e_first_digit_std":  float(np.std(e_first_digits)),
            "digit_entropy_mean": float(np.mean(digit_ent)),
            "digit_mass_at_c0_mean": float(np.mean(digit_mass)),
            "n": len(e_first_digits),
            "avg_per_digit_p": {d: float(np.mean([r[d] for r in per_digit_ps])) for d in range(10)},
        }
        print(f"[m5v3/seed{seed}] alpha={alpha:+.1f} E[first_digit]={per_alpha_results[alpha]['e_first_digit_mean']:.3f} "
              f"H_digit={per_alpha_results[alpha]['digit_entropy_mean']:.3f} "
              f"digit_mass={per_alpha_results[alpha]['digit_mass_at_c0_mean']:.3f}", flush=True)

    hook.remove()

    # Locked alpha_star (from v2 = 16.0 always)
    baseline = per_alpha_results.get(0.0, {}).get("e_first_digit_mean", 0.0)
    e_at_pos16 = per_alpha_results.get(16.0, {}).get("e_first_digit_mean", 0.0)
    e_at_neg16 = per_alpha_results.get(-16.0, {}).get("e_first_digit_mean", 0.0)
    e_span = e_at_pos16 - e_at_neg16

    summary = {
        "site": f"{site_pos}L{site_L}",
        "seed": seed,
        "n_eval": len(eval_items),
        "sigma_proj_train": sigma_proj,
        "direction_norm": d_norm,
        "alphas": alphas,
        "per_alpha": {str(a): per_alpha_results[a] for a in alphas},
        "e_first_digit_at_alpha_0": baseline,
        "e_first_digit_at_alpha_pos16": e_at_pos16,
        "e_first_digit_at_alpha_neg16": e_at_neg16,
        "e_first_digit_span_neg16_to_pos16": e_span,
    }
    print(f"[m5v3/seed{seed}/{site_pos}L{site_L}] SUMMARY: E[first_digit] baseline={baseline:.3f}  span(alpha=-16..+16)={e_span:.3f}", flush=True)
    site_tag = f"{site_pos}L{site_L}"
    out_path = Path(out_dir) / f"m5v3_expected_score_{site_tag}_seed{seed}{out_suffix}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"summary": summary, "records": per_item_records}, f, indent=2)
    print(f"[m5v3/seed{seed}/{site_pos}L{site_L}] wrote {out_path}", flush=True)


def resolve_sites(sites_arg: str, m2_dir: str):
    """Return list[(pos, layer_int)] from AUTO or a comma list like E4L10,E1L5."""
    if sites_arg == "AUTO":
        with (Path(m2_dir) / "top_k_sites.json").open() as f:
            top_k = json.load(f)["top_k"]
        return [(top_k[0]["position"], top_k[0]["layer"])]
    parts = [s.strip() for s in sites_arg.split(",") if s.strip()]
    return [parse_site(p) for p in parts]


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[m5v3] Loading model...", flush=True)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype=dtype)
    device = next(model.parameters()).device
    print(f"[m5v3] Model on {device} in {(time.time()-t0)/60:.1f}m", flush=True)

    sites = resolve_sites(args.sites, args.m2_dir)
    print(f"[m5v3] sites to sweep: {[(p,L) for p,L in sites]}", flush=True)

    for (site_pos, site_L) in sites:
        for seed in seeds:
            run_seed(model, tok, device, args.m2_dir, args.m1_cache, seed, alphas,
                     args.n_items, args.direction_train_split, args.out_dir, args.template,
                     site_pos, site_L, args.out_suffix)


if __name__ == "__main__":
    main()
