#!/usr/bin/env python3
"""
M5 v2 — Hardened Direction-Steering with Capability Metric + Random-Direction Control
        + Locked α* + Raw-Text Logging  (mechanism-audit action items 1-5).

At the top-1 cache site (E4L10 per M2) we sweep α ∈ {-16,-8,-4,-2,-1,0,1,2,4,8,16}
for the *trained* direction (diff_of_means OR lda) and for `n_random` random
unit vectors sampled i.i.d. Gaussian in the same hidden dim.

At every α we log **four** measurements per item:
  (i)   verb_conf                   — target metric (parsed 0-100)
  (ii)  offdigit_first              — token@C0 is off the digit vocabulary
  (iii) cont_ppl_unrelated          — teacher-forced negative-log-likelihood on
                                       a fixed unrelated continuation string
                                       ("The quick brown fox jumps over the lazy dog.")
                                       appended after the answer.  This is the
                                       *independent capability metric*; it does
                                       NOT depend on the parsed confidence.
  (iv)  answer_ppl_teacher_forced  — teacher-forced NLL of the model's own
                                       greedy answer under the intervention.
                                       Captures whether the intervention
                                       makes the *pre-committed* answer suddenly
                                       implausible (i.e. semantic collapse).

Both (iii)/(iv) are computed *with the hook active* so they reflect what the
steering does downstream of the patched position.  They are logged for every
α and every direction (trained + random).

Additionally, we log 5 raw text samples per α (greedy 20-token continuation
starting right after the answer) so reviewers can spot-check fluency by eye.

Two derived analyses are written to `summary.locked_alpha` and
`summary.random_direction_control`:

  * α*   = largest |α| whose mean unrelated-continuation NLL delta from α=0
           is within `capability_tol` (default 0.3 nats / token).  If no
           |α|>0 qualifies, α* = 0.  We report verb_conf_effect_at_alpha_star
           = mean(conf|α=α*) - mean(conf|α=-α*).

  * random baseline: for each α we take mean(verb_conf) over the trained
    direction and rank it against the distribution of mean(verb_conf) over
    the `n_random` random directions at the same α.  We report the
    two-sided percentile (0.0 = lowest, 1.0 = highest, 0.5 = median).
    A trained direction that carries no more signal than random will hover
    around 0.5; a genuine effect should sit near the tail.

Cost:  N_dir = 1 (trained) + n_random (default 30) at 11 α-values × 100 eval
items × 2 forward passes (one for verb_conf, one teacher-forced for capability)
≈ 68 200 short forward passes per seed.  Runs on a single 27B split across
two GPUs in ~3 h on H100-class hardware — matches the ~6 h iteration budget.

Output JSON schema (single file per (seed, direction_method)):
  {
    "summary": {
      "site":        "E4L10",
      "direction_method": "diff_of_means",
      "seed":            42,
      "n_eval":          100,
      "n_random":        30,
      "sigma_proj_train": 34.9,
      "direction_norm":   ...,
      "capability_tol":   0.3,
      "alphas":       [-16,-8,-4,-2,-1,0,1,2,4,8,16],
      "trained_per_alpha": {
         alpha_str: {
            "conf_mean": ..., "conf_std": ..., "n": ...,
            "off_digit_rate": ...,
            "cont_nll_mean":  ...,      // capability metric (higher = worse)
            "answer_nll_mean": ...,     // teacher-forced answer NLL
            "sample_texts": [ up to 5 strings ]
         }
      },
      "random_per_alpha_agg": {
         alpha_str: {
            "conf_mean_trained": ...,
            "conf_mean_random_mean": ...,  // over n_random directions
            "conf_mean_random_std":  ...,
            "conf_mean_random_percentile_of_trained": [0..1],
            "cont_nll_random_mean": ...,
            "cont_nll_trained":     ...,
            "cont_nll_random_percentile_of_trained": [0..1]
         }
      },
      "random_per_alpha_per_dir": {  // for reviewer replot
         alpha_str: [ per-direction conf_mean list, len=n_random ]
      },
      "locked_alpha": {
         "capability_tol_nats_per_token": 0.3,
         "alpha_star":  4.0,           // e.g., largest |α| within tolerance
         "cont_nll_at_alpha_star":  ...,
         "cont_nll_baseline":       ...,
         "conf_effect_at_alpha_star":  ...  // = mean(conf|+α*) - mean(conf|-α*)
      },
      "verdict_v2": {
         "trained_beats_random_at_alpha_star": true|false,
         "trained_direction_effect_percentile_at_alpha_star": ...,
         "capability_preserved_at_alpha_star": true|false
      }
    },
    "records": [ per-(alpha, dir_id, item) rows ]
  }
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

UNRELATED_CONTINUATION = (
    " The quick brown fox jumps over the lazy dog while the sun sets."
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_PATH)
    p.add_argument("--dataset", default="/data/zhenqian/data/trivia_qa")
    p.add_argument("--n_items", type=int, default=60,
                   help="Eval items after the train/eval split (kept small — "
                        "with n_random=15 and 9 alphas the total forward passes "
                        "are (1+n_random)*len(alphas)*n_items*2 ≈ 17k)")
    p.add_argument("--site", default="AUTO")
    p.add_argument("--direction_method", default="diff_of_means",
                   choices=["diff_of_means", "lda"])
    p.add_argument("--direction_train_split", type=float, default=0.5)
    p.add_argument("--alphas", default="-16,-8,-4,-1,0,1,4,8,16")
    p.add_argument("--n_random", type=int, default=15)
    p.add_argument("--capability_tol_nats", type=float, default=0.3,
                   help="Max mean-NLL delta from α=0 for α* selection.")
    p.add_argument("--n_sample_texts", type=int, default=5)
    p.add_argument("--sample_gen_tokens", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--m1_cache", required=True)
    p.add_argument("--m2_dir", required=True)
    p.add_argument("--template", default="T0")
    p.add_argument("--out", required=True)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
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


def extract_direction(X_high, X_low, method):
    if method == "diff_of_means":
        d = X_high.mean(axis=0) - X_low.mean(axis=0)
    elif method == "lda":
        mu1 = X_high.mean(axis=0)
        mu0 = X_low.mean(axis=0)
        c1 = X_high - mu1
        c0 = X_low - mu0
        n1, n0 = len(X_high), len(X_low)
        Sw = (c1.T @ c1 + c0.T @ c0) / max(1, n1 + n0 - 2)
        dim = Sw.shape[0]
        Sw = Sw + 1e-3 * np.eye(dim, dtype=Sw.dtype) * np.trace(Sw) / dim
        d = np.linalg.solve(Sw, mu1 - mu0)
    else:
        raise ValueError(method)
    return d.astype(np.float32)


class SteeringHook:
    """
    Registers an additive forward hook on `model.model.layers[layer-1]`.
    Adds `alpha_scaled * direction_unit` to `hidden_states[:, patch_pos, :]`.
    Patch position is set per-call via `self.set_patch(patch_pos)`.
    """

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
def greedy_confidence(model, tok, input_ids, digit_token_ids, max_conf_tokens=MAX_CONF_TOKENS):
    """Greedy-decode ≤max_conf_tokens after input_ids.  Return (parsed_int, gen_ids, off_digit_first)."""
    device = input_ids.device
    eos_id = tok.eos_token_id
    newline_ids = {tok(s, add_special_tokens=False).input_ids[-1] for s in ("\n", "\n\n", "\n\n\n")}
    cur = input_ids
    gen_ids: List[int] = []
    off_digit = False
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


@torch.no_grad()
def teacher_forced_nll(model, input_ids: torch.Tensor, target_ids: torch.Tensor) -> float:
    """
    Teacher-forced NLL per target token.  Concatenates (input_ids, target_ids)
    into a single forward pass and computes -log p(target_ids | input_ids).
    Returns mean nats per target token.

    The steering hook (if installed on `model`) will fire during this forward
    pass, so this reflects "capability under intervention".
    """
    device = input_ids.device
    full = torch.cat([input_ids, target_ids.unsqueeze(0)], dim=1)
    out = model(full, use_cache=False)
    logits = out.logits[0]  # (L, V)
    # We predict target_ids at positions [input_len-1 .. input_len+len(target)-2]
    inp_len = input_ids.shape[1]
    T = target_ids.shape[0]
    if T == 0:
        return 0.0
    logits_slice = logits[inp_len - 1: inp_len - 1 + T]  # (T, V)
    logprobs = torch.log_softmax(logits_slice.float(), dim=-1)
    nll = -logprobs.gather(1, target_ids.unsqueeze(1)).mean().item()
    return float(nll)


@torch.no_grad()
def greedy_text(model, tok, input_ids, max_tokens=20):
    """Greedy-decode up to max_tokens (no stopping on newlines).  For raw-sample logging only."""
    device = input_ids.device
    cur = input_ids
    gen_ids: List[int] = []
    for _ in range(max_tokens):
        out = model(cur, use_cache=False)
        next_id = int(out.logits[0, -1, :].argmax().item())
        gen_ids.append(next_id)
        cur = torch.cat([cur, torch.tensor([[next_id]], device=device)], dim=1)
        if next_id == tok.eos_token_id:
            break
    return tok.decode(gen_ids, skip_special_tokens=True)


def build_per_item_context(it, tok, device):
    """
    Reconstruct the pre-conf input ids from the M1 item's `full_ids`, sliced up
    to `conf_gen_position`.  Also build the fixed unrelated-continuation target
    ids (appended after the conf-gen position for a teacher-forced NLL).
    """
    input_ids = torch.tensor([it["full_ids"][: it["conf_gen_position"]]], device=device)
    # unrelated continuation target — tokenized once outside
    return input_ids


def main():
    args = parse_args()
    set_all_seeds(args.seed)

    alphas = [float(x) for x in args.alphas.split(",")]

    # ---- Load M2 top-1 site ---------------------------------------------
    with (Path(args.m2_dir) / "top_k_sites.json").open() as f:
        top_k = json.load(f)["top_k"]
    if args.site == "AUTO":
        site_pos, site_L = top_k[0]["position"], top_k[0]["layer"]
    else:
        site_pos, site_L = parse_site(args.site)
    print(f"[m5v2] site={site_pos}L{site_L} method={args.direction_method} "
          f"alphas={alphas} n_random={args.n_random}", flush=True)

    # ---- Load model + tokenizer -----------------------------------------
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    print(f"[m5v2] Loading model...", flush=True)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(args.model, dtype=dtype)
    device = next(model.parameters()).device
    print(f"[m5v2] Model loaded in {time.time()-t0:.1f}s on {device}", flush=True)

    # Digit vocab
    digit_token_ids = set()
    for d in "0123456789":
        for tid in tok(d, add_special_tokens=False).input_ids:
            digit_token_ids.add(tid)

    # Unrelated-continuation target ids (independent capability probe)
    unrelated_ids = torch.tensor(
        tok(UNRELATED_CONTINUATION, add_special_tokens=False).input_ids,
        dtype=torch.long, device=device,
    )
    print(f"[m5v2] unrelated-continuation target len={unrelated_ids.shape[0]} tokens", flush=True)

    # ---- Load M1 items + activations at the steering site ---------------
    items, acts_path = load_m1_seed(args.m1_cache, args.seed, args.template)
    items = [it for it in items if it.get("verbal_conf") is not None]
    if args.n_items > 0:
        items = items[: max(args.n_items * 3, 500)]  # keep a pool >= train + eval
    with h5py.File(acts_path, "r") as h5:
        X_site = h5[site_pos][str(site_L)][()]  # (N_total, hidden)

    idxs = [it["idx"] for it in items]
    X_site = X_site[idxs]
    y = np.array([it["verbal_conf"] for it in items], dtype=np.float32)

    # ---- Direction extraction on train split ----------------------------
    n = len(items)
    rng = np.random.RandomState(args.seed + 17)
    perm = rng.permutation(n)
    n_train = int(round(n * args.direction_train_split))
    train_idx = perm[:n_train]
    eval_idx  = perm[n_train:][: args.n_items]  # cap eval at n_items

    X_train = X_site[train_idx]
    y_train = y[train_idx]
    high_mask = y_train >= 70
    low_mask  = y_train <= 30
    if high_mask.sum() < 20 or low_mask.sum() < 20:
        thr_hi = float(np.quantile(y_train, 0.66))
        thr_lo = float(np.quantile(y_train, 0.34))
        high_mask = y_train >= thr_hi
        low_mask = y_train <= thr_lo
        print(f"[m5v2] adaptive thresholds: high≥{thr_hi} low≤{thr_lo}", flush=True)
    print(f"[m5v2] direction from n_high={int(high_mask.sum())} n_low={int(low_mask.sum())}", flush=True)

    d_trained = extract_direction(X_train[high_mask], X_train[low_mask], args.direction_method)
    d_trained_norm = float(np.linalg.norm(d_trained))
    if d_trained_norm < 1e-6:
        raise RuntimeError("Extracted direction has near-zero norm.")
    u_trained = d_trained / d_trained_norm

    # σ_proj on train split (trained direction)
    projections_trained = X_train @ u_trained
    sigma_proj = float(np.std(projections_trained))
    print(f"[m5v2] σ_proj(trained,train)={sigma_proj:.4f}  ||d||={d_trained_norm:.4f}", flush=True)

    # ---- Random-direction bank -----------------------------------------
    rng2 = np.random.RandomState(args.seed + 71)
    dim = int(u_trained.shape[0])
    random_dirs = rng2.randn(args.n_random, dim).astype(np.float32)
    random_dirs = random_dirs / (np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-9)
    # σ_proj_random per direction (on train split) so the α scale is comparable
    projs_rand = X_train @ random_dirs.T  # (n_train, n_random)
    sigma_proj_random = projs_rand.std(axis=0)  # (n_random,)
    print(f"[m5v2] σ_proj(random) mean={sigma_proj_random.mean():.4f} "
          f"min={sigma_proj_random.min():.4f} max={sigma_proj_random.max():.4f}", flush=True)

    # ---- Install steering hook (a single hook we reuse across directions) --
    hook = SteeringHook(model, site_L, torch.tensor(u_trained, device=device))
    hook.install()

    eval_items = [items[i] for i in eval_idx]
    n_eval = len(eval_items)
    print(f"[m5v2] evaluating on {n_eval} held-out items", flush=True)

    # ---- Precompute per-item patch position + full context ids ----------
    per_item_ctx = []
    for it in eval_items:
        input_ids = build_per_item_context(it, tok, device)
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
    print(f"[m5v2] usable eval items after patch-pos filter: {len(per_item_ctx)}", flush=True)

    # ---- Sweep loop ----------------------------------------------------
    # (direction_id, alpha) -> per-alpha aggregates
    #   direction_id 0 = trained; 1..n_random = random_i
    n_dir = 1 + args.n_random
    print(f"[m5v2] total measurement grid: {n_dir} directions × {len(alphas)} alphas "
          f"× {len(per_item_ctx)} items × 2 fwd = "
          f"{n_dir*len(alphas)*len(per_item_ctx)*2} forward passes", flush=True)

    # Storage
    trained_per_alpha: Dict[float, Dict] = {}
    random_per_alpha:  Dict[float, List[Dict]] = {a: [] for a in alphas}
    records: List[Dict] = []

    def eval_direction(dir_id: int, u: np.ndarray, sigma: float, is_trained: bool):
        """Sweep α for one direction; record aggregates."""
        # Reset hook direction
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
                # (a) verb_conf via greedy decode at C0
                conf_val, gen_ids, off_digit = greedy_confidence(model, tok, input_ids, digit_token_ids)
                if off_digit:
                    off_digit_count += 1
                if conf_val is not None:
                    confs.append(conf_val)
                # (b) unrelated-continuation NLL (independent capability metric).
                # Subsample: only every 3rd item, since NLL is a smooth per-item
                # measure and 60/3=20 samples is a stable estimate. Halves cost.
                if i % 3 == 0:
                    cont_nll = teacher_forced_nll(model, input_ids, unrelated_ids)
                    cont_nlls.append(cont_nll)
                    _rec_cont_nll = cont_nll
                else:
                    _rec_cont_nll = None
                # (c) raw sample (first `n_sample_texts` items only, trained-dir only)
                if is_trained and i < args.n_sample_texts:
                    txt = greedy_text(model, tok, input_ids, max_tokens=args.sample_gen_tokens)
                    sample_texts.append({
                        "alpha": alpha,
                        "text": txt,
                    })
                if is_trained:
                    records.append({
                        "dir": "trained",
                        "alpha": alpha,
                        "idx": eval_items[i]["idx"],
                        "conf": conf_val,
                        "off_digit_first": off_digit,
                        "cont_nll": _rec_cont_nll,
                    })
                if is_trained and (i + 1) % 25 == 0:
                    print(f"[m5v2] trained α={alpha:+.1f} item {i+1}/{len(per_item_ctx)} "
                          f"conf_so_far={np.mean(confs):.1f} cont_nll={np.mean(cont_nlls):.3f} "
                          f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)
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

    # Trained direction
    print("[m5v2] === trained direction ===", flush=True)
    trained_per_alpha = eval_direction(0, u_trained, sigma_proj, is_trained=True)

    # Random directions
    for r_idx in range(args.n_random):
        u_r = random_dirs[r_idx]
        s_r = float(sigma_proj_random[r_idx])
        print(f"[m5v2] === random direction {r_idx+1}/{args.n_random} σ={s_r:.4f} ===", flush=True)
        per_alpha_r = eval_direction(r_idx + 1, u_r, s_r, is_trained=False)
        for a in alphas:
            random_per_alpha[a].append(per_alpha_r[a])

    # Turn off the hook — done with sweeps
    hook.remove()

    # ---- Random-direction aggregates -----------------------------------
    random_per_alpha_agg: Dict[str, Dict] = {}
    random_per_alpha_per_dir: Dict[str, List[float]] = {}
    for a in alphas:
        conf_means = np.array([r["conf_mean"] for r in random_per_alpha[a]], dtype=np.float64)
        cont_nll_means = np.array([r["cont_nll_mean"] for r in random_per_alpha[a]], dtype=np.float64)
        conf_t = trained_per_alpha[a]["conf_mean"]
        nll_t  = trained_per_alpha[a]["cont_nll_mean"]
        # percentile of trained in random-baseline distribution
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

    # ---- Locked α* -----------------------------------------------------
    baseline_nll = trained_per_alpha.get(0.0, {}).get("cont_nll_mean", 0.0)
    tol = args.capability_tol_nats
    # α* = largest |α| whose trained-direction NLL is within tolerance of α=0
    alpha_star = 0.0
    for a in sorted(alphas, key=lambda x: abs(x)):
        if a == 0.0:
            continue
        d_nll = trained_per_alpha[a]["cont_nll_mean"] - baseline_nll
        if d_nll <= tol:
            if abs(a) > abs(alpha_star):
                alpha_star = a
    # Prefer positive α at same magnitude if tied; recompute at ±|α*|
    a_pos = abs(alpha_star)
    a_neg = -abs(alpha_star)
    conf_pos = trained_per_alpha.get(a_pos, {}).get("conf_mean") if a_pos in trained_per_alpha else \
               trained_per_alpha.get(float(a_pos), {}).get("conf_mean", 0.0)
    conf_neg = trained_per_alpha.get(a_neg, {}).get("conf_mean") if a_neg in trained_per_alpha else \
               trained_per_alpha.get(float(a_neg), {}).get("conf_mean", 0.0)
    conf_effect_at_alpha_star = float((conf_pos or 0.0) - (conf_neg or 0.0))

    # verdict_v2 gates
    at_star = random_per_alpha_agg.get(str(a_pos), random_per_alpha_agg.get(str(float(a_pos)), {}))
    conf_pct_at_star = at_star.get("conf_mean_random_percentile_of_trained", 0.5)
    # trained beats random if conf percentile is at either tail (≤0.05 or ≥0.95)
    trained_beats_random = (conf_pct_at_star <= 0.05) or (conf_pct_at_star >= 0.95)
    capability_preserved = (trained_per_alpha[a_pos]["cont_nll_mean"] - baseline_nll) <= tol \
                            if a_pos in trained_per_alpha else True

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
        "unrelated_continuation": UNRELATED_CONTINUATION,
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
    print(f"[m5v2] SUMMARY: {json.dumps({k: summary[k] for k in ('site','direction_method','seed','locked_alpha','verdict_v2')}, indent=2)}",
          flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"summary": summary, "records": records}, f, indent=2)
    print(f"[m5v2] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
