#!/usr/bin/env python3
"""
M5 v3 (joint variant) — joint multi-site steering at the top-5 M2 sites.

Motivation (iteration-4 reviewer): rule out the distributed-cause loophole:
information may live jointly across several weakly redundant sites, so
single-site steering underestimates the causal handle.  Test:  add
alpha * sigma_proj_s * u_s at ALL 5 top sites SIMULTANEOUSLY (one steering
hook per site, all firing on the same forward pass), and read out the
logit-level metric (E[first_digit] at C0).

Grid:
  seeds:  42, 123, 2024
  alphas: reduced grid  {-16, -8, -4, 0, 4, 8, 16}   (budget-tight per reviewer's suggestion)
  sites:  E4L10, E1L5, E2L5, E3L10, E3L5             (top-5 by M2 R^2)
  n_items: 40 per seed (same as single-site v3)
  direction: diff_of_means, per-site train/eval split identical to v2/v3

Output: results/m5_v3_joint/m5v3_joint_expected_score_seed{seed}.json

Cost estimate:
  40 items x 7 alpha x 3 seeds = 840 forwards + model load  ~ 4 + 12 = 16 min.
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
    p.add_argument("--alphas", default="-16,-8,-4,0,4,8,16")
    p.add_argument("--direction_train_split", type=float, default=0.5)
    p.add_argument("--m1_cache", default="results/m1")
    p.add_argument("--m2_dir",   default="results/m2")
    p.add_argument("--out_dir",  default="results/m5_v3_joint")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--template", default="T0")
    p.add_argument("--sites", default="E4L10,E1L5,E2L5,E3L10,E3L5",
                   help="Comma list of sites to steer jointly.")
    return p.parse_args()


def parse_site(s: str):
    m = re.match(r"^([EC]\d+)L(\d+)$", s)
    if not m:
        raise ValueError(f"Bad site spec: {s!r}")
    return m.group(1), int(m.group(2))


def load_m1_seed(m1_cache: str, seed: int, template: str):
    seed_dir = Path(m1_cache) / f"seed{seed}" / template
    items = []
    with (seed_dir / "items.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items, str(seed_dir / "activations.h5")


def extract_diff_of_means(X_high, X_low):
    return (X_high.mean(axis=0) - X_low.mean(axis=0)).astype(np.float32)


class MultiSiteSteeringHook:
    """
    Registers one forward hook per unique layer, each of which may perturb the
    hidden-state at that layer at zero or more per-site (patch_pos, direction,
    alpha_scaled) positions on the SAME forward pass.

    For a joint 5-site sweep spanning 2 layers (5 and 10), we have 2 hooks;
    each hook handles all sites at that layer.

    Usage:
      hook_mgr = MultiSiteSteeringHook(model, sites, directions_per_site)
      hook_mgr.install()
      hook_mgr.set_per_item(patch_positions_per_site, alpha)
      # then run forward pass
      hook_mgr.remove()
    """

    def __init__(self, model, site_defs: List[Tuple[str, int, np.ndarray, float]], device):
        """
        site_defs: list of tuples (site_tag, layer_int, direction_unit_np, sigma_proj)
                   e.g. [("E4L10", 10, u10_np, sigma_e4l10),
                         ("E1L5",   5, u5_np,  sigma_e1l5), ...]
        """
        self.model = model
        self.device = device
        # Group by layer
        by_layer: Dict[int, List[Dict]] = {}
        for tag, L, u_np, sigma in site_defs:
            by_layer.setdefault(L, []).append({
                "tag": tag,
                "unit": torch.tensor(u_np, device=device),
                "sigma": float(sigma),
                "patch_pos": -1,   # set per item
                "alpha_scaled": 0.0,
            })
        self.by_layer = by_layer
        self.handles = []

    def install(self):
        for L, sites in self.by_layer.items():
            L_module = self.model.model.layers[L - 1]
            sites_ref = sites  # closure

            def make_hook(sites_ref):
                def hook(module, inputs, output):
                    if isinstance(output, tuple):
                        hs = output[0]
                        cloned = False
                        for s in sites_ref:
                            pp = s["patch_pos"]
                            if hs.shape[1] > pp >= 0 and s["alpha_scaled"] != 0.0:
                                if not cloned:
                                    hs = hs.clone()
                                    cloned = True
                                add = s["unit"].to(hs.dtype).to(hs.device) * s["alpha_scaled"]
                                hs[0, pp, :] = hs[0, pp, :] + add
                        if cloned:
                            return (hs,) + output[1:]
                        return output
                    else:
                        cloned = False
                        for s in sites_ref:
                            pp = s["patch_pos"]
                            if output.shape[1] > pp >= 0 and s["alpha_scaled"] != 0.0:
                                if not cloned:
                                    output = output.clone()
                                    cloned = True
                                add = s["unit"].to(output.dtype).to(output.device) * s["alpha_scaled"]
                                output[0, pp, :] = output[0, pp, :] + add
                        return output
                return hook

            self.handles.append(L_module.register_forward_hook(make_hook(sites_ref)))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def set_per_item(self, patch_positions_per_site: Dict[str, int], alpha: float):
        """Set patch position for each site (by tag) and the shared alpha (scaled per-site by that site's sigma_proj).

        alpha_scaled at site s = alpha * sigma_proj_s.
        """
        for L, sites in self.by_layer.items():
            for s in sites:
                s["patch_pos"] = int(patch_positions_per_site.get(s["tag"], -1))
                s["alpha_scaled"] = alpha * s["sigma"]


@torch.no_grad()
def score_dist_at_c0(model, input_ids, digit_token_ids_per_digit):
    out = model(input_ids, use_cache=False)
    logits = out.logits[0, -1, :].float()
    probs = torch.log_softmax(logits, dim=-1).exp()
    per_digit = {}
    for d in range(10):
        p = 0.0
        for tid in digit_token_ids_per_digit[d]:
            p += float(probs[tid].item())
        per_digit[d] = p
    e_first = float(sum(d * per_digit[d] for d in range(10)))
    tot = sum(per_digit.values())
    if tot > 1e-9:
        H = 0.0
        for d in range(10):
            p = per_digit[d] / tot
            if p > 1e-12:
                H -= p * np.log(p)
    else:
        H = 0.0
    return {"per_digit_p": per_digit, "e_first_digit": e_first, "digit_mass": tot, "H_digit": H}


def build_site_defs(items_train, X_site_train, y_train, method="diff_of_means"):
    """Return (u_hat_np, sigma_proj) for a site given train-split activations/labels."""
    high_mask = y_train >= 70
    low_mask  = y_train <= 30
    if high_mask.sum() < 20 or low_mask.sum() < 20:
        thr_hi = float(np.quantile(y_train, 0.66))
        thr_lo = float(np.quantile(y_train, 0.34))
        high_mask = y_train >= thr_hi
        low_mask = y_train <= thr_lo
    d = extract_diff_of_means(X_site_train[high_mask], X_site_train[low_mask])
    d_norm = float(np.linalg.norm(d))
    u_hat = d / d_norm
    projs = X_site_train @ u_hat
    sigma = float(np.std(projs))
    return u_hat, sigma, d_norm


def run_seed(model, tok, device, m2_dir, m1_cache, seed, alphas, n_items,
             direction_train_split, out_dir, template, site_list):
    set_all_seeds(seed)
    print(f"[joint/seed{seed}] === starting; sites={site_list} ===", flush=True)

    # Load items and activations for ALL used positions (E1, E2, E3, E4)
    items, acts_path = load_m1_seed(m1_cache, seed, template)
    items = [it for it in items if it.get("verbal_conf") is not None]
    if n_items > 0:
        items = items[: max(n_items * 3, 500)]

    # Load per-position activation slabs for the unique positions used
    unique_positions = sorted(set(p.split("L")[0] for p in site_list))
    unique_layers = sorted(set(int(p.split("L")[1]) for p in site_list))
    slabs: Dict[str, np.ndarray] = {}
    with h5py.File(acts_path, "r") as h5:
        for pos in unique_positions:
            for L in unique_layers:
                # Only need slabs that are actually used
                needs = f"{pos}L{L}"
                if any(needs == s for s in site_list):
                    slabs[needs] = h5[pos][str(L)][()]
    print(f"[joint/seed{seed}] loaded slabs for {list(slabs.keys())}", flush=True)

    idxs = [it["idx"] for it in items]
    for k in slabs:
        slabs[k] = slabs[k][idxs]
    y = np.array([it["verbal_conf"] for it in items], dtype=np.float32)

    n = len(items)
    rng = np.random.RandomState(seed + 17)
    perm = rng.permutation(n)
    n_train = int(round(n * direction_train_split))
    train_idx = perm[:n_train]
    eval_idx  = perm[n_train:][: n_items]
    y_train = y[train_idx]

    # Build direction + sigma per site
    site_defs = []
    for site in site_list:
        pos, L = parse_site(site)
        u_hat, sigma, dn = build_site_defs([items[i] for i in train_idx], slabs[site][train_idx], y_train)
        print(f"[joint/seed{seed}/{site}] sigma_proj={sigma:.3f} ||d||={dn:.3f}", flush=True)
        site_defs.append((site, L, u_hat, sigma))

    hook_mgr = MultiSiteSteeringHook(model, site_defs, device)
    hook_mgr.install()

    digit_token_ids_per_digit = [[] for _ in range(10)]
    for d in range(10):
        for tid in tok(str(d), add_special_tokens=False).input_ids:
            digit_token_ids_per_digit[d].append(tid)

    eval_items = [items[i] for i in eval_idx]
    per_alpha = {}
    per_item_records = []
    t0 = time.time()

    for alpha in alphas:
        confs_e = []
        entropies = []
        masses = []
        for i, it in enumerate(eval_items):
            input_ids = torch.tensor([it["full_ids"][: it["conf_gen_position"]]], device=device)
            # per-site patch positions for this item
            pp_map = {}
            for (tag, L, u, sig) in site_defs:
                pos, _ = parse_site(tag)
                pp = it["post_answer_positions"].get(pos, -1)
                pp_map[tag] = pp
            hook_mgr.set_per_item(pp_map, alpha)
            stats = score_dist_at_c0(model, input_ids, digit_token_ids_per_digit)
            confs_e.append(stats["e_first_digit"])
            entropies.append(stats["H_digit"])
            masses.append(stats["digit_mass"])
            per_item_records.append({
                "seed": seed, "alpha": alpha, "idx": it["idx"],
                "e_first_digit": stats["e_first_digit"],
                "digit_entropy": stats["H_digit"],
                "digit_mass": stats["digit_mass"],
                "per_digit_p": stats["per_digit_p"],
            })
        per_alpha[alpha] = {
            "e_first_digit_mean": float(np.mean(confs_e)),
            "e_first_digit_std":  float(np.std(confs_e)),
            "H_digit_mean":       float(np.mean(entropies)),
            "digit_mass_mean":    float(np.mean(masses)),
            "n":                  len(confs_e),
        }
        print(f"[joint/seed{seed}] alpha={alpha:+.1f}  E[first]={per_alpha[alpha]['e_first_digit_mean']:.3f}  H_digit={per_alpha[alpha]['H_digit_mean']:.3f}  digit_mass={per_alpha[alpha]['digit_mass_mean']:.3f}  elapsed={(time.time()-t0)/60:.1f}m", flush=True)

    hook_mgr.remove()

    baseline = per_alpha.get(0.0, {}).get("e_first_digit_mean", 0.0)
    e_pos = per_alpha.get(max(alphas), {}).get("e_first_digit_mean", 0.0)
    e_neg = per_alpha.get(min(alphas), {}).get("e_first_digit_mean", 0.0)
    span = e_pos - e_neg

    summary = {
        "sites": site_list,
        "seed": seed,
        "n_eval": len(eval_items),
        "alphas": alphas,
        "per_alpha": {str(a): per_alpha[a] for a in alphas},
        "e_first_digit_at_alpha_0": baseline,
        "e_first_digit_at_alpha_pos_max": e_pos,
        "e_first_digit_at_alpha_neg_max": e_neg,
        "e_first_digit_span_full_range": span,
    }
    print(f"[joint/seed{seed}] SUMMARY joint {site_list}: baseline={baseline:.3f}  span(neg..pos)={span:.3f}", flush=True)

    out_path = Path(out_dir) / f"m5v3_joint_expected_score_seed{seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump({"summary": summary, "records": per_item_records}, f, indent=2)
    print(f"[joint/seed{seed}] wrote {out_path}", flush=True)


def main():
    args = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    site_list = [s.strip() for s in args.sites.split(",") if s.strip()]
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print(f"[joint] Loading model...", flush=True)
    t0 = time.time()
    model, tok = load_model_and_tokenizer(MODEL_PATH, dtype=dtype)
    device = next(model.parameters()).device
    print(f"[joint] Model on {device} in {(time.time()-t0)/60:.1f}m", flush=True)

    for seed in seeds:
        run_seed(model, tok, device, args.m2_dir, args.m1_cache, seed, alphas,
                 args.n_items, args.direction_train_split, args.out_dir, args.template,
                 site_list)


if __name__ == "__main__":
    main()
