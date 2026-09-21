#!/usr/bin/env python3
"""
M4 / M4.stab — Role dissociation (C2 modular decomposition + stability).

For each role-corruption pool (fact-swap, rule-swap, answer-swap):
  For each shortlisted component:
    Run activation patching restore of that ONE component (clean-into-corrupt),
    measuring Recovery_r(c).

Assemble the role-assignment matrix S ∈ [0,1]^{|C|×3}.

Modularity metrics:
  - dominance_ratio(c) = max_r S[c,r] / (2nd_max_r S[c,r])
  - dissociation(r) = mean_{c ∈ C_r} S[c,r] - mean_{c ∈ C_r} S[c,r'≠r]
  - null shuffle: 100 label permutations to get p-value.

For M4.stab: run on multiple cells (comma-separated), report per-cell block
partitions + pairwise Jaccard.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prop_circuit_lib import (
    load_model,
    load_jsonl,
    get_answer_token_ids,
    run_activation_patch,
    measure_kl_baseline,
    set_global_seeds,
)


def load_shortlist(m1_json_path: str):
    with open(m1_json_path) as fh:
        m1 = json.load(fh)
    shortlist = []
    for h in m1["shortlist_heads"]:
        shortlist.append(("attn", int(h["layer"]), int(h["head"])))
    for m in m1["shortlist_mlps"]:
        shortlist.append(("mlp", int(m["layer"]), None))
    return shortlist, m1


def _clip01(x):
    if x != x:  # NaN
        return 0.0
    return max(0.0, min(1.0, x))


def compute_S_matrix(
    model, tok, data_dir, cell, shortlist, roles, n_pairs, batch_size, max_length,
    per_component_cap: int = None,
):
    """
    Returns S: dict[role -> list[float]] of length len(shortlist), giving per-component
    recovery under that role's corruption pool.
    """
    true_id, false_id = get_answer_token_ids(tok)
    clean_path = Path(data_dir) / "clean" / f"split_{cell}" / "data.jsonl"
    clean = load_jsonl(clean_path)[:n_pairs]

    S = {r: [] for r in roles}
    for role in roles:
        corr_path = Path(data_dir) / f"corrupt_{role}" / f"split_{cell}" / "data.jsonl"
        corr = load_jsonl(corr_path)[:n_pairs]
        n = min(len(clean), len(corr))
        clean_sub = clean[:n]; corr_sub = corr[:n]
        print(f"[M4] cell={cell} role={role}: n={n} pairs, sweeping {len(shortlist)} components")
        for i, comp in enumerate(shortlist):
            t0 = time.time()
            r = run_activation_patch(
                model, clean_sub, corr_sub, [comp], true_id, false_id,
                batch_size=batch_size, max_length=max_length,
            )
            recovery = _clip01(r["recovery_logit_diff"])
            S[role].append(recovery)
            if (i + 1) % 5 == 0 or i == len(shortlist) - 1:
                print(f"[M4]   {i+1}/{len(shortlist)} components done "
                      f"(last: role={role} comp={comp} rec={recovery:.3f}, "
                      f"{time.time() - t0:.1f}s)")
            if per_component_cap is not None and (i + 1) >= per_component_cap:
                # Fill rest with 0 to keep matrix shape aligned.
                for j in range(i + 1, len(shortlist)):
                    S[role].append(0.0)
                break
    return S


def compute_modularity(S: Dict[str, List[float]], roles: List[str]):
    """
    Return per-component dominance ratio, dissociation d per role, and block partition.
    """
    import numpy as np
    n = len(next(iter(S.values())))
    M = np.array([[S[r][i] for r in roles] for i in range(n)])  # (n, 3)
    # Assign each component to the role with the max recovery.
    partition = {r: [] for r in roles}
    dominance = []
    for i in range(n):
        row = M[i]
        top = np.argsort(row)[::-1]
        max_r = roles[top[0]]
        max_v = row[top[0]]
        second_v = row[top[1]] if len(top) > 1 else 0.0
        ratio = max_v / max(second_v, 1e-6)
        dominance.append(float(ratio))
        partition[max_r].append(i)
    dissociation = {}
    for r in roles:
        cs = partition[r]
        if not cs:
            dissociation[r] = float("nan")
            continue
        in_role = np.mean([M[i, roles.index(r)] for i in cs])
        off_role = np.mean([np.mean([M[i, roles.index(rr)] for rr in roles if rr != r]) for i in cs])
        dissociation[r] = float(in_role - off_role)
    return {
        "dominance_ratios": dominance,
        "median_dominance_ratio": float(np.median(dominance)),
        "dissociation": dissociation,
        "block_partition": partition,
        "S_matrix": M.tolist(),
    }


def null_shuffle_pvalue(S: Dict[str, List[float]], roles: List[str], n_shuffles: int = 100, seed: int = 0):
    """Shuffle the role labels (per component) and re-compute dissociation. Return per-role p-value."""
    import numpy as np
    rng = np.random.default_rng(seed)
    observed = compute_modularity(S, roles)["dissociation"]
    null_dist = {r: [] for r in roles}
    n_comp = len(next(iter(S.values())))
    for _ in range(n_shuffles):
        S_shuf = {}
        for i in range(n_comp):
            vals = [S[r][i] for r in roles]
            perm = rng.permutation(len(vals))
            for idx, r in enumerate(roles):
                S_shuf.setdefault(r, []).append(vals[perm[idx]])
        mod = compute_modularity(S_shuf, roles)
        for r in roles:
            null_dist[r].append(mod["dissociation"][r])
    p_values = {}
    for r in roles:
        obs = observed[r]
        if obs != obs:
            p_values[r] = float("nan")
            continue
        # p = P(null >= observed)
        p_values[r] = float((np.array(null_dist[r]) >= obs).mean())
    return p_values, null_dist


def jaccard(a: List[int], b: List[int]) -> float:
    sa = set(a); sb = set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(len(sa | sb), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--cell", default="k3_chain2_natural", help="Comma-separated cells (for M4.stab)")
    ap.add_argument("--roles", default="fact,rule,answer")
    ap.add_argument("--shortlist", required=True)
    ap.add_argument("--n-pairs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-component-cap", type=int, default=None,
                    help="Optional cap on how many shortlist components to sweep (dev/debug)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    set_global_seeds(args.seed)

    roles = [r.strip() for r in args.roles.split(",")]
    cells = [c.strip() for c in args.cell.split(",")]

    print(f"[M4] cuda visible: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not-set')}")
    t0 = time.time()
    try:
        model, tok = load_model(args.model, dtype=torch.bfloat16, device="cuda", n_devices=1)
    except Exception as e:
        alt = args.model.replace("Mistral-7B-v0.1", "Mistral-7B-Instruct-v0.1")
        if alt == args.model:
            raise
        print(f"[M4] primary load failed ({e}); trying Instruct fallback {alt}")
        model, tok = load_model(alt, dtype=torch.bfloat16, device="cuda", n_devices=1)
    t_load = time.time() - t0
    print(f"[M4] model loaded in {t_load:.1f}s")

    shortlist, m1 = load_shortlist(args.shortlist)
    print(f"[M4] shortlist size = {len(shortlist)}, roles = {roles}, cells = {cells}")

    all_results = {}
    for cell in cells:
        print(f"\n[M4] ==== cell {cell} ====")
        t1 = time.time()
        S = compute_S_matrix(
            model, tok, args.data, cell, shortlist, roles,
            n_pairs=args.n_pairs, batch_size=args.batch_size,
            max_length=args.max_length,
            per_component_cap=args.per_component_cap,
        )
        modularity = compute_modularity(S, roles)
        p_values, null_dist = null_shuffle_pvalue(S, roles, n_shuffles=100, seed=42)
        modularity["null_shuffle_p_value"] = p_values
        modularity["cell_time_s"] = time.time() - t1
        all_results[cell] = modularity
        print(f"[M4] cell={cell}: median_dominance={modularity['median_dominance_ratio']:.2f}, "
              f"dissociation={modularity['dissociation']}, p={p_values}, "
              f"({modularity['cell_time_s']/60:.1f} min)")

    # Stability across cells (Jaccard).
    stability = {}
    if len(cells) >= 2:
        for r in roles:
            js = []
            for i in range(len(cells)):
                for j in range(i + 1, len(cells)):
                    ci, cj = cells[i], cells[j]
                    js.append({
                        "cell_a": ci, "cell_b": cj,
                        "jaccard": jaccard(all_results[ci]["block_partition"][r],
                                            all_results[cj]["block_partition"][r]),
                    })
            stability[r] = {
                "pairwise": js,
                "mean_jaccard": float(sum(x["jaccard"] for x in js) / len(js)) if js else float("nan"),
            }
        print(f"[M4] stability across cells: "
              + ", ".join(f"{r}={stability[r]['mean_jaccard']:.2f}" for r in roles))

    # Success criterion for anchor cell.
    anchor = all_results[cells[0]]
    success_c2 = (
        (anchor["median_dominance_ratio"] >= 2.0)
        and all(anchor["dissociation"][r] >= 0.1 for r in roles)
        and all(anchor["null_shuffle_p_value"][r] <= 0.01 for r in roles)
    )

    out = {
        "model": args.model,
        "cells": cells,
        "roles": roles,
        "n_pairs": args.n_pairs,
        "shortlist_size": len(shortlist),
        "per_cell": all_results,
        "stability": stability,
        "success_C2": success_c2,
        "success_C2_stability": all(v["mean_jaccard"] >= 0.6 for v in stability.values()) if stability else None,
        "timing": {"load_s": t_load, "total_s": time.time() - t0},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[M4] wrote {args.out} (success_C2={success_c2})")


if __name__ == "__main__":
    main()
