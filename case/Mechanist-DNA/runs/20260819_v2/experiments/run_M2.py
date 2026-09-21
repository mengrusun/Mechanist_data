"""
M2: Causal steering dose-response (Causal Intervention, core of Claim 1).

Promotes the top-ranked M1 direction to a causal knob: adds alpha*v to the residual
stream at its block during autoregressive generation and measures %H vs alpha.

Grid: alpha in [0,0.5,1,2,4,8] (escalates to 16 if still monotone-rising at 8),
seeds [42,43,44]; N generations per (alpha,seed). Same fixed prompts / decoding across
conditions. Scores every generation with the E1 harness (fast %H) + general-ability
covariates (ORF validity, %E, GC, Evo2 coding-likelihood).

Outputs: results/M2_dose_response.json, results/M2_a{alpha}_s{seed}.json, and
saves per-generation sequences to results/M2_generations.json for M3/M4 reuse.
"""
import os, sys, json, time, argparse
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from scipy.stats import mannwhitneyu, spearmanr
import evo2lib as E
from scorer import ESM2SSProbe

HERE = os.path.dirname(__file__)
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "assets"))

ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
SEEDS = [42, 43, 44]
N_PER_CELL = 40           # per (alpha, seed) -> 120 per alpha (>= plan floor of 100/condition)
N_TOKENS = 300
TEMP, TOPK, TOPP = 0.7, 4, 1.0

def fixed_prompts(n=10):
    """Fixed coding-start prompts (same across all conditions). Derived from contrastive CDS starts."""
    w = json.load(open(os.path.join(DATA, "contrastive.json")))
    prompts = []
    for x in w[:200]:
        p = x["dna"][:30]
        if len(p) == 30 and all(c in "ACGT" for c in p):
            prompts.append(p)
        if len(prompts) >= n: break
    if not prompts:
        prompts = ["ATGGCAGAACTGAAACGTATTGCAGAAGCA"[:30]]
    return prompts[:n]

def score_batch(probe, m, seqs):
    """Return list of dicts with %H, %E, orf_valid, len, gc, loglik per sequence."""
    out = []
    prots = []
    for s in seqs:
        valid = E.is_valid_dna(s)
        prot, frame, start, orf = E.find_longest_orf(s, min_aa=15) if valid else ("", -1, -1, "")
        prots.append(prot)
        out.append({"orf_valid": bool(valid and len(prot) >= 15),
                    "prot_len": len(prot), "gc": round(E.gc_content(s), 4),
                    "dna_len": len(s)})
    # coding-likelihood (batchwise)
    try:
        lls = E.coding_loglik(m, seqs)
    except Exception:
        lls = [float('nan')] * len(seqs)
    for i, prot in enumerate(prots):
        out[i]["loglik"] = lls[i]
        if len(prot) >= 15:
            out[i]["pctH"] = round(probe.pct_helix(prot), 3)
            out[i]["pctE"] = round(probe.pct_sheet(prot), 3)
        else:
            out[i]["pctH"] = float('nan'); out[i]["pctE"] = float('nan')
    return out

def gen_cell(m, hook, alpha, seed, prompts, npc):
    E.set_seed(seed)
    hook.set_alpha(alpha)
    seqs = []
    per_prompt = max(1, npc // len(prompts))
    for p in prompts:
        batch = [p] * per_prompt
        s, _ = E.generate(m, batch, n_tokens=N_TOKENS, temperature=TEMP, top_k=TOPK, top_p=TOPP)
        # full generated coding sequence = prompt + continuation
        seqs.extend([p + x for x in s])
    return seqs

def summarize(scores):
    H = np.array([d["pctH"] for d in scores if d["pctH"] == d["pctH"]])
    Evals = np.array([d["pctE"] for d in scores if d["pctE"] == d["pctE"]])
    valid = np.mean([d["orf_valid"] for d in scores])
    gc = np.mean([d["gc"] for d in scores])
    ll = np.array([d["loglik"] for d in scores if d["loglik"] == d["loglik"]])
    return {"n": len(scores), "mean_pctH": float(np.mean(H)) if len(H) else float('nan'),
            "std_pctH": float(np.std(H)) if len(H) else float('nan'),
            "mean_pctE": float(np.mean(Evals)) if len(Evals) else float('nan'),
            "orf_valid_rate": float(valid), "mean_gc": float(gc),
            "mean_loglik": float(np.mean(ll)) if len(ll) else float('nan'),
            "_H": H.tolist()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", default=None, help="direction key in directions.npz; default=best M1 additive candidate")
    args = ap.parse_args()
    t0 = time.time()

    m1 = json.load(open(os.path.join(RES, "M1_candidate_directions.json")))
    dirs = np.load(os.path.join(ASSETS, "directions.npz"))
    # choose best additive (contrastive/probe) candidate
    if args.direction:
        key = args.direction
        cand = next(c for c in m1["ranked_candidates"] if c["key"] == key)
    else:
        add_cands = [c for c in m1["ranked_candidates"] if c["family"] in ("contrastive_vector", "linear_probe")]
        cand = add_cands[0]
        key = cand["key"]
    block = cand["block"]
    v = torch.tensor(dirs[key], dtype=torch.float32).cuda()
    print(f"[M2] steering direction={key} block={block} auroc={cand['auroc_heldout']} |v|={float(v.norm()):.3f}", flush=True)

    m = E.load_evo2()
    probe = ESM2SSProbe(); probe.load_probe(os.path.join(ASSETS, "ss_probe.pkl"))
    prompts = fixed_prompts(10)
    print(f"[M2] {len(prompts)} fixed prompts", flush=True)

    alphas = list(ALPHAS)
    cells = {}; per_alpha_H = {}
    all_gens = {}
    with E.SteerHook(m.model, block, v, alpha=0.0) as hook:
        ai = 0
        while ai < len(alphas):
            a = alphas[ai]
            aH = []
            for seed in SEEDS:
                seqs = gen_cell(m, hook, a, seed, prompts, N_PER_CELL)
                scores = score_batch(probe, m, seqs)
                summ = summarize(scores)
                cells[f"a{a}_s{seed}"] = {k: v2 for k, v2 in summ.items() if k != "_H"}
                aH.extend(summ["_H"])
                all_gens[f"a{a}_s{seed}"] = seqs
                json.dump({"alpha": a, "seed": seed, "direction": key, "block": block,
                           "summary": {k: v2 for k, v2 in summ.items() if k != "_H"}},
                          open(os.path.join(RES, f"M2_a{a}_s{seed}.json"), "w"), indent=2)
                print(f"[M2] a={a} s={seed} meanH={summ['mean_pctH']:.2f} valid={summ['orf_valid_rate']:.2f} ll={summ['mean_loglik']:.3f}", flush=True)
            per_alpha_H[a] = aH
            # escalation: if at max alpha and still monotone rising vs previous, add 16
            if ai == len(alphas) - 1 and a <= 8.0:
                means = [np.mean(per_alpha_H[x]) for x in alphas]
                if len(means) >= 2 and means[-1] > means[-2] and a < 16.0:
                    alphas.append(16.0)
            ai += 1

    # stats
    base = np.array(per_alpha_H[0.0])
    means = {a: float(np.mean(per_alpha_H[a])) for a in alphas}
    best_a = max([a for a in alphas if a > 0], key=lambda a: means[a])
    bestH = np.array(per_alpha_H[best_a])
    try:
        U, p_one = mannwhitneyu(bestH, base, alternative="greater")
    except Exception:
        p_one = float('nan')
    n_comp = len([a for a in alphas if a > 0])
    p_bonf = min(1.0, p_one * n_comp) if p_one == p_one else float('nan')
    xs = sorted(alphas); ys = [means[a] for a in xs]
    rho, _ = spearmanr(xs, ys)

    result = {
        "milestone": "M2", "claim": "C1", "role": "causal_dose_response",
        "direction_key": key, "block": block, "direction_auroc": cand["auroc_heldout"],
        "alphas": alphas, "seeds": SEEDS, "n_per_alpha": len(per_alpha_H[0.0]),
        "mean_pctH_by_alpha": {str(a): round(means[a], 3) for a in alphas},
        "cells": cells,
        "baseline_mean_pctH": round(means[0.0], 3),
        "best_alpha": best_a, "best_mean_pctH": round(means[best_a], 3),
        "delta_H_best_vs_baseline": round(means[best_a] - means[0.0], 3),
        "p_one_sided_best_gt_baseline": float(p_one),
        "p_bonferroni": float(p_bonf),
        "spearman_rho_H_vs_alpha": float(rho),
        "monotonic_increasing": bool(rho > 0.5),
        "pass_partial_C1": bool((means[best_a] > means[0.0]) and (p_bonf < 0.05) and (rho > 0.5)),
        "gpu_hours": round((time.time()-t0)/3600, 3),
    }
    json.dump(result, open(os.path.join(RES, "M2_dose_response.json"), "w"), indent=2)
    json.dump(all_gens, open(os.path.join(RES, "M2_generations.json"), "w"))
    print("[M2] RESULT", json.dumps({k: result[k] for k in ["mean_pctH_by_alpha","best_alpha","delta_H_best_vs_baseline","p_bonferroni","spearman_rho_H_vs_alpha","pass_partial_C1"]}), flush=True)
    print("M2_DONE", flush=True)

if __name__ == "__main__":
    main()
