"""
M2 (refinement): multi-block steering dose-response.

Single-site probe steering gave a weak/non-monotonic dose-response (decodability != causality).
Per steering-block-selection (widen to a set of sites) and steer-a-set general rule, this
steers the probe alpha-helix directions at a SET of blocks simultaneously with a shared
coefficient alpha, and re-runs the dose-response. This tunes the method_sensitive `sites` knob.

Outputs: results/M2_dose_response.json (SUPERSEDES the single-block version),
         results/M2_singleblock_dose_response.json (archived single-block result).
"""
import os, sys, json, time, shutil
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from scipy.stats import mannwhitneyu, spearmanr
import evo2lib as E
from scorer import ESM2SSProbe
from run_M2 import fixed_prompts, score_batch, summarize, N_TOKENS, TEMP, TOPK, TOPP

HERE = os.path.dirname(__file__)
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "assets"))
STEER_BLOCKS = [18, 20, 22, 24, 26]
ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
SEEDS = [42, 43, 44]
N_PER_CELL = 40

class MultiSteer:
    """Register additive probe-direction hooks at a set of blocks with a shared alpha."""
    def __init__(self, model, block_vecs):
        self.model = model; self.block_vecs = block_vecs
        self.state = {"alpha": 0.0}; self.handles = []
    def _mk(self, v):
        def hook(mod, inp, out):
            a = self.state["alpha"]
            if a == 0.0: return out
            if isinstance(out, tuple):
                hs = out[0] + a * v.to(out[0].dtype); return (hs,) + tuple(out[1:])
            return out + a * v.to(out.dtype)
        return hook
    def __enter__(self):
        for b, v in self.block_vecs.items():
            blk = self.model.get_submodule(E.block_name(b))
            self.handles.append(blk.register_forward_hook(self._mk(v)))
        return self
    def set_alpha(self, a): self.state["alpha"] = a
    def __exit__(self, *a):
        for h in self.handles: h.remove()

def gen_cell(m, hook, alpha, seed, prompts, npc):
    E.set_seed(seed); hook.set_alpha(alpha)
    seqs = []; per = max(1, npc // len(prompts))
    for p in prompts:
        s, _ = E.generate(m, [p]*per, n_tokens=N_TOKENS, temperature=TEMP, top_k=TOPK, top_p=TOPP)
        seqs.extend([p + x for x in s])
    return seqs

def main():
    t0 = time.time()
    # archive single-block result
    src = os.path.join(RES, "M2_dose_response.json")
    if os.path.exists(src):
        shutil.copy(src, os.path.join(RES, "M2_singleblock_dose_response.json"))
    dirs = np.load(os.path.join(ASSETS, "directions.npz"))
    block_vecs = {b: torch.tensor(dirs[f"probe_b{b}"], dtype=torch.float32).cuda() for b in STEER_BLOCKS}
    print(f"[M2mb] steering blocks {STEER_BLOCKS}", {b: round(float(v.norm()),2) for b,v in block_vecs.items()}, flush=True)

    m = E.load_evo2()
    probe = ESM2SSProbe(); probe.load_probe(os.path.join(ASSETS, "ss_probe.pkl"))
    prompts = fixed_prompts(10)

    alphas = list(ALPHAS); per_alpha_H = {}; cells = {}; all_gens = {}
    with MultiSteer(m.model, block_vecs) as hook:
        ai = 0
        while ai < len(alphas):
            a = alphas[ai]; aH = []
            for seed in SEEDS:
                seqs = gen_cell(m, hook, a, seed, prompts, N_PER_CELL)
                sc = score_batch(probe, m, seqs); su = summarize(sc)
                cells[f"a{a}_s{seed}"] = {k: v for k, v in su.items() if k != "_H"}
                aH.extend(su["_H"]); all_gens[f"a{a}_s{seed}"] = seqs
                json.dump({"alpha": a, "seed": seed, "blocks": STEER_BLOCKS,
                           "summary": {k: v for k, v in su.items() if k != "_H"}},
                          open(os.path.join(RES, f"M2_a{a}_s{seed}.json"), "w"), indent=2)
                print(f"[M2mb] a={a} s={seed} meanH={su['mean_pctH']:.2f} valid={su['orf_valid_rate']:.2f} ll={su['mean_loglik']:.3f}", flush=True)
            per_alpha_H[a] = aH
            if ai == len(alphas)-1 and a <= 8.0:
                means = [np.mean(per_alpha_H[x]) for x in alphas]
                if means[-1] > means[-2] and a < 32.0:
                    alphas.append(min(32.0, a*2))
            ai += 1

    base = np.array(per_alpha_H[0.0])
    means = {a: float(np.mean(per_alpha_H[a])) for a in alphas}
    best_a = max([a for a in alphas if a > 0], key=lambda a: means[a])
    bestH = np.array(per_alpha_H[best_a])
    try: _, p_one = mannwhitneyu(bestH, base, alternative="greater")
    except Exception: p_one = float('nan')
    n_comp = len([a for a in alphas if a > 0])
    p_bonf = min(1.0, p_one*n_comp) if p_one == p_one else float('nan')
    xs = sorted(alphas); rho, _ = spearmanr(xs, [means[a] for a in xs])

    result = {
        "milestone": "M2", "claim": "C1", "role": "causal_dose_response",
        "steering_mode": "multi_block", "direction_key": "probe_multiblock",
        "blocks": STEER_BLOCKS, "block": STEER_BLOCKS[-1],
        "alphas": alphas, "seeds": SEEDS, "n_per_alpha": len(per_alpha_H[0.0]),
        "mean_pctH_by_alpha": {str(a): round(means[a], 3) for a in alphas},
        "cells": cells,
        "baseline_mean_pctH": round(means[0.0], 3),
        "best_alpha": best_a, "best_mean_pctH": round(means[best_a], 3),
        "delta_H_best_vs_baseline": round(means[best_a]-means[0.0], 3),
        "p_one_sided_best_gt_baseline": float(p_one), "p_bonferroni": float(p_bonf),
        "spearman_rho_H_vs_alpha": float(rho), "monotonic_increasing": bool(rho > 0.5),
        "pass_partial_C1": bool((means[best_a] > means[0.0]) and (p_bonf < 0.05) and (rho > 0.5)),
        "gpu_hours": round((time.time()-t0)/3600, 3),
        "superseded_note": "multi-block refinement of single-block M2 (archived in M2_singleblock_dose_response.json)",
    }
    json.dump(result, open(os.path.join(RES, "M2_dose_response.json"), "w"), indent=2)
    json.dump(all_gens, open(os.path.join(RES, "M2_generations.json"), "w"))
    print("[M2mb] RESULT", json.dumps({k: result[k] for k in ["mean_pctH_by_alpha","best_alpha","delta_H_best_vs_baseline","p_bonferroni","spearman_rho_H_vs_alpha","pass_partial_C1"]}), flush=True)
    print("M2_DONE", flush=True)

if __name__ == "__main__":
    main()
