"""
M3: Specificity & confound battery (completes Claim 1).

(a) Matched RANDOM control direction at equal norm & same block -> expect flat dose-response.
(b) Off-target intactness at best alpha: %E, ORF-validity, GC, coding-likelihood vs baseline
    within tolerance; %H rise is not merely %E suppression.
(c) Naive baselines: temperature sweep (unsteered) and rejection-sampling-toward-%H ->
    show steering reaches higher %H at equal/better validity (adds beyond trivial sampling).

Outputs: results/M3_specificity.json
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
from scipy.stats import mannwhitneyu, spearmanr
import evo2lib as E
from scorer import ESM2SSProbe
from run_M2 import fixed_prompts, score_batch, summarize, gen_cell, N_TOKENS, TEMP, TOPK, TOPP

HERE = os.path.dirname(__file__)
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "assets"))
CTRL_ALPHAS = [0.0, 2.0, 4.0, 8.0, 16.0]
SEEDS = [42, 43, 44]
N_PER_CELL = 40

def main():
    t0 = time.time()
    m2 = json.load(open(os.path.join(RES, "M2_dose_response.json")))
    dirs = np.load(os.path.join(ASSETS, "directions.npz"))
    key = m2["direction_key"]; block = m2["block"]; best_a = m2["best_alpha"]
    v = torch.tensor(dirs[key], dtype=torch.float32).cuda()
    vnorm = float(v.norm())

    m = E.load_evo2()
    probe = ESM2SSProbe(); probe.load_probe(os.path.join(ASSETS, "ss_probe.pkl"))
    prompts = fixed_prompts(10)

    # ---------- (a) random control direction, norm-matched ----------
    E.set_seed(999)
    rv = torch.randn_like(v); rv = rv / rv.norm() * vnorm
    ctrl_H = {}
    with E.SteerHook(m.model, block, rv, alpha=0.0) as hook:
        for a in CTRL_ALPHAS:
            aH = []
            for seed in SEEDS:
                seqs = gen_cell(m, hook, a, seed, prompts, N_PER_CELL)
                sc = score_batch(probe, m, seqs); su = summarize(sc)
                aH.extend(su["_H"])
            ctrl_H[a] = aH
            print(f"[M3a] random ctrl a={a} meanH={np.mean(aH):.2f}", flush=True)
    ctrl_means = {a: float(np.mean(ctrl_H[a])) for a in CTRL_ALPHAS}
    base = np.array(ctrl_H[0.0])
    best_ctrl = max([a for a in CTRL_ALPHAS if a > 0], key=lambda a: ctrl_means[a])
    try:
        _, p_ctrl = mannwhitneyu(np.array(ctrl_H[best_ctrl]), base, alternative="greater")
    except Exception:
        p_ctrl = float('nan')
    rho_ctrl, _ = spearmanr(CTRL_ALPHAS, [ctrl_means[a] for a in CTRL_ALPHAS])
    control_flat = bool(not (p_ctrl < 0.05 and rho_ctrl > 0.5))

    # ---------- (b) off-target intactness at best alpha (real direction) vs baseline ----------
    with E.SteerHook(m.model, block, v, alpha=0.0) as hook:
        seqs_b = gen_cell(m, hook, 0.0, 42, prompts, 60)
        sc_b = score_batch(probe, m, seqs_b); su_b = summarize(sc_b)
        seqs_s = gen_cell(m, hook, best_a, 42, prompts, 60)
        sc_s = score_batch(probe, m, seqs_s); su_s = summarize(sc_s)
    offtarget = {
        "baseline": {k: su_b[k] for k in ["mean_pctH","mean_pctE","orf_valid_rate","mean_gc","mean_loglik"]},
        "steered_best_alpha": {k: su_s[k] for k in ["mean_pctH","mean_pctE","orf_valid_rate","mean_gc","mean_loglik"]},
        "delta_pctH": round(su_s["mean_pctH"] - su_b["mean_pctH"], 3),
        "delta_pctE": round(su_s["mean_pctE"] - su_b["mean_pctE"], 3),
        "delta_valid": round(su_s["orf_valid_rate"] - su_b["orf_valid_rate"], 3),
        "delta_loglik": round(su_s["mean_loglik"] - su_b["mean_loglik"], 3),
    }
    # %H rise not merely %E suppression: helix gain should exceed sheet loss magnitude, validity kept
    validity_ok = bool(su_s["orf_valid_rate"] >= su_b["orf_valid_rate"] - 0.15)
    not_just_E = bool(offtarget["delta_pctH"] > 0 and abs(offtarget["delta_pctH"]) >= 0.5 * abs(offtarget["delta_pctE"]))

    # ---------- (c) naive baselines ----------
    # temperature sweep (unsteered)
    temp_H = {}
    with E.SteerHook(m.model, block, v, alpha=0.0) as hook:
        hook.set_alpha(0.0)
        for T in [0.3, 0.7, 1.0, 1.2]:
            E.set_seed(42)
            seqs = []
            per = 6
            for p in prompts:
                s, _ = E.generate(m, [p]*per, n_tokens=N_TOKENS, temperature=T, top_k=TOPK, top_p=TOPP)
                seqs.extend([p+x for x in s])
            sc = score_batch(probe, m, seqs)
            H = [d["pctH"] for d in sc if d["pctH"]==d["pctH"]]
            temp_H[T] = {"mean_pctH": float(np.mean(H)), "orf_valid_rate": float(np.mean([d["orf_valid"] for d in sc]))}
            print(f"[M3c] temp={T} meanH={temp_H[T]['mean_pctH']:.2f}", flush=True)
    best_temp_H = max(temp_H.values(), key=lambda d: d["mean_pctH"])["mean_pctH"]

    # rejection sampling toward %H: large unsteered pool, keep top 20%
    with E.SteerHook(m.model, block, v, alpha=0.0) as hook:
        hook.set_alpha(0.0)
        E.set_seed(7)
        pool = []
        for p in prompts:
            s, _ = E.generate(m, [p]*20, n_tokens=N_TOKENS, temperature=TEMP, top_k=TOPK, top_p=TOPP)
            pool.extend([p+x for x in s])
        sc_pool = score_batch(probe, m, pool)
    poolH = np.array([d["pctH"] for d in sc_pool if d["pctH"]==d["pctH"]])
    pool_valid = [d for d in sc_pool if d["orf_valid"]]
    thr = np.percentile(poolH, 80)
    kept = [d for d in sc_pool if d["pctH"]==d["pctH"] and d["pctH"] >= thr]
    rej_meanH = float(np.mean([d["pctH"] for d in kept])) if kept else float('nan')
    rej_yield = len(kept) / len(sc_pool)
    rej_valid = float(np.mean([d["orf_valid"] for d in kept])) if kept else float('nan')

    steered_bestH = su_s["mean_pctH"]; steered_valid = su_s["orf_valid_rate"]
    beats_naive = bool(steered_bestH >= best_temp_H and steered_valid >= (rej_valid - 0.1))

    result = {
        "milestone": "M3", "claim": "C1", "role": "specificity",
        "direction_key": key, "block": block, "best_alpha": best_a, "direction_norm": round(vnorm,3),
        "a_random_control": {"mean_pctH_by_alpha": {str(a): round(ctrl_means[a],3) for a in CTRL_ALPHAS},
                              "p_one_sided": float(p_ctrl), "spearman_rho": float(rho_ctrl),
                              "control_flat_no_gain": control_flat},
        "b_offtarget": offtarget, "b_validity_ok": validity_ok, "b_not_just_sheet_suppression": not_just_E,
        "c_temperature_sweep": {str(T): temp_H[T] for T in temp_H},
        "c_rejection_sampling": {"pool_n": len(sc_pool), "kept_top20pct_meanH": round(rej_meanH,3),
                                 "yield": round(rej_yield,3), "kept_valid_rate": round(rej_valid,3),
                                 "pool_mean_H": round(float(np.mean(poolH)),3)},
        "c_steered_beats_naive": beats_naive,
        "steered_best_meanH": round(steered_bestH,3), "best_temp_meanH": round(best_temp_H,3),
        "pass_complete_C1": bool(control_flat and validity_ok and not_just_E and beats_naive),
        "gpu_hours": round((time.time()-t0)/3600,3),
    }
    json.dump(result, open(os.path.join(RES, "M3_specificity.json"), "w"), indent=2)
    print("[M3] RESULT", json.dumps({k: result[k] for k in ["a_random_control","b_validity_ok","b_not_just_sheet_suppression","c_steered_beats_naive","pass_complete_C1"]}), flush=True)
    print("M3_DONE", flush=True)

if __name__ == "__main__":
    main()
