"""
M4: High-alpha-helix generation at scale (Tuning & Editing capability demo).

Uses the best M2/M3 config (direction, Pareto-optimal alpha keeping validity) to
generate a library; reports %H distribution vs baseline, ORF-validity retention, and
an (optional) ESMFold+DSSP structure-validated subset.

Outputs: results/M4_library.json, results/M4_library_sequences.json
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import evo2lib as E
from scorer import ESM2SSProbe, esmfold_available
from run_M2 import fixed_prompts, score_batch, summarize, gen_cell, N_TOKENS, TEMP, TOPK, TOPP

HERE = os.path.dirname(__file__)
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "assets"))
LIB_N = 200

def main():
    t0 = time.time()
    m2 = json.load(open(os.path.join(RES, "M2_dose_response.json")))
    dirs = np.load(os.path.join(ASSETS, "directions.npz"))
    key = m2["direction_key"]; block = m2["block"]
    v = torch.tensor(dirs[key], dtype=torch.float32).cuda()

    # Pareto-optimal alpha: max mean %H subject to validity >= 0.8 * baseline validity
    cells = m2["cells"]
    base_valid = np.mean([cells[c]["orf_valid_rate"] for c in cells if c.startswith("a0.0_")])
    by_alpha = {}
    for a in m2["alphas"]:
        cs = [cells[c] for c in cells if c.startswith(f"a{a}_")]
        if cs:
            by_alpha[a] = {"H": np.mean([c["mean_pctH"] for c in cs]),
                           "valid": np.mean([c["orf_valid_rate"] for c in cs])}
    ok = {a: d for a, d in by_alpha.items() if a > 0 and d["valid"] >= 0.8*base_valid}
    chosen_a = max(ok, key=lambda a: ok[a]["H"]) if ok else m2["best_alpha"]
    print(f"[M4] chosen alpha={chosen_a} (Pareto valid>=0.8*base={base_valid:.2f})", flush=True)

    m = E.load_evo2()
    probe = ESM2SSProbe(); probe.load_probe(os.path.join(ASSETS, "ss_probe.pkl"))
    prompts = fixed_prompts(20)

    with E.SteerHook(m.model, block, v, alpha=0.0) as hook:
        E.set_seed(2024)
        base_seqs = gen_cell(m, hook, 0.0, 2024, prompts, LIB_N)
        lib_seqs = gen_cell(m, hook, chosen_a, 2024, prompts, LIB_N)
    sc_base = score_batch(probe, m, base_seqs)
    sc_lib = score_batch(probe, m, lib_seqs)
    su_base = summarize(sc_base); su_lib = summarize(sc_lib)
    Hb = np.array([d["pctH"] for d in sc_base if d["pctH"]==d["pctH"]])
    Hl = np.array([d["pctH"] for d in sc_lib if d["pctH"]==d["pctH"]])

    def q(a, p): return round(float(np.percentile(a, p)), 2) if len(a) else float('nan')
    result = {
        "milestone": "M4", "claim": "C1", "role": "capability_demo",
        "direction_key": key, "block": block, "chosen_alpha": chosen_a, "library_n": LIB_N,
        "baseline_pctH": {"mean": round(su_base["mean_pctH"],3), "median": q(Hb,50), "p90": q(Hb,90),
                          "valid_rate": round(su_base["orf_valid_rate"],3), "mean_loglik": round(su_base["mean_loglik"],3)},
        "library_pctH": {"mean": round(su_lib["mean_pctH"],3), "median": q(Hl,50), "p90": q(Hl,90),
                         "valid_rate": round(su_lib["orf_valid_rate"],3), "mean_loglik": round(su_lib["mean_loglik"],3)},
        "uplift_mean_pctH": round(su_lib["mean_pctH"] - su_base["mean_pctH"], 3),
        "validity_retention": round(su_lib["orf_valid_rate"] / max(1e-6, su_base["orf_valid_rate"]), 3),
    }

    # optional structure-validated subset
    if esmfold_available():
        try:
            from scorer import ESMFolder
            # free some memory
            folder = ESMFolder()
            order = np.argsort([-(d["pctH"] if d["pctH"]==d["pctH"] else -1) for d in sc_lib])[:15]
            fastH = []; foldH = []
            for i in order:
                s = lib_seqs[int(i)]
                prot, *_ = E.find_longest_orf(s, min_aa=15)
                if len(prot) < 15: continue
                h, e = folder.pct_helix_sheet(prot)
                if h == h:
                    fastH.append(sc_lib[int(i)]["pctH"]); foldH.append(h)
            if len(foldH) >= 5:
                from scipy.stats import pearsonr
                r = pearsonr(fastH, foldH)[0]
                result["structure_validated_subset"] = {
                    "n": len(foldH), "mean_fast_pctH": round(float(np.mean(fastH)),2),
                    "mean_esmfold_dssp_pctH": round(float(np.mean(foldH)),2),
                    "pearson_r_fast_vs_esmfold": round(float(r),3)}
                print(f"[M4] structure subset n={len(foldH)} fastH={np.mean(fastH):.1f} foldH={np.mean(foldH):.1f} r={r:.2f}", flush=True)
        except Exception as ex:
            result["structure_note"] = f"ESMFold subset skipped: {type(ex).__name__}: {str(ex)[:120]}"
    else:
        result["structure_note"] = "ESMFold unavailable; %H grounded via E1 fast-predictor validation vs experimental DSSP."

    result["gpu_hours"] = round((time.time()-t0)/3600,3)
    json.dump(result, open(os.path.join(RES, "M4_library.json"), "w"), indent=2)
    json.dump({"baseline": base_seqs, "library": lib_seqs, "chosen_alpha": chosen_a},
              open(os.path.join(RES, "M4_library_sequences.json"), "w"))
    print("[M4] RESULT", json.dumps({k: result[k] for k in ["baseline_pctH","library_pctH","uplift_mean_pctH","validity_retention"]}), flush=True)
    print("M4_DONE", flush=True)

if __name__ == "__main__":
    main()
