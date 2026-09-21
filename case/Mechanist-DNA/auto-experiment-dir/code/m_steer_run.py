"""
Round-2 steering run (serves BOTH M2 dose-response and M3 specificity arms).

One run == one (feature_kind, c_sigma, seed). Steers the chosen feature arm at dose c (sigma_proj
units), generates N sequences with Evo2 (frozen -- no predictor involved), then folds the SAME
sequences with BOTH predictors (ESMFold + OmegaFold) and reads out the pLDDT-weighted (primary),
unweighted, and pLDDT-threshold-swept helix+sheet fractions.

sigma_proj-unit dosing detail (norm-matching by construction): delta = c * sigma_proj * unit_dir(arm),
so the injection-site perturbation NORM is c*sigma_proj for EVERY arm/direction at a given c. Arms
therefore differ only in DIRECTION at matched magnitude -- exactly the specificity contrast, and the
random-direction null (m3_random2.py) is norm-matched to S automatically. sigma_proj is S's, loaded
from m1_calibration.json (single shared dose unit).

feature_kind:
  alpha_helix_S       -> frozen helix set S            (M2 uses this; M3 primary arm)
  matched_control     -> frozen 19 matched-control     (M3 secondary sanity control)
  beta_sheet_offtarget-> beta_features_v2 (else round-1 beta)  (M3 beta arm)

Run: CUDA_VISIBLE_DEVICES=3 python code/m_steer_run.py --feature_kind alpha_helix_S \
        --c_sigma 2.0 --seed 42 --n_per_dose 300 --predictors esmfold,omegafold \
        --calib results/m1_calibration.json --out results/m2_cS_alpha_helix_S_c2.0_s42.json
"""
import os, sys, json, argparse, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import m0_data as D
from evo2_sae import load_evo2, BatchTopKSAE
import harness2 as H
from m1_harness_calibrate import natural_prompts


def pick_arm(fs, kind):
    if kind == "alpha_helix_S":
        return fs["helix_features"], fs["s_f"]
    if kind == "matched_control":
        return fs["matched_control_features"], fs["matched_control_s_f"]
    if kind == "beta_sheet_offtarget":
        bv2 = fs.get("beta_v2") or {}
        if bv2.get("beta_features_v2") and bv2.get("clears_bar_stronger_than_round1"):
            return bv2["beta_features_v2"], bv2["beta_v2_s_f"]
        return fs["beta_features"], fs["beta_s_f"]
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_set", default="results/m0_feature_set.json")
    ap.add_argument("--calib", default="results/m1_calibration.json")
    ap.add_argument("--feature_kind", required=True,
                    choices=["alpha_helix_S", "matched_control", "beta_sheet_offtarget"])
    ap.add_argument("--c_sigma", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_per_dose", type=int, default=300)
    ap.add_argument("--n_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--predictors", default="esmfold,omegafold")
    ap.add_argument("--num_cycle", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    predictors = [p for p in args.predictors.split(",") if p]
    t0 = time.time()
    fs = json.load(open(args.feature_set))
    org = fs.get("steer_organism", fs.get("organism_primary", "prokaryote"))
    table = D.ORGANISMS[org]["transl_table"]
    feats, s_f = pick_arm(fs, args.feature_kind)

    calib = json.load(open(args.calib)) if os.path.exists(args.calib) else {}
    sigma_proj = calib.get("sigma_proj")   # S's sigma_proj -- shared dose unit
    if sigma_proj is None:
        raise SystemExit(f"[m_steer] missing sigma_proj in {args.calib}; run M1 first")

    evo2 = load_evo2("cuda:0")
    sae = BatchTopKSAE(device="cuda:0")
    steerer = H.CSteerer(evo2, sae, feats, s_f, sigma_proj=sigma_proj, device="cuda:0")

    prompts, _, _ = natural_prompts(org, args.n_per_dose)
    n_unique = max(len(prompts), 1)
    prompts = (prompts * (args.n_per_dose // n_unique + 1))[:args.n_per_dose]

    print(f"[m_steer] kind={args.feature_kind} c={args.c_sigma} seed={args.seed} "
          f"n={len(prompts)} |feats|={len(feats)} sigma_proj={sigma_proj:.4f}", flush=True)

    # PHASE 1: Evo2 steered generation (frozen -- no predictor)
    recs = H.generate_proteins(evo2, steerer, prompts, args.c_sigma, args.n_tokens,
                               args.temperature, args.top_k,
                               seed_base=args.seed * 100000 + 7, table=table)
    for i, r in enumerate(recs):
        r["prompt_cluster"] = i % n_unique
    n_valid = sum(r.get("valid_orf", False) for r in recs)
    print(f"[m_steer] generated valid_orf={n_valid}/{len(recs)} impact={recs[0].get('impact_ratio')} "
          f"({time.time()-t0:.0f}s)", flush=True)
    del evo2
    torch.cuda.empty_cache()

    # PHASE 2: fold SAME sequences with BOTH predictors
    per_predictor = {}
    for pred in predictors:
        H.fold_records(recs, pred, device="cuda:0", num_cycle=args.num_cycle)
        agg = H.aggregate_predictor(recs, pred, key="helix_hgi_w")
        per_predictor[pred] = agg
        print(f"[m_steer] {pred} helix_w={agg.get('helix_hgi_w_mean')} sheet_w={agg.get('sheet_w_mean')} "
              f"valid_orf={agg.get('valid_orf_rate'):.2f} folded={agg.get('n_folded')} "
              f"plddt={agg.get('struct_mean_plddt_mean')} ({time.time()-t0:.0f}s)", flush=True)

    result = {
        "config": {"feature_kind": args.feature_kind, "c_sigma": args.c_sigma, "seed": args.seed,
                   "n_per_dose": args.n_per_dose, "organism": org, "n_features": len(feats),
                   "sigma_proj": sigma_proj, "predictors": predictors, "num_cycle": args.num_cycle,
                   "injection_perturbation_norm": args.c_sigma * sigma_proj},
        "arm_features": [int(x) for x in feats],
        "per_predictor": per_predictor,
        "elapsed_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[m_steer] WROTE {args.out} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    import os as _os; sys.stdout.flush(); _os._exit(0)
