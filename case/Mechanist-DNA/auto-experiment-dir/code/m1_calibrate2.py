"""
M1 (round-2): calibrate the sigma_proj-unit steering hook + the dual-predictor pLDDT-weighted
readout + power/N-per-dose.

Computes:
  * sigma_proj at blocks.26.post_norm on a natural-CDS reference set + ||base_dir||, and the
    raw-a <-> c map so round-1 raw alphas (0,1,2,4,8,16,32) are locatable on the sigma_proj c-axis.
  * c=0 baseline helix distribution under BOTH predictors (ESMFold + OmegaFold), in the pLDDT-
    WEIGHTED (primary) and hard-gated statistics, with mean pLDDT + valid-ORF rate.
  * positive control: natural helix-rich vs helix-poor reference proteins separate under both.
  * power: samples-per-dose to detect Delta_helix >= 0.1 at power 0.8 from the baseline sd.

Run: CUDA_VISIBLE_DEVICES=3 python code/m1_calibrate2.py --feature_set results/m0_feature_set.json \
        --predictors esmfold,omegafold --n_samples 400 --n_ref 200 --out results/m1_calibration.json
"""
import os, sys, json, argparse, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import m0_data as D
from evo2_sae import load_evo2, BatchTopKSAE
import harness2 as H
from m1_harness_calibrate import natural_prompts

ROUND1_RAW_ALPHAS = [0, 1, 2, 4, 8, 16, 32]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_set", default="results/m0_feature_set.json")
    ap.add_argument("--predictors", default="esmfold,omegafold")
    ap.add_argument("--n_samples", type=int, default=400)
    ap.add_argument("--n_ref", type=int, default=200, help="natural-CDS seqs for sigma_proj")
    ap.add_argument("--n_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--num_cycle", type=int, default=4, help="OmegaFold recycles")
    ap.add_argument("--out", default="results/m1_calibration.json")
    args = ap.parse_args()
    predictors = [p for p in args.predictors.split(",") if p]
    t0 = time.time()
    fs = json.load(open(args.feature_set))
    org = fs.get("steer_organism", fs.get("organism_primary", "prokaryote"))
    table = D.ORGANISMS[org]["transl_table"]
    feats, s_f = fs["helix_features"], fs["s_f"]

    evo2 = load_evo2("cuda:0")
    sae = BatchTopKSAE(device="cuda:0")

    # reference sequences for sigma_proj (natural CDS windows) + prompts + pos-control refs
    prompts, ref_rich, ref_poor = natural_prompts(org, args.n_samples)
    ref_recs = [json.loads(l) for l in open(os.path.join(D.DATA_DIR, f"m0_dataset_{org}.jsonl"))]
    ref_seqs = [r["cds"][:300] for r in ref_recs if len(r["cds"]) >= 120][:args.n_ref]

    # --- sigma_proj calibration ---
    sig = H.compute_sigma_proj(evo2, sae, feats, s_f, ref_seqs, device="cuda:0")
    sigma_proj, base_norm = sig["sigma_proj"], sig["base_norm"]
    # raw-a <-> c map:  delta = a*base_dir = (c*sigma_proj)*unit  => c = a*base_norm/sigma_proj
    raw_to_c = {a: a * base_norm / sigma_proj for a in ROUND1_RAW_ALPHAS}
    print(f"[m1] sigma_proj={sigma_proj:.4f} ||base_dir||={base_norm:.4f} "
          f"c(raw a=8)={raw_to_c[8]:.3f} c(raw a=16)={raw_to_c[16]:.3f}", flush=True)

    steerer = H.CSteerer(evo2, sae, feats, s_f, sigma_proj=sigma_proj, device="cuda:0")

    # --- c=0 baseline generation (Evo2 only) ---
    recs = H.generate_proteins(evo2, steerer, prompts, 0.0, args.n_tokens, args.temperature,
                               args.top_k, seed_base=1000, table=table)
    for i, r in enumerate(recs):
        r["prompt_cluster"] = i % max(len(set(prompts)), 1)
    del evo2
    torch.cuda.empty_cache()

    # --- fold baseline with BOTH predictors ---
    baseline = {}
    for pred in predictors:
        H.fold_records(recs, pred, device="cuda:0", num_cycle=args.num_cycle)
        agg = H.aggregate_predictor(recs, pred, key="helix_hgi_w")
        baseline[pred] = agg
        print(f"[m1] baseline[{pred}] helix_w={agg.get('helix_hgi_w_mean')} "
              f"helix_gated60={agg.get('helix_hgi_gate60_mean')} "
              f"valid_orf={agg.get('valid_orf_rate'):.2f} folded={agg.get('n_folded')} "
              f"plddt={agg.get('struct_mean_plddt_mean')}", flush=True)

    # --- positive control: fold natural helix-rich vs helix-poor refs under each predictor ---
    from fold import fold_and_read
    pos_ctrl = {}
    for pred in predictors:
        def fold_refs(prots):
            vals = []
            for p in prots[:30]:
                rd = fold_and_read(p, predictor=pred, device="cuda:0", num_cycle=args.num_cycle)
                if rd:
                    vals.append(rd["helix_hgi_w"])
            return vals
        rich = fold_refs(ref_rich); poor = fold_refs(ref_poor)
        pos_ctrl[pred] = {
            "helix_rich_w_mean": float(np.mean(rich)) if rich else None, "n_rich": len(rich),
            "helix_poor_w_mean": float(np.mean(poor)) if poor else None, "n_poor": len(poor),
            "separates": bool(rich and poor and np.mean(rich) > np.mean(poor)),
        }
        print(f"[m1] pos_ctrl[{pred}] rich={pos_ctrl[pred]['helix_rich_w_mean']} "
              f"poor={pos_ctrl[pred]['helix_poor_w_mean']}", flush=True)

    # --- power: N per dose to detect delta=0.1 at power 0.8 (primary endpoint sd, primary pred) ---
    from scipy import stats
    prim = predictors[0]
    sd = baseline[prim].get("helix_hgi_w_std", 0.2) or 0.2
    z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
    n_needed = int(np.ceil(2 * ((z_a + z_b) * sd / 0.1) ** 2))
    # samples-per-dose target: >= max(power N, 300 valid-ORF) per plan M1
    n_per_dose_target = max(n_needed, 300)

    result = {
        "organism": org, "predictors": predictors, "steer_site": H.STEER_SITE,
        "n_helix_features": len(feats),
        "sigma_proj": sigma_proj, "base_norm": base_norm,
        "proj_mean": sig["proj_mean"], "n_ref_codons": sig["n_ref_codons"],
        "raw_alpha_to_c_sigma": {str(k): float(v) for k, v in raw_to_c.items()},
        "c_of_round1_locked_alpha8": raw_to_c[8],
        "primary_endpoint": "helix_hgi_w (pLDDT-weighted alpha-helix HGI fraction)",
        "baseline_c0": baseline,
        "positive_control": pos_ctrl,
        "measurement_sd_primary": sd,
        "n_per_dose_needed_power0.8_delta0.1": n_needed,
        "n_per_dose_target": n_per_dose_target,
        "num_cycle_omegafold": args.num_cycle,
        "elapsed_s": time.time() - t0,
    }
    os.makedirs("results", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[m1] wrote {args.out}; sigma_proj={sigma_proj:.3f} n_per_dose_target={n_per_dose_target} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    import os as _os; sys.stdout.flush(); _os._exit(0)
