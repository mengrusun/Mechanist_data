"""
M2: dose-response steering sweep (C2). Generate N sequences per dose alpha (incl. alpha=0 and
negative alpha sign-check), translate->ESMFold->DSSP, compute mean alpha-helix fraction per dose.
One run == one (alpha, seed). Consolidation computes the Spearman trend + optimal alpha*.
Run: CUDA_VISIBLE_DEVICES=2,3,4,5 python code/m2_dose_response.py --alpha 4 --seed 42 --n_per_dose 300 --out results/m2_a4_s42.json
"""
import os, sys, json, argparse, time
import numpy as np, torch
sys.path.insert(0, "code")
import m0_data as D
from evo2_sae import load_evo2, BatchTopKSAE
import mechanism as M
from m1_harness_calibrate import natural_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_set", default="results/m0_feature_set.json")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n_per_dose", type=int, default=300)
    ap.add_argument("--n_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--plddt_min", type=float, default=50.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    fs = json.load(open(args.feature_set))
    org = fs.get("steer_organism", fs.get("organism_primary", "prokaryote"))
    table = D.ORGANISMS[org]["transl_table"]

    evo2 = load_evo2("cuda:0")
    sae = BatchTopKSAE(device="cuda:0")
    steerer = M.Steerer(evo2, sae, fs["helix_features"], fs["s_f"], device="cuda:0")

    prompts, _, _ = natural_prompts(org, args.n_per_dose)
    prompts = (prompts * (args.n_per_dose // max(len(prompts), 1) + 1))[:args.n_per_dose]
    print(f"[m2] alpha={args.alpha} seed={args.seed} n={len(prompts)} |S|={len(fs['helix_features'])}", flush=True)

    recs = M.generate_and_readout(evo2, steerer, prompts, args.alpha, args.n_tokens, args.temperature,
                                  args.top_k, seed_base=args.seed * 100000, plddt_min=args.plddt_min, table=table)
    agg_h = M.aggregate(recs, "helix_hgi")
    agg_hh = M.aggregate(recs, "helix_h")
    agg_s = M.aggregate(recs, "sheet")
    result = {"config": {"alpha": args.alpha, "seed": args.seed, "n_per_dose": args.n_per_dose,
                         "n_tokens": args.n_tokens, "temperature": args.temperature, "top_k": args.top_k,
                         "organism": org, "n_helix_features": len(fs["helix_features"])},
              "helix_hgi": agg_h, "helix_h": agg_hh, "sheet": agg_s,
              "per_sample_helix_hgi": [r.get("helix_hgi") for r in recs if not r.get("gated", True)],
              "elapsed_s": time.time() - t0}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[m2] alpha={args.alpha} seed={args.seed} helix_hgi_mean={agg_h.get('helix_hgi_mean')} "
          f"valid_orf={agg_h['valid_orf_rate']:.2f} gated_pass={agg_h['gated_pass_rate']:.2f} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    import os as _os; sys.stdout.flush(); _os._exit(0)  # force GPU release; avoid CUDA shutdown hang
