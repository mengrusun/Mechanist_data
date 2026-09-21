"""
M3: specificity / double-dissociation (C3). Steer three feature kinds at shared doses and read
out BOTH alpha-helix and beta-sheet fractions + quality guardrail (valid-ORF, pLDDT).
  - alpha_helix_S      : the frozen helix feature set S  -> expect helix up, sheet flat
  - matched_control    : random features matched on activation freq/magnitude -> expect no rise
  - beta_sheet_offtarget: M0-identified beta-sheet feature(s) -> expect sheet up, helix flat
One run == one (feature_kind, alpha, seed).
Run: CUDA_VISIBLE_DEVICES=2,3,4,5 python code/m3_specificity.py --feature_kind matched_control --alpha 8 --seed 42 --out results/m3_matched_control_a8_s42.json
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
    ap.add_argument("--feature_kind", required=True,
                    choices=["alpha_helix_S", "matched_control", "beta_sheet_offtarget"])
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

    if args.feature_kind == "alpha_helix_S":
        feats, s_f = fs["helix_features"], fs["s_f"]
    elif args.feature_kind == "matched_control":
        feats, s_f = fs["matched_control_features"], fs["matched_control_s_f"]
    else:
        feats, s_f = fs["beta_features"], fs["beta_s_f"]

    evo2 = load_evo2("cuda:0")
    sae = BatchTopKSAE(device="cuda:0")
    steerer = M.Steerer(evo2, sae, feats, s_f, device="cuda:0")

    prompts, _, _ = natural_prompts(org, args.n_per_dose)
    prompts = (prompts * (args.n_per_dose // max(len(prompts), 1) + 1))[:args.n_per_dose]
    print(f"[m3] kind={args.feature_kind} alpha={args.alpha} seed={args.seed} n={len(prompts)} "
          f"|feats|={len(feats)}", flush=True)

    recs = M.generate_and_readout(evo2, steerer, prompts, args.alpha, args.n_tokens, args.temperature,
                                  args.top_k, seed_base=args.seed * 100000 + 7, plddt_min=args.plddt_min, table=table)
    result = {"config": {"feature_kind": args.feature_kind, "alpha": args.alpha, "seed": args.seed,
                         "n_per_dose": args.n_per_dose, "organism": org, "n_features": len(feats)},
              "helix_hgi": M.aggregate(recs, "helix_hgi"),
              "helix_h": M.aggregate(recs, "helix_h"),
              "sheet": M.aggregate(recs, "sheet"),
              "per_sample": [{"helix_hgi": r.get("helix_hgi"), "sheet": r.get("sheet"),
                              "plddt": r.get("plddt"), "valid_orf": r.get("valid_orf")}
                             for r in recs],
              "elapsed_s": time.time() - t0}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    h = result["helix_hgi"].get("helix_hgi_mean"); s = result["sheet"].get("sheet_mean")
    print(f"[m3] {args.feature_kind} a={args.alpha} helix={h} sheet={s} "
          f"valid_orf={result['helix_hgi']['valid_orf_rate']:.2f} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    import os as _os; sys.stdout.flush(); _os._exit(0)  # force GPU release; avoid CUDA shutdown hang
