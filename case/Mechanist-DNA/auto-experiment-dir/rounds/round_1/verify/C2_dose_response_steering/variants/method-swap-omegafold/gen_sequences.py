"""
Verify variant (C2, dimension=method): swap the structure predictor ESMFold -> OmegaFold.
STAGE 1/2 (runs in the `scientist` conda env, needs Evo2/vortex + the SAE):
  steer Evo2-7B generation at a given (alpha, seed) with the SAME frozen feature set S and
  per-feature scale s_f as the main experiment (results/m0_feature_set.json), translate + ORF-filter
  the SAME way as code/mechanism.py, and SAVE the raw generated/translated protein sequences to a
  JSONL file (the main experiment discarded these -- see EXPERIMENT_AUDIT findings -- so this variant
  incidentally also closes that inspectability gap for the doses it covers).
STAGE 2/2 (gen_and_fold_omegafold.py, runs in the `verify_omegafold` conda env) reads this JSONL,
folds with OmegaFold, runs mkdssp (binary called directly, no conda env needed for it), computes
helix/sheet fractions -- exactly mirroring code/mechanism.py's readout, on a DIFFERENT predictor.

Run (scientist env): CUDA_VISIBLE_DEVICES=<gpu> python gen_sequences.py --alpha 8 --seed 42 --n 200 \
    --out seqs_a8_s42.jsonl
"""
import os, sys, json, argparse, time
sys.path.insert(0, "/data/wanghaoxiong/intergene_mechanist_v6/code")
import torch
import m0_data as D
from evo2_sae import load_evo2, BatchTopKSAE
import mechanism as M
from m1_harness_calibrate import natural_prompts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_set", default="/data/wanghaoxiong/intergene_mechanist_v6/results/m0_feature_set.json")
    ap.add_argument("--alpha", type=float, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)  # explicit, no misleading default (code-review fix)
    # n_tokens/temperature/top_k/min_aa defaults below are IDENTICAL to code/m2_dose_response.py's own
    # defaults and code/mechanism.py's translate_orf default -- not independently chosen for this variant.
    ap.add_argument("--n_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--min_aa", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    fs = json.load(open(args.feature_set))
    org = fs.get("steer_organism", fs.get("organism_primary", "prokaryote"))
    table = D.ORGANISMS[org]["transl_table"]

    evo2 = load_evo2("cuda:0")
    sae = BatchTopKSAE(device="cuda:0")
    steerer = M.Steerer(evo2, sae, fs["helix_features"], fs["s_f"], device="cuda:0")

    prompts, _, _ = natural_prompts(org, args.n)
    prompts = (prompts * (args.n // max(len(prompts), 1) + 1))[:args.n]
    print(f"[gen] alpha={args.alpha} seed={args.seed} n={len(prompts)} |S|={len(fs['helix_features'])}", flush=True)

    ctx = steerer.steer(args.alpha) if args.alpha != 0.0 else __import__("contextlib").nullcontext()
    n_valid = 0
    with open(args.out, "w") as fout, ctx, torch.no_grad():
        for i, prompt in enumerate(prompts):
            try:
                full = M.generate_dna(evo2, prompt, args.n_tokens, args.temperature, args.top_k,
                                      seed=args.seed * 100000 + i)
                gen = full[len(prompt):] if full.startswith(prompt) else full
                prot, valid, meta = M.translate_orf(prompt + gen, min_aa=args.min_aa, table=table)
                rec = {"idx": i, "alpha": args.alpha, "seed": args.seed, "valid_orf": valid,
                       "len_aa": meta.get("len_aa", 0), "had_internal_stop": meta.get("had_internal_stop", False),
                       "prot": prot if valid else None}
                if valid:
                    n_valid += 1
                fout.write(json.dumps(rec) + "\n"); fout.flush()
            except Exception as e:
                fout.write(json.dumps({"idx": i, "alpha": args.alpha, "seed": args.seed,
                                       "error": f"{type(e).__name__}: {str(e)[:150]}"}) + "\n")
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(prompts)} generated; valid_orf={n_valid}", flush=True)

    print(f"[gen] wrote {args.out}: n={len(prompts)} valid_orf={n_valid} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    import os as _os; sys.stdout.flush(); _os._exit(0)  # force GPU release
