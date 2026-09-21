"""M3: steer + dose-response (Causal Intervention, the applied lever).

CAA direction = mean(high-helix DEV acts) - mean(low-helix DEV acts) at a chosen block, added to
that block's residual stream during generation over a steering-coefficient dose grid. Evaluated on
TEST-split primers/seeds (anti-circularity). coef 0.0 reproduces the in-run baseline control.
Optional SAE-feature-clamp arm at the SAE block. Every sweep point scored on alpha-helix AND
validity + Evo2-NLL naturalness (Pareto / General-Rule-2).

Parametrized by --site so the queue expands the grid by site (each job runs all coefs x seeds,
keeping the model resident). --mode {caa,sae_clamp}.
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C

def build_caa_vector(evo, m1_out, block):
    recs = {r["gid"]: r for r in json.load(open(os.path.join(m1_out, "baseline_records.json")))}
    con = json.load(open(os.path.join(m1_out, "contrast_set.json")))
    hi = [recs[g]["dna"] for g in con["high_gids"]]
    lo = [recs[g]["dna"] for g in con["low_gids"]]
    a_hi = evo.block_activations(hi, [block])[block].mean(0)
    a_lo = evo.block_activations(lo, [block])[block].mean(0)
    return (a_hi - a_lo).astype(np.float32), con["n_pairs"]

def score_generations(evo, assay, seqs, nlls):
    out = []
    for dna, nll in zip(seqs, nlls):
        orf_dna, protein = C.extract_orf(dna)
        ss = assay.ss_fractions(protein) if protein else dict(
            helix_frac=0.0, sheet_frac=0.0, plddt=0.0, n_res=0, ok=False)
        valid, _ = C.validity_label(dna, protein, nll)
        out.append(dict(helix_frac=ss["helix_frac"], sheet_frac=ss["sheet_frac"],
                        plddt=ss["plddt"], mean_nll=nll, gc=C.gc_content(dna),
                        prot_len=(len(protein) if protein else 0),
                        aa_helixfav=C.aa_composition(protein) if protein else 0.0,
                        valid=bool(valid), fold_ok=ss["ok"], dna=dna, protein=protein))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", type=int, required=True, help="target block index")
    ap.add_argument("--coefs", default="0.0,0.5,1.0,2.0,4.0,8.0")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n", type=int, default=250, help="generations per (coef,seed)")
    ap.add_argument("--mode", choices=["caa", "sae_clamp"], default="caa")
    ap.add_argument("--sae_feats", default="", help="comma feature idxs for sae_clamp")
    ap.add_argument("--clamp_scale", type=float, default=1.0)
    ap.add_argument("--m1_out", default=os.path.join(C.PROJECT, "results/m1"))
    ap.add_argument("--out", default=os.path.join(C.PROJECT, "results/m3"))
    ap.add_argument("--out_tag", default="", help="suffix for sharded (per-seed) outputs")
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()
    if args.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)

    coefs = [float(c) for c in args.coefs.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    evo = C.Evo2Wrapper("evo2_7b")
    assay = C.HelixAssay()

    vec, n_pairs = (None, 0)
    sae = feats = None
    if args.mode == "caa":
        vec, n_pairs = build_caa_vector(evo, args.m1_out, args.site)
        vnorm = float(np.linalg.norm(vec))
    else:
        sae = C.TopKSAE()
        feats = [int(x) for x in args.sae_feats.split(",") if x != ""]
        vnorm = None

    # TEST-split primers (disjoint from DEV vector construction): use the fixed primer set with
    # a TEST seed offset so generations are not those used to build the vector.
    rows = []
    for coef in coefs:
        for seed in seeds:
            n_calls = int(np.ceil(args.n / len(C.DNA_PRIMERS)))
            gen = []
            for cc in range(n_calls):
                evo.clear_hooks()
                if coef != 0.0:
                    if args.mode == "caa":
                        evo.add_steering_hook(args.site, vec, coef)
                    else:
                        evo.add_sae_clamp_hook(args.site, sae, feats, args.clamp_scale * coef)
                seqs = evo.generate(C.DNA_PRIMERS, seed=900000 + seed * 1000 + cc)  # TEST offset
                evo.clear_hooks()
                nlls = evo.sequence_nll(seqs)
                gen.extend(score_generations(evo, assay, seqs, nlls))
            gen = gen[:args.n]
            hall = np.array([g["helix_frac"] for g in gen])
            vrate = np.mean([g["valid"] for g in gen])
            rows.append(dict(site=args.site, mode=args.mode, coef=coef, seed=seed,
                             n=len(gen),
                             helix_all=float(hall.mean()),
                             helix_valid=float(np.mean([g["helix_frac"] for g in gen if g["valid"]])
                                                if any(g["valid"] for g in gen) else 0.0),
                             validity_rate=float(vrate),
                             mean_nll=float(np.nanmean([g["mean_nll"] for g in gen])),
                             sheet=float(np.mean([g["sheet_frac"] for g in gen])),
                             gc=float(np.mean([g["gc"] for g in gen])),
                             plddt=float(np.mean([g["plddt"] for g in gen])),
                             per_gen_helix=[float(x) for x in hall]))
            print("[M3] site=%d mode=%s coef=%.2f seed=%d helix_all=%.3f valid=%.3f nll=%.3f"
                  % (args.site, args.mode, coef, seed, hall.mean(), vrate,
                     np.nanmean([g["mean_nll"] for g in gen])), flush=True)
    tag = f"s{args.site}_{args.mode}{args.out_tag}"
    C.save_json(dict(site=args.site, mode=args.mode, vnorm=vnorm, n_pairs=n_pairs, rows=rows),
                os.path.join(args.out, f"{tag}.json"))

if __name__ == "__main__":
    main()
