"""M1: baseline generation + measurement/validity harness + high/low-helix contrast set.

Freezes the assay (harness_config.json), produces the unintervened baseline distribution
(>=500 valid ORFs, 3 seeds), and builds the DEV-split high- vs low-helix contrast set used to
source the CAA direction / probes. TEST split is held out for M3/M4 inference (anti-circularity).
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="valid-ORF target for baseline")
    ap.add_argument("--pool_mult", type=float, default=2.5, help="oversample factor for pool")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--contrast_k", type=int, default=300, help="per-class contrast size (DEV)")
    ap.add_argument("--out", default=os.path.join(C.PROJECT, "results/m1"))
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()
    if args.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)
    C.freeze_harness(os.path.join(args.out, "harness_config.json"))

    seeds = [int(s) for s in args.seeds.split(",")]
    evo = C.Evo2Wrapper("evo2_7b")
    assay = C.HelixAssay()

    # ---- generate a pool across seeds/primers, score every sequence ----
    records = []
    target_pool = int(args.n * args.pool_mult)
    per_call = len(C.DNA_PRIMERS)
    gid = 0
    for seed in seeds:
        n_calls = int(np.ceil(target_pool / len(seeds) / per_call))
        for ccall in range(n_calls):
            prompts = C.DNA_PRIMERS
            seqs = evo.generate(prompts, seed=seed * 1000 + ccall)
            nlls = evo.sequence_nll(seqs)
            for dna, nll, primer in zip(seqs, nlls, prompts):
                orf_dna, protein = C.extract_orf(dna)
                ss = assay.ss_fractions(protein) if protein else dict(
                    helix_frac=0.0, sheet_frac=0.0, plddt=0.0, n_res=0, ok=False)
                valid, checks = C.validity_label(dna, protein, nll)
                records.append(dict(
                    gid=gid, seed=seed, primer=primer, dna=dna,
                    orf_dna=orf_dna, protein=protein,
                    helix_frac=ss["helix_frac"], sheet_frac=ss["sheet_frac"],
                    plddt=ss["plddt"], n_res=ss["n_res"], fold_ok=ss["ok"],
                    mean_nll=nll, gc=C.gc_content(dna),
                    prot_len=(len(protein) if protein else 0),
                    aa_helixfav=C.aa_composition(protein) if protein else 0.0,
                    valid=bool(valid), checks=checks))
                gid += 1
        if sum(r["valid"] for r in records) >= args.n:
            pass  # keep going through seeds for seed coverage
    # ---- DEV/TEST split by gid (group = gid; no leakage) ----
    n = len(records)
    dev, test = C.dev_test_split(n, dev_frac=0.5)
    for r in records:
        r["split"] = "DEV" if r["gid"] in dev else "TEST"

    valid_recs = [r for r in records if r["valid"]]
    helix_all = np.array([r["helix_frac"] for r in records])
    helix_valid = np.array([r["helix_frac"] for r in valid_recs])
    baseline = dict(
        n_total=n, n_valid=len(valid_recs),
        validity_rate=len(valid_recs) / max(1, n),
        primary_helix_all=float(helix_all.mean()),        # invalid->0 already (fold_ok False->0)
        secondary_helix_valid=float(helix_valid.mean()) if len(helix_valid) else 0.0,
        mean_plddt=float(np.mean([r["plddt"] for r in records])),
        seeds=seeds)

    # ---- DEV contrast set: high vs low helix among DEV valid folded seqs ----
    dev_valid = [r for r in valid_recs if r["split"] == "DEV" and r["fold_ok"]]
    dev_valid.sort(key=lambda r: r["helix_frac"])
    k = min(args.contrast_k, len(dev_valid) // 2)
    low = dev_valid[:k]; high = dev_valid[-k:]
    contrast = dict(
        n_pairs=k,
        high_gids=[r["gid"] for r in high], low_gids=[r["gid"] for r in low],
        high_mean_helix=float(np.mean([r["helix_frac"] for r in high])) if k else 0.0,
        low_mean_helix=float(np.mean([r["helix_frac"] for r in low])) if k else 0.0)

    C.save_json(records, os.path.join(args.out, "baseline_records.json"))
    C.save_json(baseline, os.path.join(args.out, "baseline_metrics.json"))
    C.save_json(contrast, os.path.join(args.out, "contrast_set.json"))
    print("[M1] baseline:", json.dumps(baseline, indent=2))
    print("[M1] contrast pairs:", contrast["n_pairs"],
          "high_helix=%.3f low_helix=%.3f" % (contrast["high_mean_helix"], contrast["low_mean_helix"]))

if __name__ == "__main__":
    main()
