"""
M1: build + calibrate the DNA->protein->ESMFold->DSSP readout and the SAE steering hook.
Establishes alpha=0 baseline helix distribution, valid-ORF/pLDDT rates, measurement noise, and
samples-per-dose for power to detect Delta_helix >= 0.1 at power 0.8. Positive control: natural
helix-rich vs helix-poor reference proteins.
Run: CUDA_VISIBLE_DEVICES=2,3,4,5 python code/m1_harness_calibrate.py --n_samples 500 --out results/m1_calibration.json
"""
import os, sys, json, argparse, time
import numpy as np, torch
sys.path.insert(0, "code")
import m0_data as D
from evo2_sae import load_evo2, BatchTopKSAE
import mechanism as M


def natural_prompts(org, n, prompt_nt=45):
    recs = [json.loads(l) for l in open(os.path.join(D.DATA_DIR, f"m0_dataset_{org}.jsonl"))]
    prompts = []
    ref_helix_rich, ref_helix_poor = [], []
    for r in recs:
        cds = r["cds"]; ss = r["codon_ss"]
        if len(cds) < prompt_nt + 30:
            continue
        prompts.append(cds[:prompt_nt])
        # reference proteins for positive control
        labeled = [s for s in ss if s is not None]
        if len(labeled) >= 50:
            hf = sum(s in D.HELIX_HGI for s in labeled) / len(labeled)
            prot = str(__import__("Bio.Seq", fromlist=["Seq"]).Seq(cds[:len(cds)-len(cds)%3]).translate(
                table=D.ORGANISMS[org]["transl_table"])).rstrip("*").replace("*", "")
            if hf >= 0.5:
                ref_helix_rich.append(prot)
            elif hf <= 0.15:
                ref_helix_poor.append(prot)
    return prompts[:n], ref_helix_rich[:40], ref_helix_poor[:40]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_set", default="results/m0_feature_set.json")
    ap.add_argument("--n_samples", type=int, default=500)
    ap.add_argument("--n_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--plddt_min", type=float, default=50.0)
    ap.add_argument("--out", default="results/m1_calibration.json")
    args = ap.parse_args()
    t0 = time.time()
    fs = json.load(open(args.feature_set))
    org = fs.get("steer_organism", fs.get("organism_primary", "prokaryote"))
    table = D.ORGANISMS[org]["transl_table"]

    evo2 = load_evo2("cuda:0")
    sae = BatchTopKSAE(device="cuda:0")
    steerer = M.Steerer(evo2, sae, fs["helix_features"], fs["s_f"], device="cuda:0")

    prompts, ref_rich, ref_poor = natural_prompts(org, args.n_samples)
    print(f"[m1] prompts={len(prompts)} ref_rich={len(ref_rich)} ref_poor={len(ref_poor)}", flush=True)

    # alpha=0 baseline
    recs0 = M.generate_and_readout(evo2, steerer, prompts, 0.0, args.n_tokens, args.temperature,
                                   args.top_k, seed_base=1000, plddt_min=args.plddt_min, table=table)
    agg0 = M.aggregate(recs0, "helix_hgi")
    agg0_sheet = M.aggregate(recs0, "sheet")
    print(f"[m1] alpha=0 helix_hgi_mean={agg0.get('helix_hgi_mean')} valid_orf={agg0['valid_orf_rate']:.2f} "
          f"gated_pass={agg0['gated_pass_rate']:.2f}", flush=True)

    # positive control: fold natural helix-rich vs helix-poor reference proteins
    def fold_refs(prots):
        vals = []
        for p in prots:
            pdb, plddt = M.esmfold_pdb(p, "cuda:0")
            if pdb and plddt and plddt >= args.plddt_min:
                fr = M.dssp_fractions(pdb)
                if fr:
                    vals.append(fr["helix_hgi"])
        return vals
    rich = fold_refs(ref_rich); poor = fold_refs(ref_poor)
    pos_ctrl = {"helix_rich_mean": float(np.mean(rich)) if rich else None, "n_rich": len(rich),
                "helix_poor_mean": float(np.mean(poor)) if poor else None, "n_poor": len(poor)}

    # power: samples per dose to detect delta=0.1 at power 0.8 (two-sample, sd from baseline)
    sd = agg0.get("helix_hgi_std", 0.2) or 0.2
    from scipy import stats
    z_a, z_b = stats.norm.ppf(0.975), stats.norm.ppf(0.8)
    n_needed = int(np.ceil(2 * ((z_a + z_b) * sd / 0.1) ** 2))

    result = {
        "organism": org, "n_prompts": len(prompts), "n_tokens": args.n_tokens,
        "temperature": args.temperature, "top_k": args.top_k, "plddt_min": args.plddt_min,
        "steer_site": M.STEER_SITE, "n_helix_features": len(fs["helix_features"]),
        "baseline_alpha0": agg0, "baseline_alpha0_sheet": agg0_sheet,
        "positive_control": pos_ctrl,
        "measurement_sd": sd, "n_per_dose_needed_power0.8_delta0.1": n_needed,
        "pos_control_separates": (pos_ctrl["helix_rich_mean"] is not None and
                                  pos_ctrl["helix_poor_mean"] is not None and
                                  pos_ctrl["helix_rich_mean"] > pos_ctrl["helix_poor_mean"]),
        "elapsed_s": time.time() - t0,
    }
    os.makedirs("results", exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[m1] wrote {args.out}; n_needed/dose={n_needed}; pos_ctrl_separates={result['pos_control_separates']} "
          f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    import os as _os; sys.stdout.flush(); _os._exit(0)  # force GPU release; avoid CUDA shutdown hang
