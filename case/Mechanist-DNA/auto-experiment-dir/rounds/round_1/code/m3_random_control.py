"""
M3 fix (C3, iteration-loop type-② mechanism-harness repair): genuine random-direction control.

The verify mechanism-audit FAILed C3 because the decisive double-dissociation stat was taken at a
capability-degraded dose (alpha=16, valid-ORF 0.888->0.667) and the only "specificity" comparator was
ONE fixed matched_control direction -- not the catalogue-required >=30 independent random directions.

This script builds N independent random-direction controls, each = |S| SAE features sampled uniformly
from the dictionary EXCLUDING the helix set S, the beta set, and the matched_control set, weighted by
their own natural activation magnitude s_f (mean nonzero activation, same convention as S), then the
combined steering vector is RESCALED to match ||S.base_dir|| at the injection site (activation-magnitude
matched). Each direction is steered at the LOCKED mid-plateau dose alpha=8 (where the S arm's own
valid-ORF is preserved at ~0.89) and read out DNA->translate->ESMFold->DSSP, exactly like the S arm.

Runtimes are kept STRICTLY SEPARATED (all Evo2 generation first, then all ESMFold folding) across the
whole direction batch -- alternating the two in one process corrupts vortex/flash-attn kernel dispatch.

Run (one contiguous id-range per GPU):
  CUDA_VISIBLE_DEVICES=3 python code/m3_random_control.py --dir_start 0  --dir_end 10 \
      --alpha 8 --n_per_dose 80 --out results/m3_random_control_d0-10.json
"""
import os, sys, json, argparse, time
import numpy as np, torch
sys.path.insert(0, "code")
import m0_data as D
from evo2_sae import load_evo2, BatchTopKSAE
import mechanism as M
from m1_harness_calibrate import natural_prompts

GLOBAL_SEED = 20260718  # fixed base so the 33 random directions are reproducible & independent


def build_base_dir(sae, feats, s_f, device):
    Wc = sae.W[:, torch.as_tensor(feats, dtype=torch.long, device=device)].float()  # (4096,|feats|)
    sf = torch.as_tensor(s_f, dtype=torch.float32, device=device)
    return (Wc * sf.unsqueeze(0)).sum(1)  # (4096,)


def col_scales(Xcsc, feats):
    """s_f = mean nonzero activation of each feature (norm space), same convention as S/matched/beta."""
    out = []
    for f in feats:
        col = Xcsc[:, int(f)]
        out.append(float(col.data.mean()) if col.nnz > 0 else 1.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_set", default="results/m0_feature_set.json")
    ap.add_argument("--dir_start", type=int, required=True)
    ap.add_argument("--dir_end", type=int, required=True)   # inclusive
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--n_per_dose", type=int, default=80)
    ap.add_argument("--n_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--plddt_min", type=float, default=50.0)
    ap.add_argument("--n_features", type=int, default=19)   # == |S|
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    t0 = time.time()
    dev = "cuda:0"

    fs = json.load(open(args.feature_set))
    org = fs.get("steer_organism", fs.get("organism_primary", "prokaryote"))
    table = D.ORGANISMS[org]["transl_table"]

    evo2 = load_evo2(dev)
    sae = BatchTopKSAE(device=dev)

    # target norm = ||S.base_dir|| (activation-magnitude-matched null)
    S_feats, S_sf = fs["helix_features"], fs["s_f"]
    S_base = build_base_dir(sae, S_feats, S_sf, dev)
    target_norm = float(S_base.norm().item())

    # exclusion set: S + beta + matched (uniform draw from everything else)
    excl = set(int(x) for x in fs["helix_features"]) | set(int(x) for x in fs["beta_features"]) \
        | set(int(x) for x in fs["matched_control_features"])
    n_dict = int(sae.W.shape[1])
    pool = np.array([f for f in range(n_dict) if f not in excl], dtype=np.int64)

    from scipy import sparse
    Xcsc = sparse.load_npz(os.path.join(D.DATA_DIR, f"m0_acts_{org}.npz")).tocsc()

    prompts_all, _, _ = natural_prompts(org, args.n_per_dose)
    prompts_all = (prompts_all * (args.n_per_dose // max(len(prompts_all), 1) + 1))[:args.n_per_dose]

    steerer = M.Steerer(evo2, sae, S_feats, S_sf, device=dev)  # base_dir overridden per direction

    dir_ids = list(range(args.dir_start, args.dir_end + 1))
    per_dir = {}     # dir_id -> {feats, s_f, records}
    # ---- PHASE 1: Evo2 generation for ALL directions (no ESMFold loaded yet) ----
    for did in dir_ids:
        rng = np.random.default_rng(GLOBAL_SEED + did)
        feats = sorted(int(x) for x in rng.choice(pool, size=args.n_features, replace=False))
        s_f = col_scales(Xcsc, feats)
        base = build_base_dir(sae, feats, s_f, dev)
        base = base * (target_norm / float(base.norm().item()))  # match ||S.base_dir||
        steerer.base_dir = base
        seed_base = did * 100000 + 7
        records = []
        with steerer.steer(args.alpha):
            for i, prompt in enumerate(prompts_all):
                rec = {"valid_orf": False, "gated": True, "prot": None}
                try:
                    full = M.generate_dna(evo2, prompt, args.n_tokens, args.temperature,
                                          args.top_k, seed=seed_base + i)
                    gen = full[len(prompt):] if full.startswith(prompt) else full
                    prot, valid, meta = M.translate_orf(prompt + gen, min_aa=30, table=table)
                    rec = {"valid_orf": valid, "len_aa": meta.get("len_aa", 0),
                           "gated": True, "prot": prot if valid else None}
                except Exception as e:
                    rec["error"] = f"gen: {type(e).__name__}: {str(e)[:80]}"
                records.append(rec)
        per_dir[did] = {"feats": feats, "s_f": s_f, "records": records}
        nv = sum(r.get("valid_orf", False) for r in records)
        print(f"[rand] gen dir={did} |feats|={len(feats)} valid_orf={nv}/{len(records)} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # ---- PHASE 2: ESMFold + DSSP for ALL directions (Evo2 generation fully done) ----
    for did in dir_ids:
        for rec in per_dir[did]["records"]:
            if not rec.get("prot"):
                continue
            try:
                pdb, plddt = M.esmfold_pdb(rec["prot"], device=dev)
                rec["plddt"] = plddt
                if pdb is not None and plddt is not None and plddt >= args.plddt_min:
                    fr = M.dssp_fractions(pdb)
                    if fr:
                        rec.update({"helix_hgi": fr["helix_hgi"], "sheet": fr["sheet"],
                                    "n_resolved": fr["n_resolved"], "gated": False})
            except Exception as e:
                rec["error"] = f"fold: {type(e).__name__}: {str(e)[:80]}"
                torch.cuda.empty_cache()
        print(f"[rand] fold dir={did} done ({time.time()-t0:.0f}s)", flush=True)

    # ---- aggregate per direction ----
    directions = []
    for did in dir_ids:
        recs = per_dir[did]["records"]
        agg_h = M.aggregate(recs, "helix_hgi")
        agg_s = M.aggregate(recs, "sheet")
        directions.append({
            "dir_id": did,
            "feats": per_dir[did]["feats"],
            "helix_mean": agg_h.get("helix_hgi_mean"),
            "sheet_mean": agg_s.get("sheet_mean"),
            "valid_orf_rate": agg_h.get("valid_orf_rate"),
            "n_folded_gated": agg_h.get("n_folded_gated"),
            "plddt_mean": agg_h.get("plddt_mean"),
            "helix_samples": [r["helix_hgi"] for r in recs if (not r.get("gated", True)) and "helix_hgi" in r],
        })
        for rec in recs:
            rec.pop("prot", None)

    result = {"alpha": args.alpha, "n_per_dose": args.n_per_dose, "n_features": args.n_features,
              "organism": org, "target_norm": target_norm, "global_seed": GLOBAL_SEED,
              "dir_start": args.dir_start, "dir_end": args.dir_end,
              "directions": directions, "elapsed_s": time.time() - t0}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    print(f"[rand] WROTE {args.out}  n_dirs={len(directions)} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    import os as _os; sys.stdout.flush(); _os._exit(0)  # force GPU release; avoid CUDA shutdown hang
