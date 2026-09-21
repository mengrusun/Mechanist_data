"""
M3 random-direction null (round-2, PRIMARY C3 statistic): >=30 (target >=50) independent
norm-matched random directions at the locked interior dose c*, read out on the pLDDT-weighted
primary endpoint under BOTH predictors.

Each random direction = |S| SAE features drawn uniformly EXCLUDING S u beta u matched-control,
weighted by their own natural s_f. In sigma_proj-unit dosing the injection-site perturbation norm is
c*sigma_proj for every direction, so directions are norm-matched to S BY CONSTRUCTION (plan P4).
Per-direction generation impact (valid-ORF + perturbation ratio) is recorded so S is not special
merely by a milder/harsher global perturbation.

Runtimes STRICTLY SEPARATED: all Evo2 generation first (frozen), then fold each predictor.

CRASH-SAFE (round-2 null re-dispatch fix): incremental phase-level checkpoint written atomically to
`<out>.ckpt.json` after EACH direction's generation and after EACH (predictor, direction) fold. A
wall-kill therefore loses at most the single in-progress direction's current phase, never the whole
chunk. On restart the script resumes from the checkpoint: it skips already-generated directions
(and does NOT even load Evo2 if generation is fully done) and skips already-folded (predictor,dir)
pairs. The final `<out>` is written only when every direction/predictor is complete, then the
checkpoint is deleted.

Run (one contiguous id-range per GPU, both predictors folded per run):
  CUDA_VISIBLE_DEVICES=3 python code/m3_random2.py --dir_start 0 --dir_end 2 --c_sigma 21.4801 \
     --n_per_dose 120 --predictors esmfold,omegafold --calib results/m1_calibration.json \
     --out results/m3_random_c21.4801_d0-2.json
"""
import os, sys, json, argparse, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(__file__))
import m0_data as D
from evo2_sae import load_evo2, BatchTopKSAE
import harness2 as H
from m1_harness_calibrate import natural_prompts

GLOBAL_SEED = 20260719  # reproducible & independent random directions (round-2)


def col_scales(Xcsc, feats):
    out = []
    for f in feats:
        col = Xcsc[:, int(f)]
        out.append(float(col.data.mean()) if col.nnz > 0 else 1.0)
    return out


def _ckpt_meta(args, predictors):
    return {"dir_start": args.dir_start, "dir_end": args.dir_end,
            "c_sigma": args.c_sigma, "n_per_dose": args.n_per_dose,
            "predictors": predictors}


def save_ckpt(ckpt_path, args, predictors, per_dir, gen_done, fold_done):
    """Atomically persist the in-progress chunk state so a wall-kill is recoverable."""
    payload = {"meta": _ckpt_meta(args, predictors),
               "gen_done": sorted(int(x) for x in gen_done),
               "fold_done": {p: sorted(int(x) for x in fold_done[p]) for p in predictors},
               "per_dir": {str(did): per_dir[did] for did in per_dir}}
    tmp = ckpt_path + ".tmp"
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, ckpt_path)  # atomic on POSIX


def load_ckpt(ckpt_path, args, predictors):
    """Return (per_dir, gen_done, fold_done) resumed from a matching checkpoint, else empty state."""
    per_dir, gen_done = {}, set()
    fold_done = {p: set() for p in predictors}
    if not os.path.exists(ckpt_path):
        return per_dir, gen_done, fold_done
    try:
        ck = json.load(open(ckpt_path))
        if ck.get("meta") != _ckpt_meta(args, predictors):
            print(f"[rand] ckpt meta mismatch -> ignoring {ckpt_path}, starting fresh", flush=True)
            return per_dir, gen_done, fold_done
        per_dir = {int(k): v for k, v in ck["per_dir"].items()}
        gen_done = set(int(x) for x in ck.get("gen_done", []))
        fd = ck.get("fold_done", {})
        fold_done = {p: set(int(x) for x in fd.get(p, [])) for p in predictors}
        print(f"[rand] RESUME from {ckpt_path}: gen_done={sorted(gen_done)} "
              f"fold_done={{ {', '.join(p+':'+str(sorted(fold_done[p])) for p in predictors)} }}",
              flush=True)
    except Exception as e:
        print(f"[rand] ckpt load failed ({type(e).__name__}: {e}) -> starting fresh", flush=True)
        return {}, set(), {p: set() for p in predictors}
    return per_dir, gen_done, fold_done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feature_set", default="results/m0_feature_set.json")
    ap.add_argument("--calib", default="results/m1_calibration.json")
    ap.add_argument("--dir_start", type=int, required=True)
    ap.add_argument("--dir_end", type=int, required=True)   # inclusive
    ap.add_argument("--c_sigma", type=float, required=True)
    ap.add_argument("--n_per_dose", type=int, default=120)
    ap.add_argument("--n_tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_k", type=int, default=4)
    ap.add_argument("--predictors", default="esmfold,omegafold")
    ap.add_argument("--num_cycle", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    predictors = [p for p in args.predictors.split(",") if p]
    t0 = time.time()
    dev = "cuda:0"
    fs = json.load(open(args.feature_set))
    org = fs.get("steer_organism", fs.get("organism_primary", "prokaryote"))
    table = D.ORGANISMS[org]["transl_table"]
    calib = json.load(open(args.calib))
    sigma_proj = calib["sigma_proj"]

    sae = BatchTopKSAE(device=dev)
    S_feats, S_sf = fs["helix_features"], fs["s_f"]
    n_features = len(S_feats)

    excl = set(int(x) for x in fs["helix_features"]) | set(int(x) for x in fs["beta_features"]) \
        | set(int(x) for x in fs["matched_control_features"])
    n_dict = int(sae.W.shape[1])
    pool = np.array([f for f in range(n_dict) if f not in excl], dtype=np.int64)

    from scipy import sparse
    Xcsc = sparse.load_npz(os.path.join(D.DATA_DIR, f"m0_acts_{org}.npz")).tocsc()

    prompts, _, _ = natural_prompts(org, args.n_per_dose)
    n_unique = max(len(prompts), 1)
    prompts = (prompts * (args.n_per_dose // n_unique + 1))[:args.n_per_dose]

    dir_ids = list(range(args.dir_start, args.dir_end + 1))
    ckpt_path = args.out + ".ckpt.json"
    per_dir, gen_done, fold_done = load_ckpt(ckpt_path, args, predictors)

    # PHASE 1: Evo2 generation for the directions that still need it (Evo2 loaded ONLY if needed)
    need_gen = [did for did in dir_ids if did not in gen_done]
    if need_gen:
        evo2 = load_evo2(dev)
        steerer = H.CSteerer(evo2, sae, S_feats, S_sf, sigma_proj=sigma_proj, device=dev)
        for did in need_gen:
            rng = np.random.default_rng(GLOBAL_SEED + did)
            feats = sorted(int(x) for x in rng.choice(pool, size=n_features, replace=False))
            s_f = col_scales(Xcsc, feats)
            # sigma_proj-unit dosing => magnitude c*sigma_proj is identical to S (norm-matched)
            steerer.set_direction(feats, s_f, sigma_proj=sigma_proj)
            recs = H.generate_proteins(evo2, steerer, prompts, args.c_sigma, args.n_tokens,
                                       args.temperature, args.top_k,
                                       seed_base=did * 100000 + 7, table=table)
            for i, r in enumerate(recs):
                r["prompt_cluster"] = i % n_unique
            per_dir[did] = {"feats": feats, "records": recs}
            gen_done.add(did)
            save_ckpt(ckpt_path, args, predictors, per_dir, gen_done, fold_done)
            nv = sum(r.get("valid_orf", False) for r in recs)
            print(f"[rand] gen dir={did} valid_orf={nv}/{len(recs)} impact={recs[0].get('impact_ratio'):.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        del evo2
        torch.cuda.empty_cache()
    else:
        print(f"[rand] all {len(dir_ids)} dirs already generated (resumed) -> skip Evo2 load "
              f"({time.time()-t0:.0f}s)", flush=True)

    # PHASE 2: fold with BOTH predictors, checkpointing after each (predictor, direction)
    for pred in predictors:
        for did in dir_ids:
            if did in fold_done[pred]:
                continue
            H.fold_records(per_dir[did]["records"], pred, device=dev, num_cycle=args.num_cycle)
            fold_done[pred].add(did)
            save_ckpt(ckpt_path, args, predictors, per_dir, gen_done, fold_done)
        print(f"[rand] fold[{pred}] done for {len(dir_ids)} dirs ({time.time()-t0:.0f}s)", flush=True)

    directions = []
    for did in dir_ids:
        recs = per_dir[did]["records"]
        entry = {"dir_id": did, "feats": per_dir[did]["feats"], "per_predictor": {}}
        for pred in predictors:
            agg = H.aggregate_predictor(recs, pred, key="helix_hgi_w")
            entry["per_predictor"][pred] = agg
        directions.append(entry)
        for r in recs:
            r.pop("prot", None)
            for pred in predictors:
                r.pop(f"struct_{pred}", None)

    result = {"c_sigma": args.c_sigma, "sigma_proj": sigma_proj, "n_per_dose": args.n_per_dose,
              "n_features": n_features, "organism": org, "global_seed": GLOBAL_SEED,
              "dir_start": args.dir_start, "dir_end": args.dir_end, "predictors": predictors,
              "directions": directions, "elapsed_s": time.time() - t0}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(result, open(args.out, "w"), indent=2)
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)  # chunk fully complete -> checkpoint no longer needed
    print(f"[rand] WROTE {args.out} n_dirs={len(directions)} ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
    import os as _os; sys.stdout.flush(); _os._exit(0)
