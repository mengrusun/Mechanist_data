"""M4: specificity + validity controls (C2 + causal specificity), all on the TEST split.

At the C1-winning (site, coef):
- matched_control: random unit direction of equal norm -> expect no helix gain
- sham: permuted/shuffled CAA direction of identical norm at the same site -> expect no gain
- validity: C2 non-inferiority on validity rate over the FULL generated set + off-target vs baseline
- frontier: helix gain vs validity rate across the coefficient sweep (reads M3 output)
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import common as C
from run_m3 import build_caa_vector, score_generations

def gen_condition(evo, assay, site, vec, coef, seeds, n):
    rows = []
    for seed in seeds:
        gen = []
        for cc in range(int(np.ceil(n / len(C.DNA_PRIMERS)))):
            evo.clear_hooks()
            if vec is not None and coef != 0.0:
                evo.add_steering_hook(site, vec, coef)
            seqs = evo.generate(C.DNA_PRIMERS, seed=900000 + seed * 1000 + cc)
            evo.clear_hooks()
            nlls = evo.sequence_nll(seqs)
            gen.extend(score_generations(evo, assay, seqs, nlls))
        gen = gen[:n]
        rows.append(gen)
    return rows

def agg(rows):
    allg = [g for seed in rows for g in seed]
    h = np.array([g["helix_frac"] for g in allg])
    return dict(helix_all=float(h.mean()),
                validity_rate=float(np.mean([g["valid"] for g in allg])),
                sheet=float(np.mean([g["sheet_frac"] for g in allg])),
                gc=float(np.mean([g["gc"] for g in allg])),
                mean_nll=float(np.nanmean([g["mean_nll"] for g in allg])),
                n=len(allg))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--winning", default=os.path.join(C.PROJECT, "results/m3/winning.json"),
                    help="JSON: {site, coef}")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--ni_margin", type=float, default=0.05, help="C2 non-inferiority margin")
    ap.add_argument("--m1_out", default=os.path.join(C.PROJECT, "results/m1"))
    ap.add_argument("--out", default=os.path.join(C.PROJECT, "results/m4"))
    ap.add_argument("--gpu", default=None)
    args = ap.parse_args()
    if args.gpu: os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.makedirs(args.out, exist_ok=True)
    win = json.load(open(args.winning))
    site, coef = int(win["site"]), float(win["coef"])
    seeds = [int(s) for s in args.seeds.split(",")]

    evo = C.Evo2Wrapper("evo2_7b")
    assay = C.HelixAssay()
    vec, _ = build_caa_vector(evo, args.m1_out, site)
    vnorm = float(np.linalg.norm(vec))
    rng = np.random.default_rng(0)

    # matched-control: random unit vector, equal norm
    rand = rng.standard_normal(vec.shape).astype(np.float32)
    rand = rand / (np.linalg.norm(rand) + 1e-9) * vnorm
    # sham: permuted CAA direction, identical norm
    sham = vec[rng.permutation(vec.shape[0])].copy()
    sham = sham / (np.linalg.norm(sham) + 1e-9) * vnorm

    conds = dict(
        baseline=agg(gen_condition(evo, assay, site, None, 0.0, seeds, args.n)),
        caa_win=agg(gen_condition(evo, assay, site, vec, coef, seeds, args.n)),
        matched_control=agg(gen_condition(evo, assay, site, rand, coef, seeds, args.n)),
        sham=agg(gen_condition(evo, assay, site, sham, coef, seeds, args.n)),
    )
    base_v = conds["baseline"]["validity_rate"]; win_v = conds["caa_win"]["validity_rate"]
    conds["c2_noninferiority"] = dict(
        margin=args.ni_margin, baseline_validity=base_v, winning_validity=win_v,
        delta=win_v - base_v, non_inferior=bool(win_v >= base_v - args.ni_margin))
    conds["specificity"] = dict(
        caa_gain=conds["caa_win"]["helix_all"] - conds["baseline"]["helix_all"],
        matched_gain=conds["matched_control"]["helix_all"] - conds["baseline"]["helix_all"],
        sham_gain=conds["sham"]["helix_all"] - conds["baseline"]["helix_all"])

    # frontier from M3 rows
    frontier = []
    for f in os.listdir(os.path.join(C.PROJECT, "results/m3")):
        if f.startswith(f"s{site}_caa") and f.endswith(".json"):
            for r in json.load(open(os.path.join(C.PROJECT, "results/m3", f)))["rows"]:
                frontier.append(dict(coef=r["coef"], helix=r["helix_all"],
                                     validity=r["validity_rate"]))
    conds["validity_frontier"] = sorted(frontier, key=lambda d: d["coef"])
    C.save_json(dict(site=site, coef=coef, vnorm=vnorm, **conds),
                os.path.join(args.out, "specificity.json"))
    print("[M4] specificity:", json.dumps(conds["specificity"], indent=2))
    print("[M4] C2 non-inferiority:", json.dumps(conds["c2_noninferiority"], indent=2))

if __name__ == "__main__":
    main()
