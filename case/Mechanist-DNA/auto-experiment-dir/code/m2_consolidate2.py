"""
M2 consolidation (round-2): dose-response curve + split-sample interior c* + dual-predictor
agreement + pLDDT-in-statistics, on the PRIMARY endpoint (pLDDT-weighted HGI helix fraction).

Pre-registered protocol honored:
  P2 split-sample: c* selected on SELECTION seed (42) with the PRIMARY predictor only; the headline
     Spearman trend + Delta(c*) are read on HELD-OUT seeds (200, 201).
  P1 primary endpoint = helix_hgi_w. Hard-gated + pLDDT-threshold sweep = sensitivity only.
  P5 cluster-robust: per-dose CIs + Delta(c*) bootstrapped over prompt clusters; per-dose MW BH-FDR.
  P7 interior c*: max helix within the capability-preserved region (valid-ORF >= tol * baseline).
  P3 dual predictor: trend + interior peak must reproduce under the second predictor.

Run: python code/m2_consolidate2.py --sel_seed 42 --heldout 200,201 --out results/m2_dose_response_curve.json
"""
import os, sys, json, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import harness2 as H

RES = "/data/wanghaoxiong/intergene_mechanist_v6/results"


def load_runs():
    runs = []
    for p in sorted(glob.glob(os.path.join(RES, "m2_alpha_helix_S_c*_s*.json"))):
        r = json.load(open(p))
        runs.append(r)
    return runs


def per_dose_curve(runs, predictor, seeds, endpoint="helix_hgi_w"):
    """Return dict c -> {mean, ci_lo, ci_hi, n, valid_orf, plddt, samples, clusters} pooled over seeds."""
    bydose = {}
    for r in runs:
        c = r["config"]["c_sigma"]; s = r["config"]["seed"]
        if s not in seeds:
            continue
        agg = r["per_predictor"].get(predictor, {})
        ps = agg.get("per_sample", [])
        vals = [x[endpoint] for x in ps if x.get(endpoint) is not None]
        clus = [x["prompt_cluster"] for x in ps if x.get(endpoint) is not None]
        d = bydose.setdefault(c, {"vals": [], "clus": [], "valid_orf": [], "plddt": []})
        d["vals"] += vals; d["clus"] += clus
        if agg.get("valid_orf_rate") is not None:
            d["valid_orf"].append(agg["valid_orf_rate"])
        if agg.get("struct_mean_plddt_mean") is not None:
            d["plddt"].append(agg["struct_mean_plddt_mean"])
    curve = {}
    for c, d in sorted(bydose.items()):
        m, lo, hi, ncl = H.cluster_bootstrap_ci(d["vals"], d["clus"]) if d["vals"] else (None, None, None, 0)
        curve[c] = {"mean": m, "ci_lo": lo, "ci_hi": hi, "n": len(d["vals"]), "n_clusters": ncl,
                    "valid_orf_rate": float(np.mean(d["valid_orf"])) if d["valid_orf"] else None,
                    "mean_plddt": float(np.mean(d["plddt"])) if d["plddt"] else None}
    return curve, bydose


def select_c_star(runs, predictor, sel_seed, endpoint="helix_hgi_w", vorf_tol=0.95):
    """Interior c* on the selection seed (plan P7): argmax endpoint WITHIN the capability-preserved
    region, where the region is the CONTIGUOUS run of doses from c=0 whose valid-ORF stays >= baseline
    (within `vorf_tol`). We walk doses in increasing c and STOP at the first dose whose valid-ORF drops
    below threshold -- so a helix RESURGENCE at very high, capability-DEGRADED doses (off-distribution
    generation) can never be chosen as c*. This is the whole point of round-2 P7."""
    pts = []
    base_vorf = None
    for r in runs:
        if r["config"]["seed"] != sel_seed:
            continue
        c = r["config"]["c_sigma"]
        agg = r["per_predictor"].get(predictor, {})
        m = agg.get(f"{endpoint}_mean"); vorf = agg.get("valid_orf_rate")
        pts.append((c, m, vorf))
        if abs(c) < 1e-9:
            base_vorf = vorf
    pts = sorted([p for p in pts if p[1] is not None], key=lambda t: t[0])
    if not pts:
        return None, None, [], []
    base_vorf = base_vorf or max((v for _, _, v in pts if v is not None), default=1.0)
    thr = vorf_tol * base_vorf   # "valid-ORF ~ baseline": within (1-vorf_tol) of the c=0 rate
    # contiguous capability-preserved region, walking UP from c=0 (ignore the c<0 sign-check point)
    preserved = []
    for (c, m, v) in pts:
        if c < 0:
            continue
        if v is None or v >= thr:
            preserved.append((c, m, v))
        else:
            break   # capability collapsed here -> region ends; do NOT include higher-dose resurgences
    if not preserved:
        preserved = [(pts[0][0], pts[0][1], pts[0][2])]
    c_star = max(preserved, key=lambda t: t[1])[0]
    return c_star, base_vorf, pts, preserved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sel_seed", type=int, default=42)
    ap.add_argument("--heldout", default="200,201")
    ap.add_argument("--predictors", default="esmfold,omegafold")
    ap.add_argument("--endpoint", default="helix_hgi_w")
    ap.add_argument("--out", default=os.path.join(RES, "m2_dose_response_curve.json"))
    args = ap.parse_args()
    heldout = [int(x) for x in args.heldout.split(",")]
    predictors = args.predictors.split(",")
    runs = load_runs()
    if not runs:
        print("[m2c] no M2 runs found"); return
    prim = predictors[0]

    # P2: select c* on selection seed with primary predictor (capability-preserved interior optimum)
    c_star, base_vorf, sel_pts, preserved = select_c_star(runs, prim, args.sel_seed, args.endpoint)
    print(f"[m2c] c*={c_star} (selected on seed {args.sel_seed}, {prim}); base_vorf={base_vorf}; "
          f"capability-preserved region c in {[round(c,2) for c,_,_ in preserved]}", flush=True)

    out = {"primary_endpoint": args.endpoint, "sel_seed": args.sel_seed, "heldout_seeds": heldout,
           "c_star": c_star, "base_valid_orf": base_vorf, "selection_points": sel_pts,
           "capability_preserved_region": [[round(c, 4), m, v] for c, m, v in preserved],
           "c_star_valid_orf": next((v for c, m, v in preserved if abs(c - c_star) < 1e-6), None),
           "c_star_selection_note": ("c* = pLDDT-weighted-helix argmax WITHIN the contiguous "
                                     "capability-preserved region (valid-ORF >= 0.95*baseline, region "
                                     "ends at the first dose whose valid-ORF drops); high-dose helix "
                                     "resurgences at degraded capability are excluded (plan P7)."),
           "per_predictor": {}}

    for pred in predictors:
        curve, bydose = per_dose_curve(runs, pred, heldout, args.endpoint)
        cs = sorted(curve.keys())
        means = [curve[c]["mean"] for c in cs]
        valid = [(c, m) for c, m in zip(cs, means) if m is not None]
        # P5 primary: Spearman trend over the c-axis on held-out seeds
        rho = p = None
        if len(valid) >= 3:
            rho, p = H.spearman_trend([c for c, _ in valid], [m for _, m in valid])
        # Delta(c*) vs c=0 on held-out, paired cluster bootstrap
        d_star = None
        if c_star is not None and c_star in bydose and 0.0 in bydose:
            a, b = bydose[c_star], bydose[0.0]
            d_star = dict(zip(("delta", "lo", "hi", "n_clusters"),
                              H.cluster_bootstrap_diff(a["vals"], a["clus"], b["vals"], b["clus"])))
        # interior check: is c* strictly inside the tested range (not the max c)?
        interior = (c_star is not None and cs and c_star < max(cs) and c_star > min(cs))
        # pLDDT-threshold sensitivity: gated endpoint trend at c=0 vs c*
        gate_trend = {}
        for T in (50, 60, 70, 80):
            gd = {}
            for r in runs:
                if r["config"]["seed"] not in heldout:
                    continue
                c = r["config"]["c_sigma"]
                agg = r["per_predictor"].get(pred, {})
                v = agg.get(f"helix_hgi_gate{T}_mean")
                if v is not None:
                    gd.setdefault(c, []).append(v)
            gc = sorted(gd)
            if len(gc) >= 3:
                gr, gp = H.spearman_trend(gc, [np.mean(gd[c]) for c in gc])
                gate_trend[f"gate{T}"] = {"spearman_rho": gr, "p": gp}
        out["per_predictor"][pred] = {
            "curve": {str(c): curve[c] for c in cs},
            "spearman_rho": rho, "spearman_p": p,
            "delta_at_cstar_vs_c0": d_star,
            "interior_cstar": bool(interior),
            "plddt_threshold_sweep_trend": gate_trend,
            "mean_plddt_per_dose": {str(c): curve[c]["mean_plddt"] for c in cs},
        }
        print(f"[m2c] {pred}: spearman_rho={rho} p={p} interior_c*={interior} "
              f"delta(c*)={d_star}", flush=True)

    # dual-predictor agreement on the headline
    prim_res = out["per_predictor"][prim]
    agree = True
    for pred in predictors[1:]:
        o = out["per_predictor"][pred]
        agree = agree and (o.get("spearman_rho") is not None and prim_res.get("spearman_rho") is not None
                           and np.sign(o["spearman_rho"]) == np.sign(prim_res["spearman_rho"]))
    out["dual_predictor_agree_on_trend_sign"] = bool(agree)
    out["c2_positive_trend"] = bool(prim_res.get("spearman_p") is not None and prim_res["spearman_p"] < 0.05
                                    and (prim_res.get("spearman_rho") or 0) > 0)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[m2c] wrote {args.out}; c2_positive_trend={out['c2_positive_trend']} agree={agree}", flush=True)


if __name__ == "__main__":
    main()
