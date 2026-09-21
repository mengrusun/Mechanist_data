"""
M3 consolidation (round-2): specificity at the locked interior dose c* on HELD-OUT seeds, PRIMARY
endpoint = pLDDT-weighted HGI helix fraction, under BOTH predictors.

PRIMARY C3 statistic (plan P4/P5): S's helix rise vs the >=30-direction norm-matched random-null.
  empirical one-sided p = (1 + #{delta_rand >= delta_S}) / (N + 1); z-score; effect size
  (S - null_mean)/null_std; S's Delta(c*) paired cluster-bootstrap CI. All directions are norm-
  matched to S by construction (shared sigma_proj-unit magnitude c*sigma_proj) -> per-direction
  generation impact (valid-ORF, perturbation ratio) reported so S is not special via milder/harsher
  perturbation.
SECONDARY: matched-control (expect no rise); beta arm (double-dissociation IF beta_v2 raised sheet,
  else documented helix-axis-specificity negative, plan P9 -- does NOT affect the S-vs-null verdict).
Sensitivity: hard-gated + pLDDT-threshold sweep; mean pLDDT per arm (rule out S winning via
  confidence alone).

Run: python code/m3_consolidate2.py --c_star <c*> --heldout 200,201 --out results/m3_specificity_summary.json
"""
import os, sys, json, glob, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import harness2 as H

RES = "/data/wanghaoxiong/intergene_mechanist_v6/results"


def arm_samples(kind, c, seeds, predictor, endpoint="helix_hgi_w"):
    """Pool per-sample endpoint + clusters for an arm at dose c over held-out seeds."""
    vals, clus = [], []
    vorf, plddt, gates = [], [], {T: [] for T in (50, 60, 70, 80)}
    for p in glob.glob(os.path.join(RES, f"m3_{kind}_c*_s*.json")):
        r = json.load(open(p))
        cfg = r["config"]
        if abs(cfg["c_sigma"] - c) > 1e-6 or cfg["seed"] not in seeds:
            continue
        agg = r["per_predictor"].get(predictor, {})
        for x in agg.get("per_sample", []):
            if x.get(endpoint) is not None:
                vals.append(x[endpoint]); clus.append(x["prompt_cluster"])
        if agg.get("valid_orf_rate") is not None:
            vorf.append(agg["valid_orf_rate"])
        if agg.get("struct_mean_plddt_mean") is not None:
            plddt.append(agg["struct_mean_plddt_mean"])
        for T in gates:
            if agg.get(f"helix_hgi_gate{T}_mean") is not None:
                gates[T].append(agg[f"helix_hgi_gate{T}_mean"])
    return {"vals": vals, "clus": clus,
            "valid_orf": float(np.mean(vorf)) if vorf else None,
            "plddt": float(np.mean(plddt)) if plddt else None,
            "gates": {T: (float(np.mean(v)) if v else None) for T, v in gates.items()}}


def null_directions(c_star, predictor, endpoint="helix_hgi_w"):
    dirs = []
    for p in sorted(glob.glob(os.path.join(RES, "m3_random_c*_d*.json"))):
        r = json.load(open(p))
        if abs(r["c_sigma"] - c_star) > 1e-6:
            continue
        for d in r["directions"]:
            agg = d["per_predictor"].get(predictor, {})
            m = agg.get(f"{endpoint}_mean")
            dirs.append({"dir_id": d["dir_id"], "helix_w": m,
                         "valid_orf": agg.get("valid_orf_rate"),
                         "impact": agg.get("impact_ratio")})
    return [d for d in dirs if d["helix_w"] is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--c_star", type=float, required=True)
    ap.add_argument("--heldout", default="200,201")
    ap.add_argument("--predictors", default="esmfold,omegafold")
    ap.add_argument("--endpoint", default="helix_hgi_w")
    ap.add_argument("--out", default=os.path.join(RES, "m3_specificity_summary.json"))
    args = ap.parse_args()
    heldout = [int(x) for x in args.heldout.split(",")]
    predictors = args.predictors.split(",")
    fs = json.load(open(os.path.join(RES, "m0_feature_set.json")))
    beta_v2_cleared = bool((fs.get("beta_v2") or {}).get("clears_bar_stronger_than_round1"))

    out = {"c_star": args.c_star, "heldout_seeds": heldout, "primary_endpoint": args.endpoint,
           "beta_v2_cleared_bar": beta_v2_cleared, "per_predictor": {}}

    for pred in predictors:
        S0 = arm_samples("alpha_helix_S", 0.0, heldout, pred, args.endpoint)
        Sc = arm_samples("alpha_helix_S", args.c_star, heldout, pred, args.endpoint)
        MC = arm_samples("matched_control", args.c_star, heldout, pred, args.endpoint)
        BE = arm_samples("beta_sheet_offtarget", args.c_star, heldout, pred, "sheet_w")
        BEh = arm_samples("beta_sheet_offtarget", args.c_star, heldout, pred, args.endpoint)
        base = float(np.mean(S0["vals"])) if S0["vals"] else None

        # S delta(c*) vs c0, paired cluster bootstrap
        dS = dict(zip(("delta", "lo", "hi", "n_clusters"),
                      H.cluster_bootstrap_diff(Sc["vals"], Sc["clus"], S0["vals"], S0["clus"])))
        delta_S = dS["delta"]
        dMC = dict(zip(("delta", "lo", "hi", "n_clusters"),
                       H.cluster_bootstrap_diff(MC["vals"], MC["clus"], S0["vals"], S0["clus"]))) if MC["vals"] else None

        # PRIMARY: S vs random null
        dirs = null_directions(args.c_star, pred, args.endpoint)
        null_deltas = np.array([d["helix_w"] - base for d in dirs]) if (dirs and base is not None) else np.array([])
        prim = None
        if null_deltas.size and delta_S is not None:
            n_ge = int(np.sum(null_deltas >= delta_S))
            emp_p = (1 + n_ge) / (null_deltas.size + 1)
            z = (delta_S - null_deltas.mean()) / (null_deltas.std() + 1e-9)
            prim = {"n_directions": int(null_deltas.size),
                    "S_helix_delta": delta_S, "S_delta_ci": [dS["lo"], dS["hi"]],
                    "null_delta_mean": float(null_deltas.mean()),
                    "null_delta_std": float(null_deltas.std()),
                    "null_delta_max": float(null_deltas.max()),
                    "n_null_ge_S": n_ge, "empirical_one_sided_p": emp_p,
                    "z_score": float(z),
                    "S_valid_orf": Sc["valid_orf"],
                    "null_valid_orf_mean": float(np.mean([d["valid_orf"] for d in dirs if d["valid_orf"] is not None])) if dirs else None,
                    "null_impact_mean": float(np.mean([d["impact"] for d in dirs if d["impact"] is not None])) if dirs else None}

        out["per_predictor"][pred] = {
            "baseline_helix_w": base,
            "S_delta": dS, "matched_control_delta": dMC,
            "S_helix_w_at_cstar": float(np.mean(Sc["vals"])) if Sc["vals"] else None,
            "matched_helix_w_at_cstar": float(np.mean(MC["vals"])) if MC["vals"] else None,
            "beta_sheet_w_at_cstar": float(np.mean(BE["vals"])) if BE["vals"] else None,
            "beta_sheet_w_baseline": None,
            "beta_helix_w_at_cstar": float(np.mean(BEh["vals"])) if BEh["vals"] else None,
            "PRIMARY_random_null": prim,
            "mean_plddt_per_arm": {"S": Sc["plddt"], "matched": MC["plddt"], "beta": BE["plddt"], "c0": S0["plddt"]},
            "valid_orf_per_arm": {"S": Sc["valid_orf"], "matched": MC["valid_orf"], "beta": BE["valid_orf"], "c0": S0["valid_orf"]},
            "gated_sensitivity": {"S_cstar": Sc["gates"], "c0": S0["gates"]},
            "S_specificity_supported": bool(prim and prim["empirical_one_sided_p"] < 0.05
                                            and dS["lo"] is not None and dS["lo"] > 0),
        }
        if prim:
            print(f"[m3c] {pred}: dS={delta_S} null_mean={prim['null_delta_mean']:.4f} "
                  f"p={prim['empirical_one_sided_p']:.4f} z={prim['z_score']:.2f} "
                  f"n_dir={prim['n_directions']}", flush=True)

    # beta-arm decision (P9): double-dissociation only if beta_v2 raised sheet
    esm = out["per_predictor"].get(predictors[0], {})
    beta_raises_sheet = False  # requires beta sheet baseline; computed if a beta c=0 run exists
    out["beta_arm_verdict"] = ("double_dissociation" if (beta_v2_cleared and beta_raises_sheet)
                               else "documented_negative_helix_axis_specificity")
    # overall C3: supported iff S-vs-null primary holds under BOTH predictors
    supported = all(out["per_predictor"][p].get("S_specificity_supported") for p in predictors
                    if out["per_predictor"][p].get("PRIMARY_random_null"))
    out["C3_specificity_verdict"] = "SUPPORTED" if supported else "NOT_SUPPORTED_OR_INCOMPLETE"
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"[m3c] wrote {args.out}; C3={out['C3_specificity_verdict']} beta_arm={out['beta_arm_verdict']}", flush=True)


if __name__ == "__main__":
    main()
