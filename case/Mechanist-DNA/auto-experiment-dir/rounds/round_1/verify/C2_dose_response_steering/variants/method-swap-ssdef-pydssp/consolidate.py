"""
Consolidate the method-swap-ssdef-pydssp variant's per-(alpha,seed) outputs, mirroring
code/m2_consolidate.py's shape (per_dose table + Spearman trend), for BOTH SS-assignment methods
(mkdssp and pydssp) computed on the identical ESMFold-predicted structures -- so the two curves are
directly comparable dose-by-dose, seed-by-seed.

Run: python consolidate.py --data_dir data --out variant_summary.json
"""
import os, sys, json, glob, argparse
import numpy as np
from scipy import stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="variant_summary.json")
    args = ap.parse_args()

    runs = {}  # alpha -> list of per-seed result dicts
    for f in glob.glob(os.path.join(args.data_dir, "results_a*_s*.json")):
        d = json.load(open(f))
        a = d["config"]["alpha"]
        runs.setdefault(a, []).append(d)

    alphas = sorted(runs.keys())
    per_dose = {}
    for a in alphas:
        ds = runs[a]
        mk_helix = [d["helix_hgi_mkdssp"]["mean"] for d in ds if d["helix_hgi_mkdssp"]["mean"] is not None]
        mk_sheet = [d["sheet_mkdssp"]["mean"] for d in ds if d["sheet_mkdssp"]["mean"] is not None]
        pd_helix = [d["helix_pydssp"]["mean"] for d in ds if d["helix_pydssp"]["mean"] is not None]
        pd_sheet = [d["sheet_pydssp"]["mean"] for d in ds if d["sheet_pydssp"]["mean"] is not None]
        gated = [d["gated_pass_rate"] for d in ds]
        orf = [d["valid_orf_rate"] for d in ds]
        per_dose[a] = {
            "n_seeds": len(ds),
            "helix_mkdssp_mean": float(np.mean(mk_helix)) if mk_helix else None,
            "sheet_mkdssp_mean": float(np.mean(mk_sheet)) if mk_sheet else None,
            "helix_pydssp_mean": float(np.mean(pd_helix)) if pd_helix else None,
            "sheet_pydssp_mean": float(np.mean(pd_sheet)) if pd_sheet else None,
            "valid_orf_rate": float(np.mean(orf)) if orf else None,
            "gated_pass_rate": float(np.mean(gated)) if gated else None,
        }

    def spearman_over(key):
        xs, ys = [], []
        for a in alphas:
            for d in runs[a]:
                v = d[key]["mean"]
                if v is not None:
                    xs.append(a); ys.append(v)
        return stats.spearmanr(xs, ys) if len(set(xs)) > 1 else (None, None)

    sp_mk = spearman_over("helix_hgi_mkdssp")
    sp_pd = spearman_over("helix_pydssp")

    main_exp = json.load(open("/data/wanghaoxiong/intergene_mechanist_v6/results/m2_dose_response_curve.json"))
    main_per_dose = {float(k): v for k, v in main_exp["per_dose"].items()}

    result = {
        "alphas": alphas, "per_dose": per_dose,
        "spearman_rho_mkdssp": float(sp_mk[0]) if sp_mk[0] is not None else None,
        "spearman_p_mkdssp": float(sp_mk[1]) if sp_mk[1] is not None else None,
        "spearman_rho_pydssp": float(sp_pd[0]) if sp_pd[0] is not None else None,
        "spearman_p_pydssp": float(sp_pd[1]) if sp_pd[1] is not None else None,
        "helix_at_0_mkdssp": per_dose.get(0.0, {}).get("helix_mkdssp_mean"),
        "helix_at_0_pydssp": per_dose.get(0.0, {}).get("helix_pydssp_mean"),
        "helix_at_max_mkdssp": per_dose.get(max(alphas), {}).get("helix_mkdssp_mean") if alphas else None,
        "helix_at_max_pydssp": per_dose.get(max(alphas), {}).get("helix_pydssp_mean") if alphas else None,
        "main_experiment_comparison": {
            str(a): {"main_experiment_helix_hgi_mean": main_per_dose.get(a, {}).get("helix_hgi_mean"),
                     "variant_mkdssp_helix_hgi_mean": per_dose[a]["helix_mkdssp_mean"],
                     "variant_pydssp_helix_mean": per_dose[a]["helix_pydssp_mean"]}
            for a in alphas if a in main_per_dose
        },
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print("=== Method-swap (pydssp SS-definition) variant summary ===")
    for a in alphas:
        pd_ = per_dose[a]
        print(f"  alpha={a:>5}: mkdssp_helix={pd_['helix_mkdssp_mean']} pydssp_helix={pd_['helix_pydssp_mean']} "
              f"mkdssp_sheet={pd_['sheet_mkdssp_mean']} pydssp_sheet={pd_['sheet_pydssp_mean']} "
              f"orf={pd_['valid_orf_rate']} gated={pd_['gated_pass_rate']} n_seeds={pd_['n_seeds']}")
    print(f"Spearman mkdssp: rho={result['spearman_rho_mkdssp']} p={result['spearman_p_mkdssp']}")
    print(f"Spearman pydssp: rho={result['spearman_rho_pydssp']} p={result['spearman_p_pydssp']}")


if __name__ == "__main__":
    main()
