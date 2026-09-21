"""
Consolidate the method-swap-omegafold variant's per-(alpha,seed) outputs into one summary,
mirroring code/m2_consolidate.py's shape (per_dose table + Spearman trend) so it can be compared
directly against results/m2_dose_response_curve.json. Also folds in the same-sequence ESMFold
cross-check and sequence-diagnostics outputs for a 3-way comparison.

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

    omega_runs = {}   # alpha -> [per-seed omegafold aggregate dicts]
    esm_runs = {}      # alpha -> [per-seed esmfold-samecells aggregate dicts]
    diag_runs = {}     # alpha -> [per-seed seq-diagnostics dicts]

    for f in glob.glob(os.path.join(args.data_dir, "omegafold_a*_s*.json")):
        d = json.load(open(f))
        base = os.path.basename(f)
        alpha = float(base.split("_a")[1].split("_s")[0])
        omega_runs.setdefault(alpha, []).append(d["aggregate"])
    for f in glob.glob(os.path.join(args.data_dir, "esmfold_samecells_a*_s*.json")):
        d = json.load(open(f))
        base = os.path.basename(f)
        alpha = float(base.split("_a")[1].split("_s")[0])
        esm_runs.setdefault(alpha, []).append(d["aggregate"])
    for f in glob.glob(os.path.join(args.data_dir, "diag_a*_s*.json")):
        d = json.load(open(f))
        base = os.path.basename(f)
        alpha = float(base.split("_a")[1].split("_s")[0])
        diag_runs.setdefault(alpha, []).append(d)

    alphas = sorted(omega_runs.keys())
    per_dose = {}
    for a in alphas:
        oagg = omega_runs[a]
        omega_helix = [x["helix_hgi_mean"] for x in oagg if x.get("helix_hgi_mean") is not None]
        omega_sheet = [x["sheet_mean"] for x in oagg if x.get("sheet_mean") is not None]
        eagg = esm_runs.get(a, [])
        esm_helix = [x["helix_hgi_mean"] for x in eagg if x.get("helix_hgi_mean") is not None]
        esm_sheet = [x["sheet_mean"] for x in eagg if x.get("sheet_mean") is not None]
        dagg = diag_runs.get(a, [])
        entropy = [x["entropy"]["mean"] for x in dagg if x.get("entropy", {}).get("mean") is not None]
        homopoly = [x["max_homopolymer_run"]["mean"] for x in dagg if x.get("max_homopolymer_run", {}).get("mean") is not None]
        repeat = [x["repeat_fraction_k4"]["mean"] for x in dagg if x.get("repeat_fraction_k4", {}).get("mean") is not None]
        per_dose[a] = {
            "omegafold_helix_hgi_mean": float(np.mean(omega_helix)) if omega_helix else None,
            "omegafold_sheet_mean": float(np.mean(omega_sheet)) if omega_sheet else None,
            "omegafold_n_seeds": len(oagg),
            "esmfold_samecells_helix_hgi_mean": float(np.mean(esm_helix)) if esm_helix else None,
            "esmfold_samecells_sheet_mean": float(np.mean(esm_sheet)) if esm_sheet else None,
            "seq_entropy_mean": float(np.mean(entropy)) if entropy else None,
            "seq_max_homopolymer_run_mean": float(np.mean(homopoly)) if homopoly else None,
            "seq_repeat_fraction_k4_mean": float(np.mean(repeat)) if repeat else None,
        }

    xs, ys = [], []
    for a in alphas:
        for d in omega_runs[a]:
            if d.get("helix_hgi_mean") is not None:
                xs.append(a); ys.append(d["helix_hgi_mean"])
    spearman = stats.spearmanr(xs, ys) if len(set(xs)) > 1 else (None, None)

    main_exp = json.load(open("/data/wanghaoxiong/intergene_mechanist_v6/results/m2_dose_response_curve.json"))
    main_per_dose = {float(k): v for k, v in main_exp["per_dose"].items()}

    result = {
        "alphas": alphas, "per_dose": per_dose,
        "spearman_rho_omegafold": float(spearman[0]) if spearman[0] is not None else None,
        "spearman_p_omegafold": float(spearman[1]) if spearman[1] is not None else None,
        "helix_at_0_omegafold": per_dose.get(0.0, {}).get("omegafold_helix_hgi_mean"),
        "helix_at_max_omegafold": per_dose.get(max(alphas), {}).get("omegafold_helix_hgi_mean") if alphas else None,
        "main_experiment_comparison": {
            str(a): {"main_esmfold_helix_hgi_mean": main_per_dose.get(a, {}).get("helix_hgi_mean"),
                     "variant_omegafold_helix_hgi_mean": per_dose[a]["omegafold_helix_hgi_mean"],
                     "variant_esmfold_samecells_helix_hgi_mean": per_dose[a]["esmfold_samecells_helix_hgi_mean"]}
            for a in alphas if a in main_per_dose
        },
    }
    json.dump(result, open(args.out, "w"), indent=2)
    print("=== Method-swap (OmegaFold) variant summary ===")
    for a in alphas:
        pd = per_dose[a]
        print(f"  alpha={a:>5}: omegafold_helix={pd['omegafold_helix_hgi_mean']} "
              f"esmfold_samecells_helix={pd['esmfold_samecells_helix_hgi_mean']} "
              f"seq_entropy={pd['seq_entropy_mean']} max_homopoly={pd['seq_max_homopolymer_run_mean']}")
    print(f"Spearman (OmegaFold) rho={result['spearman_rho_omegafold']} p={result['spearman_p_omegafold']}")


if __name__ == "__main__":
    main()
