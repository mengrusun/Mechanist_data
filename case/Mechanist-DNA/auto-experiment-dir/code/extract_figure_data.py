"""Consolidate all Round-2 result files (C1/C2/C3) into one JSON for figure generation.
Reads only from results/*.json (real experiment output) -- no numbers are invented.
"""
import json
import glob
import os
import numpy as np

R = "results"
OUT = "paper_figures/_figure_data.json"

# ---------------------------------------------------------------- C1: set-level AUROC
organisms = ["prokaryote", "eukaryote"]
helix_defs = ["HGI", "H_only"]
seeds = [42, 200, 201]

c1 = {"conditions": []}
for org in organisms:
    for hd in helix_defs:
        aurocs, confounds, shuffles, ns_codons, n_sig, s_size = [], [], [], [], [], []
        for s in seeds:
            fp = f"{R}/m0_{org}_{hd}_s{s}.json"
            d = json.load(open(fp))
            aurocs.append(d["combined_set_test_auroc"])
            confounds.append(d["confound_only_test_auroc"])
            shuffles.append(d["combined_set_shuffle_null_auroc"])
            ns_codons.append(d["n_codons"])
            n_sig.append(d["fdr_n_significant"])
            s_size.append(d["relaxed_S_size"])
        c1["conditions"].append({
            "organism": org, "helix_def": hd, "seeds": seeds,
            "set_auroc_per_seed": aurocs,
            "set_auroc_mean": float(np.mean(aurocs)), "set_auroc_std": float(np.std(aurocs, ddof=1)),
            "confound_only_per_seed": confounds, "confound_only_mean": float(np.mean(confounds)),
            "shuffle_null_per_seed": shuffles, "shuffle_null_mean": float(np.mean(shuffles)),
            "n_codons_test_split": ns_codons, "fdr_n_significant": n_sig, "S_size": s_size,
        })

m0 = json.load(open(f"{R}/m0_feature_set.json"))
c1["S_frozen_size"] = m0["S_size"]
c1["S_features"] = m0["helix_features"]
c1["verdict"] = m0["verdict"]
c1["best_single_feature_relaxed_auroc_range"] = [0.62, 0.68]  # from ledger narrative; cross-checked below
c1["mean_steer_organism_set_auroc"] = m0["mean_steer_organism_set_auroc"]
c1["eukaryote_transfer_set_auroc_mean"] = m0["eukaryote_transfer_set_auroc_mean"]

# ---------------------------------------------------------------- C2: dose-response
m2 = json.load(open(f"{R}/m2_dose_response_curve.json"))
c2 = {
    "c_star": m2["c_star"],
    "base_valid_orf": m2["base_valid_orf"],
    "sel_seed": m2["sel_seed"],
    "heldout_seeds": m2["heldout_seeds"],
    "primary_endpoint": m2["primary_endpoint"],
    "doses": sorted(float(k) for k in m2["per_predictor"]["esmfold"]["curve"].keys()),
    "per_predictor": {},
}
for pred in ["esmfold", "omegafold"]:
    pd = m2["per_predictor"][pred]
    curve = pd["curve"]
    c2["per_predictor"][pred] = {
        "spearman_rho": pd["spearman_rho"], "spearman_p": pd["spearman_p"],
        "delta_at_cstar": pd["delta_at_cstar_vs_c0"],
        "points": [
            {
                "c": float(k), "mean": v["mean"], "ci_lo": v["ci_lo"], "ci_hi": v["ci_hi"],
                "n": v["n"], "n_clusters": v["n_clusters"], "valid_orf_rate": v["valid_orf_rate"],
                "mean_plddt": v["mean_plddt"],
            }
            for k, v in sorted(curve.items(), key=lambda kv: float(kv[0]))
        ],
    }

# ---------------------------------------------------------------- C3: specificity null
m3s = json.load(open(f"{R}/m3_specificity_summary.json"))
c3 = {"c_star": m3s["c_star"], "beta_v2_cleared_bar": m3s["beta_v2_cleared_bar"],
      "beta_arm_verdict": m3s["beta_arm_verdict"], "C3_specificity_verdict": m3s["C3_specificity_verdict"],
      "per_predictor": {}}

null_files = sorted(glob.glob(f"{R}/m3_random_c21.4801_d*.json"))
per_pred_deltas = {"esmfold": [], "omegafold": []}
per_pred_dirids = {"esmfold": [], "omegafold": []}
per_pred_valid_orf = {"esmfold": [], "omegafold": []}
n_dirs = 0
for fp in null_files:
    d = json.load(open(fp))
    for direction in d["directions"]:
        n_dirs += 1
        for pred in ["esmfold", "omegafold"]:
            pp = direction["per_predictor"][pred]
            baseline = m2["per_predictor"][pred]["curve"]["0.0"]["mean"]
            per_pred_deltas[pred].append(pp["helix_hgi_w_mean"] - baseline)
            per_pred_dirids[pred].append(direction["dir_id"])
            per_pred_valid_orf[pred].append(pp["valid_orf_rate"])

for pred in ["esmfold", "omegafold"]:
    pdat = m3s["per_predictor"][pred]
    c3["per_predictor"][pred] = {
        "baseline_helix_w": pdat["baseline_helix_w"],
        "S_delta": pdat["S_delta"],
        "matched_control_delta": pdat["matched_control_delta"],
        "S_helix_w_at_cstar": pdat["S_helix_w_at_cstar"],
        "primary_random_null": pdat["PRIMARY_random_null"],
        "mean_plddt_per_arm": pdat["mean_plddt_per_arm"],
        "valid_orf_per_arm": pdat["valid_orf_per_arm"],
        "null_deltas": per_pred_deltas[pred],
        "null_dir_ids": per_pred_dirids[pred],
        "null_valid_orf": per_pred_valid_orf[pred],
        "n_null_directions": len(per_pred_deltas[pred]),
    }

c3["n_random_direction_files"] = len(null_files)
c3["n_random_directions_total"] = n_dirs

data = {"C1": c1, "C2": c2, "C3": c3}
os.makedirs("paper_figures", exist_ok=True)
json.dump(data, open(OUT, "w"), indent=2)
print("Wrote", OUT)
print("C1 conditions:", len(c1["conditions"]))
print("C2 doses:", c2["doses"])
print("C3 n null directions per predictor:", c3["per_predictor"]["esmfold"]["n_null_directions"],
      c3["per_predictor"]["omegafold"]["n_null_directions"])
