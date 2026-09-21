"""
M3 v2 consolidation (C3 fix). Two changes over m3_consolidate.py that address the verify
mechanism-audit FAIL:
  1. LOCK the decisive specificity dose at the mid-plateau alpha=8 (capability-preserved: S valid-ORF
     ~0.89), NOT the grid-max alpha=16 (valid-ORF crashed to 0.667). Uses the already-collected
     m3_*_a8_s*.json samples.
  2. Make S-vs-RANDOM-DIRECTION-NULL the PRIMARY specificity statistic: >=30 independent random
     norm-matched directions (results/m3_random_control_d*.json). Empirical one-sided
     p = (1 + #{delta_rand >= delta_S}) / (N + 1); plus z-score. The single matched_control stays as a
     SECONDARY sanity control. Also reports capability (valid-ORF) for S/matched/random at the locked
     dose, and the beta arm honestly as a NEGATIVE result.
-> results/m3_specificity_summary_v2.json
"""
import os, sys, json, glob
import numpy as np
from scipy import stats

RES = "/data/wanghaoxiong/intergene_mechanist_v6/results"
KINDS = ["alpha_helix_S", "matched_control", "beta_sheet_offtarget"]
LOCKED_DOSE = 8.0
BASE_DOSE = 0.0


def load_arm_runs():
    runs = {}
    for f in glob.glob(os.path.join(RES, "m3_*_a*_s*.json")):
        d = json.load(open(f)); c = d["config"]
        runs.setdefault((c["feature_kind"], c["alpha"]), []).append(d)
    return runs


def arm_cell(runs, kind, alpha):
    ds = runs.get((kind, alpha), [])
    hmeans = [d["helix_hgi"]["helix_hgi_mean"] for d in ds if d["helix_hgi"].get("helix_hgi_mean") is not None]
    smeans = [d["sheet"]["sheet_mean"] for d in ds if d["sheet"].get("sheet_mean") is not None]
    orf = [d["helix_hgi"]["valid_orf_rate"] for d in ds]
    helix_s = [x["helix_hgi"] for d in ds for x in d.get("per_sample", []) if x.get("helix_hgi") is not None]
    sheet_s = [x["sheet"] for d in ds for x in d.get("per_sample", []) if x.get("sheet") is not None]
    return {"helix_mean": float(np.mean(hmeans)) if hmeans else None,
            "sheet_mean": float(np.mean(smeans)) if smeans else None,
            "valid_orf_rate": float(np.mean(orf)) if orf else None,
            "n_seeds": len(ds), "helix_samples": helix_s, "sheet_samples": sheet_s}


def main():
    runs = load_arm_runs()
    cells = {kind: {a: arm_cell(runs, kind, a) for a in [BASE_DOSE, LOCKED_DOSE]} for kind in KINDS}

    base_helix = cells["alpha_helix_S"][BASE_DOSE]["helix_mean"]      # shared baseline (alpha=0)
    base_sheet = cells["alpha_helix_S"][BASE_DOSE]["sheet_mean"]

    d_S_helix = cells["alpha_helix_S"][LOCKED_DOSE]["helix_mean"] - base_helix
    d_matched_helix = cells["matched_control"][LOCKED_DOSE]["helix_mean"] - base_helix
    d_beta_helix = cells["beta_sheet_offtarget"][LOCKED_DOSE]["helix_mean"] - base_helix
    d_S_sheet = cells["alpha_helix_S"][LOCKED_DOSE]["sheet_mean"] - base_sheet
    d_beta_sheet = cells["beta_sheet_offtarget"][LOCKED_DOSE]["sheet_mean"] - base_sheet

    # ---- PRIMARY: random-direction null ----
    rand_dirs = []
    for f in sorted(glob.glob(os.path.join(RES, "m3_random_control_d*.json"))):
        d = json.load(open(f))
        for dd in d["directions"]:
            if dd.get("helix_mean") is not None and dd.get("n_folded_gated", 0) >= 5:
                rand_dirs.append(dd)
    rand_helix = np.array([dd["helix_mean"] for dd in rand_dirs])
    rand_delta = rand_helix - base_helix
    N = len(rand_dirs)
    n_ge = int(np.sum(rand_delta >= d_S_helix))
    emp_p = (1 + n_ge) / (N + 1) if N > 0 else None
    z = float((d_S_helix - rand_delta.mean()) / rand_delta.std(ddof=1)) if N > 1 and rand_delta.std(ddof=1) > 0 else None
    rand_orf = np.array([dd["valid_orf_rate"] for dd in rand_dirs if dd.get("valid_orf_rate") is not None])

    # ---- SECONDARY: S vs matched-control Mann-Whitney (pooled per-sample) at locked dose ----
    s_hi = cells["alpha_helix_S"][LOCKED_DOSE]["helix_samples"]
    m_hi = cells["matched_control"][LOCKED_DOSE]["helix_samples"]
    try:
        S_vs_matched_p = float(stats.mannwhitneyu(s_hi, m_hi, alternative="greater")[1]) if s_hi and m_hi else None
    except Exception:
        S_vs_matched_p = None

    result = {
        "locked_dose": LOCKED_DOSE,
        "locked_dose_rationale": "mid-plateau, capability preserved (S valid-ORF ~0.89 at alpha=8 vs "
                                 "0.667 at alpha=16 where the original stat was taken); addresses "
                                 "mechanism-audit FAIL.",
        "baseline_helix": base_helix, "baseline_sheet": base_sheet,
        "arm_table_at_locked_dose": {
            kind: {"helix_mean": cells[kind][LOCKED_DOSE]["helix_mean"],
                   "sheet_mean": cells[kind][LOCKED_DOSE]["sheet_mean"],
                   "valid_orf_rate": cells[kind][LOCKED_DOSE]["valid_orf_rate"],
                   "n_seeds": cells[kind][LOCKED_DOSE]["n_seeds"]} for kind in KINDS},
        "deltas_vs_baseline": {
            "helix_delta_S": d_S_helix, "helix_delta_matched": d_matched_helix,
            "helix_delta_beta": d_beta_helix, "sheet_delta_S": d_S_sheet, "sheet_delta_beta": d_beta_sheet},
        "PRIMARY_random_direction_null": {
            "n_directions": N,
            "random_helix_delta_mean": float(rand_delta.mean()) if N else None,
            "random_helix_delta_std": float(rand_delta.std(ddof=1)) if N > 1 else None,
            "random_helix_delta_max": float(rand_delta.max()) if N else None,
            "S_helix_delta": d_S_helix,
            "n_random_ge_S": n_ge,
            "empirical_one_sided_p": emp_p,
            "z_score": z,
            "random_valid_orf_mean": float(rand_orf.mean()) if len(rand_orf) else None,
            "S_valid_orf_at_locked": cells["alpha_helix_S"][LOCKED_DOSE]["valid_orf_rate"],
            "capability_note": "S's valid-ORF vs the random-direction null mean is reported so a "
                               "capability-selection artifact can be ruled out."},
        "SECONDARY_matched_control": {
            "S_vs_matched_helix_delta": d_S_helix - d_matched_helix,
            "S_vs_matched_mannwhitney_p_at_locked": S_vs_matched_p},
        "beta_arm_NEGATIVE_result": {
            "sheet_delta_beta": d_beta_sheet,
            "verdict": "NEGATIVE: the beta-sheet off-target feature did NOT amplify sheet at the locked "
                       "dose (delta ~= {:.3f}); reported honestly as a failed manipulation, NOT as half "
                       "of a double dissociation. C3 rests on S-vs-null/matched helix specificity + "
                       "beta-does-not-raise-helix.".format(d_beta_sheet if d_beta_sheet is not None else float('nan'))},
        "specificity_verdict": None,
    }
    # verdict logic
    passed = (emp_p is not None and emp_p < 0.05 and d_S_helix > 0.05
              and abs(d_matched_helix) < 0.05 and abs(d_beta_helix) < 0.05)
    result["specificity_verdict"] = "SUPPORTED (capability-preserved, random-null-controlled)" if passed \
        else "NOT SUPPORTED at locked dose"

    json.dump(result, open(os.path.join(RES, "m3_specificity_summary_v2.json"), "w"), indent=2)
    print("=== M3 v2 specificity (locked dose alpha=%.0f) ===" % LOCKED_DOSE)
    print(json.dumps({k: result[k] for k in ["deltas_vs_baseline", "PRIMARY_random_direction_null",
                                              "SECONDARY_matched_control", "beta_arm_NEGATIVE_result",
                                              "specificity_verdict"]}, indent=2))


if __name__ == "__main__":
    main()
