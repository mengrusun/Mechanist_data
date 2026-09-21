"""
C2 model-swap variant: H-only pLDDT-weighted helix fraction
Reads pre-computed M2 result files and computes Spearman trend on helix_h_w
(H-only pLDDT-weighted helix fraction) instead of helix_hgi_w (HGI, primary endpoint).
No GPU needed.
"""
import json
import glob
import os
import argparse
import numpy as np
from scipy.stats import spearmanr


def load_m2_files(results_dir):
    """Load all M2 alpha_helix_S result files."""
    records = []
    for fpath in sorted(glob.glob(os.path.join(results_dir, "m2_alpha_helix_S_c*_s*.json"))):
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            data = json.load(f)
        config = data.get("config", {})
        c_sigma = config.get("c_sigma")
        seed = config.get("seed")
        if c_sigma is None or seed is None:
            # Try parsing from filename: m2_alpha_helix_S_c{c}_s{seed}.json
            parts = fname.replace(".json", "").split("_")
            for p in parts:
                if p.startswith("c") and c_sigma is None:
                    try:
                        c_sigma = float(p[1:])
                    except ValueError:
                        pass
                if p.startswith("s") and seed is None:
                    try:
                        seed = int(p[1:])
                    except ValueError:
                        pass
        for pred_name, pred_data in data.get("per_predictor", {}).items():
            record = {
                "file": fname,
                "c_sigma": c_sigma,
                "seed": seed,
                "predictor": pred_name,
                # Primary endpoint (main experiment)
                "helix_hgi_w_mean": pred_data.get("helix_hgi_w_mean"),
                # Swap endpoint (H-only)
                "helix_h_w_mean": pred_data.get("helix_h_w_mean"),
                "valid_orf_rate": pred_data.get("valid_orf_rate"),
            }
            records.append(record)
    return records


def compute_spearman_on_heldout(records, predictor, heldout_seeds=(200, 201),
                                 capability_threshold=0.85):
    """Compute Spearman rho on H-only helix fraction for held-out seeds."""
    # Filter to held-out seeds and given predictor
    filtered = [r for r in records
                if r["seed"] in heldout_seeds
                and r["predictor"] == predictor
                and r["helix_h_w_mean"] is not None]

    if not filtered:
        return None

    # Average over held-out seeds per dose
    by_dose = {}
    for r in filtered:
        c = r["c_sigma"]
        if c not in by_dose:
            by_dose[c] = []
        by_dose[c].append(r["helix_h_w_mean"])

    # Compute mean per dose
    dose_means = {c: float(np.mean(v)) for c, v in by_dose.items()}

    # Sort by dose
    doses_sorted = sorted(dose_means.keys())
    helix_values = [dose_means[c] for c in doses_sorted]

    if len(doses_sorted) < 3:
        return None

    # Spearman over all doses
    rho, p = spearmanr(doses_sorted, helix_values)

    # Also compute delta at c*=21.48 vs c=0
    c_star = 21.4801
    c0 = 0.0
    helix_at_cstar = dose_means.get(c_star) or dose_means.get(21.48)
    helix_at_c0 = dose_means.get(c0)

    delta_at_cstar = None
    if helix_at_cstar is not None and helix_at_c0 is not None:
        delta_at_cstar = helix_at_cstar - helix_at_c0

    return {
        "predictor": predictor,
        "heldout_seeds": list(heldout_seeds),
        "doses_used": doses_sorted,
        "helix_h_w_per_dose": dose_means,
        "spearman_rho": float(rho),
        "spearman_p": float(p),
        "delta_at_cstar_vs_c0": delta_at_cstar,
        "n_doses": len(doses_sorted),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[C2 H-only] Loading M2 result files...")
    records = load_m2_files(args.results_dir)
    print(f"  Loaded {len(records)} predictor-dose-seed records")

    predictors = sorted(set(r["predictor"] for r in records))
    print(f"  Predictors: {predictors}")

    per_predictor = {}
    for pred in predictors:
        stats = compute_spearman_on_heldout(records, pred, heldout_seeds=(200, 201))
        if stats is not None:
            per_predictor[pred] = stats
            print(f"\n  [{pred}] H-only Spearman rho = {stats['spearman_rho']:.4f}, p = {stats['spearman_p']:.4f}")
            print(f"  [{pred}] delta at c*=21.48 vs c=0 = {stats['delta_at_cstar_vs_c0']}")

    # Determine if claim is supported under H-only
    all_significant = all(s["spearman_p"] < 0.05 for s in per_predictor.values())
    all_positive = all(s["spearman_rho"] > 0 for s in per_predictor.values())
    all_delta_positive = all(
        (s["delta_at_cstar_vs_c0"] is not None and s["delta_at_cstar_vs_c0"] > 0.05)
        for s in per_predictor.values()
    )
    dual_agree = len(per_predictor) == 2 and all_positive

    claim_supported = all_significant and all_positive and all_delta_positive and dual_agree

    result = {
        "variant": "model-swap-h-only-helix-frac",
        "dimension": "model",
        "claim_id": "C2",
        "swap_description": "H-only pLDDT-weighted helix fraction (helix_h_w) instead of HGI pLDDT-weighted (helix_hgi_w)",
        "endpoint": "helix_h_w (pLDDT-weighted H-only, strict alpha helix only)",
        "per_predictor": per_predictor,
        "all_significant_p05": all_significant,
        "all_positive_rho": all_positive,
        "all_delta_positive": all_delta_positive,
        "dual_predictor_agree_on_sign": dual_agree,
        "claim_supported_h_only": claim_supported,
        "n_records": len(records),
        "note": (
            "All data from pre-computed M2 result files (helix_h_w_mean stored per run). "
            "No GPU re-run. Spearman rho computed on held-out seeds 200+201."
        ),
    }

    out_path = os.path.join(args.out_dir, "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[C2 H-only] result.json written to {out_path}")
    print(f"  claim_supported_h_only = {claim_supported}")


if __name__ == "__main__":
    main()
