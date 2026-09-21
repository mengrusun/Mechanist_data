"""
C3 model-swap variant: H-only pLDDT-weighted helix fraction for specificity
Reads pre-computed M3 arm + random null files and computes S-vs-null specificity
using helix_h_w (H-only) instead of helix_hgi_w (HGI, primary endpoint).
No GPU needed.
"""
import json
import glob
import os
import argparse
import numpy as np


def load_m3_arm_data(results_dir, feature_kind, c_sigma_str, seeds=(200, 201)):
    """Load M3 arm result files and extract aggregate helix stats."""
    records = []
    for seed in seeds:
        # Try both exact c_sigma representation
        for c_str in [c_sigma_str, "21.4801", "21.48", "21.4800"]:
            pattern = os.path.join(
                results_dir,
                f"m3_{feature_kind}_c{c_str}_s{seed}.json"
            )
            matches = glob.glob(pattern)
            if matches:
                fpath = matches[0]
                with open(fpath) as f:
                    data = json.load(f)
                for pred_name, pred_data in data.get("per_predictor", {}).items():
                    records.append({
                        "seed": seed,
                        "predictor": pred_name,
                        "helix_hgi_w_mean": pred_data.get("helix_hgi_w_mean"),
                        "helix_h_w_mean": pred_data.get("helix_h_w_mean"),
                        "valid_orf_rate": pred_data.get("valid_orf_rate"),
                        "file": os.path.basename(fpath),
                    })
                break
    return records


def load_m3_baseline(results_dir, seeds=(200, 201)):
    """Load M3 c=0 baseline (S arm at c=0)."""
    records = []
    for seed in seeds:
        for c_str in ["0.0", "0", "0.00"]:
            pattern = os.path.join(
                results_dir,
                f"m3_alpha_helix_S_c{c_str}_s{seed}.json"
            )
            matches = glob.glob(pattern)
            if matches:
                fpath = matches[0]
                with open(fpath) as f:
                    data = json.load(f)
                for pred_name, pred_data in data.get("per_predictor", {}).items():
                    records.append({
                        "seed": seed,
                        "predictor": pred_name,
                        "helix_hgi_w_mean": pred_data.get("helix_hgi_w_mean"),
                        "helix_h_w_mean": pred_data.get("helix_h_w_mean"),
                        "file": os.path.basename(fpath),
                    })
                break
    return records


def load_null_h_only(results_dir):
    """Load M3 random null files and extract per-direction helix_h_w_mean per predictor."""
    null_records = []
    for fpath in sorted(glob.glob(os.path.join(results_dir, "m3_random_c21.4801_d*.json"))):
        with open(fpath) as f:
            data = json.load(f)
        for direction in data.get("directions", []):
            dir_id = direction.get("dir_id")
            for pred_name, pred_data in direction.get("per_predictor", {}).items():
                null_records.append({
                    "dir_id": dir_id,
                    "file": os.path.basename(fpath),
                    "predictor": pred_name,
                    "helix_hgi_w_mean": pred_data.get("helix_hgi_w_mean"),
                    "helix_h_w_mean": pred_data.get("helix_h_w_mean"),
                    "valid_orf_rate": pred_data.get("valid_orf_rate"),
                })
    return null_records


def compute_specificity(s_helix_h_w, baseline_helix_h_w, null_helix_h_w_list,
                         matched_control_helix_h_w=None):
    """Compute S-vs-null specificity statistics for H-only endpoint."""
    s_delta = s_helix_h_w - baseline_helix_h_w
    null_deltas = [h - baseline_helix_h_w for h in null_helix_h_w_list]
    N = len(null_deltas)

    n_null_ge_S = sum(1 for d in null_deltas if d >= s_delta)
    empirical_p = (1 + n_null_ge_S) / (N + 1)
    null_mean = float(np.mean(null_deltas))
    null_std = float(np.std(null_deltas))
    z_score = (s_delta - null_mean) / null_std if null_std > 0 else float("nan")

    result = {
        "S_helix_h_w_at_cstar": s_helix_h_w,
        "baseline_helix_h_w": baseline_helix_h_w,
        "S_delta_h": s_delta,
        "null_delta_mean": null_mean,
        "null_delta_std": null_std,
        "null_delta_max": float(np.max(null_deltas)) if null_deltas else None,
        "n_null_ge_S": n_null_ge_S,
        "n_directions": N,
        "empirical_one_sided_p": empirical_p,
        "z_score": z_score,
    }
    if matched_control_helix_h_w is not None:
        result["matched_control_delta_h"] = matched_control_helix_h_w - baseline_helix_h_w

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[C3 H-only] Loading M3 arm data for S at c*=21.4801 (held-out seeds 200,201)...")
    s_arm = load_m3_arm_data(args.results_dir, "alpha_helix_S", "21.4801", seeds=(200, 201))
    print(f"  S arm records: {len(s_arm)}")

    print("[C3 H-only] Loading baseline (c=0) data...")
    baseline_records = load_m3_baseline(args.results_dir, seeds=(200, 201))
    print(f"  Baseline records: {len(baseline_records)}")

    print("[C3 H-only] Loading matched control at c*...")
    mc_arm = load_m3_arm_data(args.results_dir, "matched_control", "21.4801", seeds=(200, 201))
    print(f"  Matched control records: {len(mc_arm)}")

    print("[C3 H-only] Loading random null files...")
    null_records = load_null_h_only(args.results_dir)
    print(f"  Null direction records: {len(null_records)} (unique dirs per predictor)")

    # Aggregate per predictor
    predictors = sorted(set(r["predictor"] for r in s_arm))
    per_predictor = {}

    for pred in predictors:
        # S arm at c*: average over held-out seeds
        s_vals = [r["helix_h_w_mean"] for r in s_arm
                  if r["predictor"] == pred and r["helix_h_w_mean"] is not None]
        if not s_vals:
            print(f"  WARNING: No H-only S arm data for predictor {pred}")
            continue
        s_helix_h_w = float(np.mean(s_vals))

        # Baseline at c=0: average over held-out seeds
        base_vals = [r["helix_h_w_mean"] for r in baseline_records
                     if r["predictor"] == pred and r["helix_h_w_mean"] is not None]
        if not base_vals:
            print(f"  WARNING: No H-only baseline data for predictor {pred}")
            continue
        baseline_h_w = float(np.mean(base_vals))

        # Matched control at c*
        mc_vals = [r["helix_h_w_mean"] for r in mc_arm
                   if r["predictor"] == pred and r["helix_h_w_mean"] is not None]
        mc_h_w = float(np.mean(mc_vals)) if mc_vals else None

        # Null directions: per-direction helix_h_w_mean
        null_vals = [r["helix_h_w_mean"] for r in null_records
                     if r["predictor"] == pred and r["helix_h_w_mean"] is not None]

        print(f"\n  [{pred}]")
        print(f"    S helix_h_w at c*: {s_helix_h_w:.4f}")
        print(f"    baseline helix_h_w: {baseline_h_w:.4f}")
        print(f"    S delta_h: {s_helix_h_w - baseline_h_w:.4f}")
        print(f"    n_null_directions: {len(null_vals)}")
        if mc_h_w is not None:
            print(f"    matched_control delta_h: {mc_h_w - baseline_h_w:.4f}")

        stats = compute_specificity(
            s_helix_h_w, baseline_h_w, null_vals,
            matched_control_helix_h_w=mc_h_w
        )
        print(f"    n_null_ge_S: {stats['n_null_ge_S']}/{stats['n_directions']}")
        print(f"    empirical_p: {stats['empirical_one_sided_p']:.4f}")
        print(f"    z_score: {stats['z_score']:.3f}")

        per_predictor[pred] = stats

    # Claim supported if: 0/N null dirs >= S under BOTH predictors, z>=2.0 both
    n_null_ge_S_vals = [s["n_null_ge_S"] for s in per_predictor.values()]
    z_scores = [s["z_score"] for s in per_predictor.values()]
    empirical_ps = [s["empirical_one_sided_p"] for s in per_predictor.values()]

    all_null_ge_S_zero = all(v == 0 for v in n_null_ge_S_vals)
    all_z_above_2 = all(z >= 2.0 for z in z_scores if not np.isnan(z))
    all_p_sig = all(p < 0.05 for p in empirical_ps)
    dual_agree = len(per_predictor) == 2

    claim_supported = all_null_ge_S_zero and all_z_above_2 and all_p_sig and dual_agree

    result = {
        "variant": "model-swap-h-only-specificity",
        "dimension": "model",
        "claim_id": "C3",
        "swap_description": "H-only pLDDT-weighted helix fraction (helix_h_w) as specificity endpoint instead of HGI (helix_hgi_w)",
        "endpoint": "helix_h_w (pLDDT-weighted H-only strict alpha helix)",
        "per_predictor": per_predictor,
        "all_null_ge_S_zero": all_null_ge_S_zero,
        "all_z_above_2": all_z_above_2,
        "all_p_significant": all_p_sig,
        "dual_predictor_agree": dual_agree,
        "claim_supported_h_only": claim_supported,
        "n_null_records": len(null_records),
        "note": (
            "All data from pre-computed M3 arm + null files (helix_h_w_mean stored per run). "
            "No GPU re-run. Empirical p = (1+n_null_ge_S)/(N+1); no cluster-bootstrap CI available "
            "for H-only (per-sample H-only not stored) — aggregate-level test only."
        ),
    }

    out_path = os.path.join(args.out_dir, "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[C3 H-only] result.json written to {out_path}")
    print(f"  claim_supported_h_only = {claim_supported}")


if __name__ == "__main__":
    main()
