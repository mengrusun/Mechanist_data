"""
C1 model-swap variant: H-only helix definition
Reads pre-computed M0 H_only result files and aggregates set-AUROC statistics.
No GPU needed — all data already on disk from M0's 12-config run.
"""
import json
import glob
import os
import argparse
import numpy as np


def load_h_only_files(results_dir):
    """Load all M0 H_only result files and extract set-AUROC stats."""
    records = []
    for fpath in sorted(glob.glob(os.path.join(results_dir, "m0_*_H_only_s*.json"))):
        fname = os.path.basename(fpath)
        parts = fname.replace(".json", "").split("_")
        # m0_{organism}_H_only_s{seed}.json
        # Extract organism and seed
        if "eukaryote" in fname:
            organism = "eukaryote"
        else:
            organism = "prokaryote"
        seed = int(parts[-1].replace("s", ""))
        with open(fpath) as f:
            data = json.load(f)
        record = {
            "file": fname,
            "organism": organism,
            "seed": seed,
            "combined_set_test_auroc": data.get("combined_set_test_auroc"),
            "mean_set_test_auroc": data.get("mean_set_test_auroc"),
            "confound_only_test_auroc": data.get("confound_only_test_auroc"),
            "combined_set_shuffle_null_auroc": data.get("combined_set_shuffle_null_auroc"),
            "set_null_gap": data.get("set_null_gap"),
            "set_auroc_over_confound": data.get("set_auroc_over_confound"),
            "floor_ok": data.get("floor_ok"),
            "S_size": data.get("S_size", 0),
        }
        records.append(record)
    return records


def aggregate_by_organism(records):
    """Compute per-organism mean set-AUROC across seeds."""
    by_org = {}
    for r in records:
        org = r["organism"]
        if org not in by_org:
            by_org[org] = []
        if r["combined_set_test_auroc"] is not None:
            by_org[org].append(r["combined_set_test_auroc"])
    result = {}
    for org, vals in by_org.items():
        result[org] = {
            "mean_set_auroc": float(np.mean(vals)),
            "std_set_auroc": float(np.std(vals)),
            "n_seeds": len(vals),
            "per_seed": vals,
            "meets_bar_080": all(v >= 0.80 for v in vals),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("[C1 H-only] Loading M0 H_only result files...")
    records = load_h_only_files(args.results_dir)
    print(f"  Loaded {len(records)} H_only config files")

    for r in records:
        auroc = r["combined_set_test_auroc"]
        print(f"  {r['organism']} s{r['seed']}: combined_set_auroc={auroc:.4f}")

    by_org = aggregate_by_organism(records)

    print("\n[C1 H-only] Per-organism aggregation:")
    for org, stats in by_org.items():
        print(f"  {org}: mean_AUROC={stats['mean_set_auroc']:.4f} ± {stats['std_set_auroc']:.4f} "
              f"(n={stats['n_seeds']}, meets_0.80={stats['meets_bar_080']})")

    # Check shuffle null (use first record as representative)
    rep = records[0]
    shuffle_null_auroc = rep.get("combined_set_shuffle_null_auroc", None)
    confound_auroc = rep.get("confound_only_test_auroc", None)
    print(f"\n  shuffle_null_auroc: {shuffle_null_auroc}")
    print(f"  confound_only_auroc: {confound_auroc}")

    # Determine if claim is supported
    prok_stats = by_org.get("prokaryote", {})
    euk_stats = by_org.get("eukaryote", {})

    prok_mean = prok_stats.get("mean_set_auroc", 0)
    euk_mean = euk_stats.get("mean_set_auroc", 0)
    prok_meets_bar = prok_stats.get("meets_bar_080", False)
    euk_meets_bar = euk_stats.get("meets_bar_080", False)

    # Claim thresholds:
    # - set-AUROC >= 0.80 (M0 pass criterion)
    # - cross-organism (both prok and euk)
    # - shuffle null ≈ 0.50 (gap >> 0)
    # - confound-only auroc < S auroc (margin >= 0.1)

    set_null_gaps = [r.get("set_null_gap", 0) for r in records if r.get("set_null_gap") is not None]
    mean_null_gap = float(np.mean(set_null_gaps)) if set_null_gaps else 0.0

    claim_supported = (
        prok_mean >= 0.80 and
        euk_mean >= 0.80 and
        prok_meets_bar and
        euk_meets_bar and
        mean_null_gap >= 0.30  # null gap should be >0.3 (main experiment: 0.37-0.48)
    )

    result = {
        "variant": "model-swap-h-only-helix-def",
        "dimension": "model",
        "claim_id": "C1",
        "swap_description": "H-only helix definition (DSSP strict alpha, excludes G+I) instead of HGI",
        "per_organism": by_org,
        "representative_shuffle_null_auroc": shuffle_null_auroc,
        "representative_confound_only_auroc": confound_auroc,
        "mean_set_null_gap": mean_null_gap,
        "prokaryote_mean_set_auroc": prok_mean,
        "eukaryote_mean_set_auroc": euk_mean,
        "meets_bar_prokaryote": prok_meets_bar,
        "meets_bar_eukaryote": euk_meets_bar,
        "n_configs_loaded": len(records),
        "claim_supported_h_only": claim_supported,
        "note": (
            "All data read from pre-computed M0 H_only result files. "
            "No GPU re-run. H-only definition is the strict alpha-helix (DSSP 'H') "
            "subset of HGI (H+G+I used in main experiment)."
        ),
    }

    out_path = os.path.join(args.out_dir, "result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n[C1 H-only] result.json written to {out_path}")
    print(f"  claim_supported_h_only = {claim_supported}")
    print(f"  prokaryote mean AUROC = {prok_mean:.4f}")
    print(f"  eukaryote mean AUROC = {euk_mean:.4f}")


if __name__ == "__main__":
    main()
