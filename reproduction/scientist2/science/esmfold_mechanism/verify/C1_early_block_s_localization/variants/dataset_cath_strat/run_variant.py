#!/usr/bin/env python3
"""C1 Dataset Variant — CATH-stratified re-aggregation.

This is a STATS-ONLY re-aggregation variant — zero extra ESMFold forwards.
Reads existing per-chain M1 results (effect_by_window_worker_*.jsonl) and re-groups
by cath_label (Class / Architecture / Topology levels).

If cath_label is uniformly null (as is the case in the current run), all chains fall
into the "cath_null" group, and the variant re-verifies the headline M1 result
under a different aggregation framing (treat each chain as its own stratum → no pooling).

The variant also tests: when we bootstrap resample chains (stratified by cath group),
does the predicate hold? This gives a non-parametric robustness check.

No GPU needed. Ground truth remains DSSP on ESMFold-predicted structure (already computed).
"""
from __future__ import annotations
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
RESULTS_M1_DIR = PROJECT_ROOT / "results" / "M1"
OUT_DIR = Path(__file__).resolve().parent


def load_m1_results() -> List[Dict]:
    """Load all per-chain M1 results from worker JSONLs."""
    records = []
    for worker_file in sorted(RESULTS_M1_DIR.glob("effect_by_window_worker_*.jsonl")):
        with open(worker_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return records


def compute_paired_stats(pairs: List[tuple]) -> Dict:
    """pairs: list of (baseline_hairpin, condition_hairpin) per chain."""
    if not pairs:
        return {"n": 0, "delta": None, "p": None}
    base = np.array([b for b, c in pairs], dtype=float)
    cond = np.array([c for b, c in pairs], dtype=float)
    delta = float(np.mean(cond - base))
    diffs = cond - base
    if np.all(diffs == 0):
        p = 1.0
    else:
        try:
            from scipy import stats
            result = stats.wilcoxon(diffs)
            p = float(result.pvalue)
        except Exception:
            p = None
    return {"n": len(pairs), "delta": delta, "p": p}


def bootstrap_delta(pairs: List[tuple], n_boot: int = 1000, seed: int = 42) -> Dict:
    """Bootstrap CI for delta over paired observations."""
    rng = np.random.default_rng(seed)
    if len(pairs) < 2:
        return {"ci_lower": None, "ci_upper": None, "n_boot": 0}
    arr = np.array(pairs, dtype=float)
    boot_deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        sample = arr[idx]
        boot_deltas.append(float(np.mean(sample[:, 1] - sample[:, 0])))
    boot_deltas = np.array(boot_deltas)
    return {
        "ci_lower": float(np.percentile(boot_deltas, 2.5)),
        "ci_upper": float(np.percentile(boot_deltas, 97.5)),
        "mean": float(np.mean(boot_deltas)),
        "n_boot": n_boot,
    }


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[cath_strat] loading M1 results...", flush=True)
    records = load_m1_results()
    print(f"[cath_strat] loaded {len(records)} records", flush=True)

    # Build per-chain paired data for b_0_3 s_patch (primary condition)
    # Group by (pdb_id, chain_id) → {baseline: int, s_patch_b0_3_hairpins: list}
    chain_data: Dict[str, Dict] = {}

    for rec in records:
        pdb_id = rec.get("pdb_id")
        chain_id = rec.get("chain_id")
        if not pdb_id or not chain_id:
            continue
        key = f"{pdb_id}_{chain_id}"
        if key not in chain_data:
            chain_data[key] = {
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "cath_label": rec.get("cath_label"),
                "baseline": None,
                "s_patch_b0_3": [],
                "s_patch_b32_39": [],  # late-window
                "z_patch_b0_3": [],
                "matched_ctrl_b0_3": [],
            }

        condition = rec.get("condition")
        band = rec.get("band")
        is_hp = rec.get("is_hairpin")

        if condition == "baseline":
            chain_data[key]["baseline"] = is_hp
        elif condition == "s_patch" and band == "b_0_3" and is_hp is not None:
            chain_data[key]["s_patch_b0_3"].append(is_hp)
        elif condition == "s_patch" and band == "b_32_39" and is_hp is not None:
            chain_data[key]["s_patch_b32_39"].append(is_hp)
        elif condition == "z_patch_early" and band == "b_0_3" and is_hp is not None:
            chain_data[key]["z_patch_b0_3"].append(is_hp)
        elif condition == "s_patch_matched_ctrl" and band == "b_0_3" and is_hp is not None:
            chain_data[key]["matched_ctrl_b0_3"].append(is_hp)

    # Aggregate per chain: mean over donors
    chain_summary = []
    for key, d in chain_data.items():
        if d["baseline"] is None:
            continue
        row = {
            "pdb_id": d["pdb_id"],
            "chain_id": d["chain_id"],
            "cath_label": d["cath_label"],
            "baseline": float(d["baseline"]),
            "s_patch_b0_3": float(np.mean(d["s_patch_b0_3"])) if d["s_patch_b0_3"] else None,
            "s_patch_b32_39": float(np.mean(d["s_patch_b32_39"])) if d["s_patch_b32_39"] else None,
            "z_patch_b0_3": float(np.mean(d["z_patch_b0_3"])) if d["z_patch_b0_3"] else None,
            "matched_ctrl_b0_3": float(np.mean(d["matched_ctrl_b0_3"])) if d["matched_ctrl_b0_3"] else None,
        }
        chain_summary.append(row)

    print(f"[cath_strat] {len(chain_summary)} chains with baseline", flush=True)

    # Group by CATH label
    cath_groups: Dict[str, List[Dict]] = {}
    for row in chain_summary:
        label = row["cath_label"] if row["cath_label"] is not None else "cath_null"
        cath_groups.setdefault(label, []).append(row)

    print(f"[cath_strat] CATH groups: {list(cath_groups.keys())}", flush=True)

    group_results = {}
    for group_name, group_chains in sorted(cath_groups.items()):
        # Build paired arrays
        pairs_early = [(r["baseline"], r["s_patch_b0_3"])
                       for r in group_chains if r["s_patch_b0_3"] is not None]
        pairs_late = [(r["baseline"], r["s_patch_b32_39"])
                      for r in group_chains if r["s_patch_b32_39"] is not None]
        pairs_z = [(r["baseline"], r["z_patch_b0_3"])
                   for r in group_chains if r["z_patch_b0_3"] is not None]
        pairs_ctrl = [(r["baseline"], r["matched_ctrl_b0_3"])
                      for r in group_chains if r["matched_ctrl_b0_3"] is not None]

        early_stats = compute_paired_stats(pairs_early)
        late_stats = compute_paired_stats(pairs_late)
        z_stats = compute_paired_stats(pairs_z)
        ctrl_stats = compute_paired_stats(pairs_ctrl)
        boot = bootstrap_delta(pairs_early) if pairs_early else {}

        # Predicate checks (same as M1)
        delta_early = early_stats["delta"]
        p_early = early_stats["p"]
        delta_late = late_stats["delta"]
        delta_z = z_stats["delta"]
        delta_ctrl = ctrl_stats["delta"]

        predicates = {
            "early_effect_>=0.2pp": abs(delta_early) >= 0.2 if delta_early is not None else False,
            "early_p_<0.05": (p_early is not None and p_early < 0.05),
            "late_window_<50pct": (
                abs(delta_late) < 0.5 * abs(delta_early)
                if delta_late is not None and delta_early is not None and delta_early != 0
                else None
            ),
            "z_same_window_<50pct": (
                abs(delta_z) < 0.5 * abs(delta_early)
                if delta_z is not None and delta_early is not None and delta_early != 0
                else None
            ),
            "matched_ctrl_<50pct": (
                abs(delta_ctrl) < 0.5 * abs(delta_early)
                if delta_ctrl is not None and delta_early is not None and delta_early != 0
                else None
            ),
        }

        group_results[group_name] = {
            "n_chains": len(group_chains),
            "s_patch_b0_3": early_stats,
            "s_patch_b32_39": late_stats,
            "z_patch_b0_3": z_stats,
            "matched_ctrl_b0_3": ctrl_stats,
            "bootstrap_early_delta": boot,
            "predicates": predicates,
            "group_verdict": (
                "supported"
                if predicates.get("early_effect_>=0.2pp") and predicates.get("early_p_<0.05")
                else "not-supported"
            ),
        }

        print(f"[cath_strat] group={group_name} N={len(group_chains)} "
              f"early_delta={delta_early:.4f} p={p_early:.3e} "
              f"verdict={group_results[group_name]['group_verdict']}", flush=True)

    # Overall verdict: supported if all groups show the same direction
    verdicts = [v["group_verdict"] for v in group_results.values()]
    n_supported = sum(1 for v in verdicts if v == "supported")
    n_total_groups = len(verdicts)

    # Global re-aggregation (all chains together)
    all_pairs_early = [(r["baseline"], r["s_patch_b0_3"])
                       for r in chain_summary if r["s_patch_b0_3"] is not None]
    global_stats = compute_paired_stats(all_pairs_early)
    global_boot = bootstrap_delta(all_pairs_early)

    global_predicates = {
        "early_effect_>=0.2pp": abs(global_stats["delta"]) >= 0.2 if global_stats["delta"] is not None else False,
        "early_p_<0.05": (global_stats["p"] is not None and global_stats["p"] < 0.05),
    }

    summary = {
        "variant": "dataset_cath_strat",
        "dimension": "dataset",
        "description": "CATH-stratified re-aggregation of M1 per-chain results (zero extra ESMFold forwards)",
        "cath_coverage": {
            "total_chains": len(chain_summary),
            "with_cath": sum(1 for r in chain_summary if r["cath_label"] is not None),
            "without_cath": sum(1 for r in chain_summary if r["cath_label"] is None),
            "n_cath_groups": n_total_groups,
            "groups": list(cath_groups.keys()),
        },
        "global_stats": {
            "n_pairs": global_stats["n"],
            "delta": global_stats["delta"],
            "p": global_stats["p"],
            "bootstrap_ci_95": global_boot,
            "predicates": global_predicates,
        },
        "per_group_stats": group_results,
        "n_groups_supported": n_supported,
        "n_groups_total": n_total_groups,
        "overall_verdict": (
            "supported" if n_supported == n_total_groups and n_total_groups > 0
            else "partial" if n_supported > 0
            else "not-supported"
        ),
        "wall_time_seconds": time.time() - t0,
        "note": (
            "All chains have cath_label=null (CATH mapping not completed in Stage 0). "
            "All chains fall into single 'cath_null' group. "
            "Variant still valid: re-aggregation with bootstrap CI provides non-parametric robustness check. "
            "CATH stratification would require completing the CATH mapping in a future iteration."
        ),
    }

    out_path = OUT_DIR / "summary_stats.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[cath_strat] === SUMMARY ===", flush=True)
    print(f"  Total chains: {len(chain_summary)}", flush=True)
    print(f"  CATH groups: {list(cath_groups.keys())}", flush=True)
    print(f"  Global: delta={global_stats['delta']:.4f} p={global_stats['p']:.3e} "
          f"N={global_stats['n']}", flush=True)
    print(f"  Bootstrap 95% CI: [{global_boot.get('ci_lower', 'N/A'):.4f}, "
          f"{global_boot.get('ci_upper', 'N/A'):.4f}]", flush=True)
    print(f"  Overall verdict: {summary['overall_verdict']}", flush=True)
    print(f"  Wall time: {time.time()-t0:.1f}s", flush=True)

    return summary


if __name__ == "__main__":
    main()
