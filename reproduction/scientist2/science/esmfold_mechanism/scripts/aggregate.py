#!/usr/bin/env python3
"""Aggregate per-worker JSONL outputs from M1/M2/M3b into summary statistics + verdicts.

Usage:
  python aggregate.py --milestone M1
  python aggregate.py --milestone M2
  python aggregate.py --milestone M3b
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats
from scipy.stats import wilcoxon
try:
    from statsmodels.stats.contingency_tables import mcnemar
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir))
from esmfold_lib import PROJECT_ROOT, write_json  # noqa: E402


def load_jsonl(paths: List[str]) -> List[Dict]:
    out = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                out.append(json.loads(line))
    return out


def paired_mcnemar(x: List[int], y: List[int]) -> Dict:
    """McNemar test on two paired binary vectors x, y (same length, same chain order).
    Returns p-value + effect size = mean(y) - mean(x)."""
    if not HAS_STATSMODELS:
        # Fall back to Wilcoxon signed-rank
        try:
            stat, p = wilcoxon(x, y, zero_method="wilcox")
        except ValueError:
            stat, p = float("nan"), 1.0
        return {"test": "wilcoxon", "stat": float(stat), "p": float(p),
                "effect_size": float(np.mean(y) - np.mean(x))}
    a = np.array(x)
    b = np.array(y)
    n11 = int(((a == 1) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n00 = int(((a == 0) & (b == 0)).sum())
    if n10 + n01 == 0:
        return {"test": "mcnemar", "p": 1.0, "effect_size": 0.0,
                "table": [[n11, n10], [n01, n00]]}
    m = mcnemar([[n11, n10], [n01, n00]], exact=(n10 + n01 < 25))
    return {"test": "mcnemar", "stat": float(m.statistic), "p": float(m.pvalue),
            "effect_size": float(np.mean(b) - np.mean(a)),
            "table": [[n11, n10], [n01, n00]]}


def paired_wilcoxon_proportion(x_props: List[float], y_props: List[float]) -> Dict:
    """Wilcoxon signed-rank on paired per-chain proportions (mean-over-donors ∈ [0,1]).
    Preserves the count-of-donors information without pretending each chain is a single
    Bernoulli. Returns effect = mean(y) - mean(x).
    """
    if len(x_props) < 3:
        return {"test": "wilcoxon", "stat": float("nan"), "p": 1.0,
                "effect_size": float(np.mean(y_props) - np.mean(x_props)) if y_props and x_props else 0.0,
                "n_pairs": len(x_props)}
    diffs = np.array(y_props) - np.array(x_props)
    # If all diffs are zero, Wilcoxon fails
    if np.all(diffs == 0):
        return {"test": "wilcoxon", "stat": 0.0, "p": 1.0,
                "effect_size": 0.0, "n_pairs": len(x_props)}
    try:
        stat, p = wilcoxon(x_props, y_props, zero_method="wilcox")
    except ValueError:
        stat, p = float("nan"), 1.0
    return {"test": "wilcoxon", "stat": float(stat), "p": float(p),
            "effect_size": float(np.mean(y_props) - np.mean(x_props)),
            "n_pairs": len(x_props),
            "base_mean_prop": float(np.mean(x_props)),
            "condition_mean_prop": float(np.mean(y_props))}


def _ok(r: Dict) -> bool:
    """Return True if row's status is ok AND is_hairpin is a valid int."""
    if r.get("status") is not None and r["status"] != "ok":
        return False
    v = r.get("is_hairpin")
    return v is not None and v in (0, 1)


def aggregate_m1() -> Dict:
    out_dir = PROJECT_ROOT / "results" / "M1"
    files = sorted(glob.glob(str(out_dir / "effect_by_window_worker_*.jsonl")))
    records = load_jsonl(files)
    print(f"[agg-M1] loaded {len(records)} rows from {len(files)} workers", flush=True)

    # Organize by (chain, condition, band) with donor-level granularity, ok-only rows
    per_chain: Dict[str, Dict] = {}
    for r in records:
        cid = f"{r['pdb_id']}_{r['chain_id']}"
        per_chain.setdefault(cid, {"baseline": None, "conditions": defaultdict(list),
                                    "cath_label": r.get("cath_label")})
        if not _ok(r):
            continue
        if r["condition"] == "baseline":
            per_chain[cid]["baseline"] = r["is_hairpin"]
        else:
            key = (r["condition"], r["band"])
            per_chain[cid]["conditions"][key].append(r["is_hairpin"])

    all_keys = set()
    for c in per_chain.values():
        all_keys.update(c["conditions"].keys())
    all_keys = sorted(all_keys)

    per_condition_stats: List[Dict] = []
    # Group-level stats per (condition, band) — use proportion + Wilcoxon signed-rank
    for (cond, band) in all_keys:
        base_props = []
        cond_props = []
        for c in per_chain.values():
            if c["baseline"] is None or not c["conditions"].get((cond, band)):
                continue
            # Baseline is one boolean (a single forward) → proportion = value in {0, 1}
            base_props.append(float(c["baseline"]))
            # Condition: mean over donors as proportion ∈ {0, 1/3, 2/3, 1}
            cond_props.append(float(np.mean(c["conditions"][(cond, band)])))
        if len(base_props) < 10:
            continue
        test = paired_wilcoxon_proportion(base_props, cond_props)
        per_condition_stats.append({
            "condition": cond, "band": band,
            "n_chains": len(base_props),
            "baseline_rate": float(np.mean(base_props)),
            "condition_rate": float(np.mean(cond_props)),
            "delta_rate": float(np.mean(cond_props) - np.mean(base_props)),
            "test": test,
        })

    # Pick early_window: band with LARGEST |delta| among condition=s_patch on bands b_0_3..b_16_23
    early_band = None
    max_neg_delta = 0.0
    for row in per_condition_stats:
        if row["condition"] == "s_patch" and row["band"] in {"b_0_3", "b_4_7", "b_8_11", "b_12_15"}:
            if row["delta_rate"] < max_neg_delta:  # most negative
                max_neg_delta = row["delta_rate"]
                early_band = row["band"]
    band_to_blocks = {"b_0_3": list(range(0, 4)), "b_4_7": list(range(4, 8)),
                      "b_8_11": list(range(8, 12)), "b_12_15": list(range(12, 16)),
                      "b_16_23": list(range(16, 24)), "b_24_31": list(range(24, 32)),
                      "b_32_39": list(range(32, 40)), "b_40_47": list(range(40, 48))}
    if early_band is None:
        # Fallback
        early_band = "b_0_3"
    early_blocks = band_to_blocks[early_band]

    # Verdict for Claim 1:
    #   PASS if:
    #     - early s_patch band Δ ≤ -0.2 pp and paired p < 0.05
    #     - matched-control at same band |Δ| < 50% of early s_patch |Δ|
    #     - late-window s_patch |Δ| < 50% of early s_patch |Δ|
    #     - same-window z_patch |Δ| < 50% of early s_patch |Δ|
    def _row(cond, band):
        for r in per_condition_stats:
            if r["condition"] == cond and r["band"] == band:
                return r
        return None
    early_row = _row("s_patch", early_band)
    late_row_1 = _row("s_patch", "b_32_39") or _row("s_patch", "b_40_47")
    z_row = _row("z_patch_early", early_band)
    matched_row = _row("s_patch_matched_ctrl", early_band)

    def _delta(r): return abs(r["delta_rate"]) if r else 0.0
    early_delta = _delta(early_row)
    passes = {
        "early_effect_>=0.2pp":
            early_row is not None and early_row["delta_rate"] <= -0.2,
        "early_p_<0.05":
            early_row is not None and early_row["test"]["p"] < 0.05,
        "matched_ctrl_<50pct":
            matched_row is None or _delta(matched_row) < 0.5 * early_delta,
        "late_window_<50pct":
            late_row_1 is None or _delta(late_row_1) < 0.5 * early_delta,
        "z_same_window_<50pct":
            z_row is None or _delta(z_row) < 0.5 * early_delta,
    }
    verdict = "supported" if all(passes.values()) else \
              ("partial" if passes["early_effect_>=0.2pp"] and passes["early_p_<0.05"]
               else "not-supported")

    summary = {
        "milestone": "M1", "claim": "C1",
        "n_chains_paired": (early_row["n_chains"] if early_row else 0),
        "n_chains_total": len(per_chain),
        "early_band": early_band,
        "early_blocks": early_blocks,
        "early_delta_rate": early_row["delta_rate"] if early_row else None,
        "early_p": early_row["test"]["p"] if early_row else None,
        "per_condition_stats": per_condition_stats,
        "predicate_checks": passes,
        "verdict": verdict,
    }
    write_json(out_dir / "summary_stats.json", summary)
    # Emit the "early_window" file that M2 / M3b read
    write_json(out_dir / "early_window.json", {
        "band": early_band, "blocks": early_blocks,
        "delta_rate": early_row["delta_rate"] if early_row else None,
        "p": early_row["test"]["p"] if early_row else None,
    })
    print(f"[agg-M1] verdict={verdict}  early_band={early_band}  Δ={early_row['delta_rate'] if early_row else None}",
          flush=True)
    return summary


def aggregate_m2() -> Dict:
    out_dir = PROJECT_ROOT / "results" / "M2"
    files = sorted(glob.glob(str(out_dir / "pathway_effect_worker_*.jsonl")))
    records = load_jsonl(files)
    print(f"[agg-M2] loaded {len(records)} rows", flush=True)

    per_chain: Dict[str, Dict] = {}
    for r in records:
        cid = f"{r['pdb_id']}_{r['chain_id']}"
        per_chain.setdefault(cid, {"baseline": None, "conditions": defaultdict(list),
                                    "cath_label": r.get("cath_label")})
        if not _ok(r):
            continue
        if r["condition"] == "baseline":
            per_chain[cid]["baseline"] = r["is_hairpin"]
        else:
            per_chain[cid]["conditions"][r["condition"]].append(r["is_hairpin"])

    per_condition_stats = []
    all_conds = ["seq2pair_donor", "pair2seq_donor", "seq2pair_zero", "seq2pair_matched_ctrl"]
    for cond in all_conds:
        base_props = []
        cond_props = []
        for c in per_chain.values():
            if c["baseline"] is None or not c["conditions"].get(cond):
                continue
            base_props.append(float(c["baseline"]))
            cond_props.append(float(np.mean(c["conditions"][cond])))
        if len(base_props) < 10:
            continue
        test = paired_wilcoxon_proportion(base_props, cond_props)
        per_condition_stats.append({
            "condition": cond, "n_chains": len(base_props),
            "baseline_rate": float(np.mean(base_props)),
            "condition_rate": float(np.mean(cond_props)),
            "delta_rate": float(np.mean(cond_props) - np.mean(base_props)),
            "test": test,
        })

    # Direct paired comparison seq2pair vs pair2seq (proportion-based)
    s2p_vs_p2s = {"n_chains": 0, "delta": 0.0, "p": 1.0}
    s2p_props = []
    p2s_props = []
    for c in per_chain.values():
        if not c["conditions"].get("seq2pair_donor") or not c["conditions"].get("pair2seq_donor"):
            continue
        s2p_props.append(float(np.mean(c["conditions"]["seq2pair_donor"])))
        p2s_props.append(float(np.mean(c["conditions"]["pair2seq_donor"])))
    if len(s2p_props) >= 10:
        test = paired_wilcoxon_proportion(s2p_props, p2s_props)
        s2p_vs_p2s = {"n_chains": len(s2p_props),
                      "s2p_rate": float(np.mean(s2p_props)),
                      "p2s_rate": float(np.mean(p2s_props)),
                      "delta_p2s_minus_s2p": float(np.mean(p2s_props) - np.mean(s2p_props)),
                      "test": test}

    def _row(cond):
        for r in per_condition_stats:
            if r["condition"] == cond:
                return r
        return None
    s2p = _row("seq2pair_donor")
    p2s = _row("pair2seq_donor")
    matched = _row("seq2pair_matched_ctrl")

    passes = {
        "s2p_effect_>=0.2pp": s2p is not None and s2p["delta_rate"] <= -0.2,
        "s2p_p_<0.05": s2p is not None and s2p["test"]["p"] < 0.05,
        "p2s_effect_<50pct_of_s2p":
            p2s is None or s2p is None or abs(p2s["delta_rate"]) < 0.5 * abs(s2p["delta_rate"]),
        "matched_ctrl_<50pct":
            matched is None or s2p is None or abs(matched["delta_rate"]) < 0.5 * abs(s2p["delta_rate"]),
        "s2p_vs_p2s_p_<0.05":
            isinstance(s2p_vs_p2s.get("test"), dict)
            and s2p_vs_p2s["test"].get("p", 1.0) < 0.05,
    }
    verdict = "supported" if all(passes.values()) else \
              ("partial" if passes["s2p_effect_>=0.2pp"] and passes["s2p_p_<0.05"]
               else "not-supported")

    summary = {
        "milestone": "M2", "claim": "C2",
        "per_condition_stats": per_condition_stats,
        "s2p_vs_p2s_paired": s2p_vs_p2s,
        "predicate_checks": passes,
        "verdict": verdict,
    }
    write_json(out_dir / "summary_stats.json", summary)
    print(f"[agg-M2] verdict={verdict}  s2p_Δ={s2p['delta_rate'] if s2p else None}  "
          f"p2s_Δ={p2s['delta_rate'] if p2s else None}", flush=True)
    return summary


def aggregate_m3a() -> Dict:
    out_dir = PROJECT_ROOT / "results" / "M3a"
    summary_path = out_dir / "m3a_summary.json"
    if not summary_path.exists():
        return {"milestone": "M3a", "verdict": "not-run"}
    summary = json.loads(open(summary_path).read())
    # Predicate: any block with test_balanced_acc >= 0.75 and perm_p < 0.001
    passed_any = False
    best_ba = 0
    best_block = None
    best_p = 1.0
    for row in summary["per_block"]:
        if row["test_balanced_acc"] >= 0.75 and row.get("permutation_p", 1.0) < 0.001:
            if row["test_balanced_acc"] > best_ba:
                best_ba = row["test_balanced_acc"]
                best_block = row["block"]
                best_p = row["permutation_p"]
                passed_any = True
    verdict = "supported" if passed_any else "not-supported"
    summary["milestone"] = "M3a"
    summary["claim"] = "C3a"
    summary["predicate_checks"] = {
        "balanced_acc_>=0.75": best_ba >= 0.75,
        "permutation_p_<0.001": best_p < 0.001,
    }
    summary["verdict"] = verdict
    write_json(out_dir / "summary_stats.json", summary)
    print(f"[agg-M3a] verdict={verdict}  best_ba={best_ba:.4f}  best_p={best_p}",
          flush=True)
    return summary


def aggregate_m3b() -> Dict:
    out_dir = PROJECT_ROOT / "results" / "M3b"
    files = sorted(glob.glob(str(out_dir / "steering_effect_worker_*.jsonl")))
    records = load_jsonl(files)
    print(f"[agg-M3b] loaded {len(records)} rows", flush=True)

    # Organize: for each chain, keep baseline hairpin + per (condition, config, direction, β) hairpin
    per_chain: Dict[str, Dict] = {}
    for r in records:
        cid = f"{r['pdb_id']}_{r['chain_id']}"
        per_chain.setdefault(cid, {"baseline": None, "conditions": {}, "cath_label": r.get("cath_label")})
        if not _ok(r):
            continue
        if r["condition"] == "baseline":
            per_chain[cid]["baseline"] = {"is_hairpin": r["is_hairpin"],
                                          "plddt_target": r["plddt_target"]}
        else:
            key = (r["condition"], r["config"], r["direction"], float(r["beta"]))
            per_chain[cid]["conditions"][key] = {
                "is_hairpin": r["is_hairpin"], "plddt_target": r["plddt_target"],
            }

    # Dose-response for v_charge target: per-chain (β, hairpin) points → Spearman ρ
    spearman_same = []
    spearman_opp = []
    betas_sorted = sorted({key[3] for c in per_chain.values() for key in c["conditions"].keys()
                          if key[0] == "steer_target"})
    for cid, c in per_chain.items():
        # For each config, extract vector (hairpin over β)
        for config, coll in [("same", spearman_same), ("opposite", spearman_opp)]:
            xs = []
            ys = []
            plddts = []
            for beta in betas_sorted:
                rec = c["conditions"].get(("steer_target", config, "v_charge", beta))
                if rec is None:
                    continue
                xs.append(beta)
                ys.append(rec["is_hairpin"])
                plddts.append(rec["plddt_target"])
            if len(xs) >= 3 and len(set(ys)) > 1:
                rho, _ = stats.spearmanr(xs, ys)
                if np.isfinite(rho):
                    coll.append(rho)

    def _mean_if(l):
        return float(np.mean(l)) if l else None

    # Same vs Opposite paired diff at β = ±3 (extreme dose)
    # Rule: same-charge (both +3σ) vs opposite (+3σ / -3σ)
    same_hi_vec = []
    opp_hi_vec = []
    for cid, c in per_chain.items():
        same_hi = c["conditions"].get(("steer_target", "same", "v_charge", 3.0))
        opp_hi = c["conditions"].get(("steer_target", "opposite", "v_charge", 3.0))
        if same_hi and opp_hi:
            same_hi_vec.append(same_hi["is_hairpin"])
            opp_hi_vec.append(opp_hi["is_hairpin"])
    same_vs_opp = None
    if len(same_hi_vec) >= 10:
        same_vs_opp = paired_mcnemar(same_hi_vec, opp_hi_vec)
        same_vs_opp["same_rate"] = float(np.mean(same_hi_vec))
        same_vs_opp["opp_rate"] = float(np.mean(opp_hi_vec))
        same_vs_opp["n_chains"] = len(same_hi_vec)

    # Matched-control specificity: same beta comparison
    matched_vs_target = None
    tgt_opp_vec = []
    mctrl_opp_vec = []
    for cid, c in per_chain.items():
        tgt_opp = c["conditions"].get(("steer_target", "opposite", "v_charge", 3.0))
        mctrl_opp = c["conditions"].get(("steer_matched_ctrl", "opposite", "v_charge", 3.0))
        if tgt_opp and mctrl_opp:
            tgt_opp_vec.append(tgt_opp["is_hairpin"])
            mctrl_opp_vec.append(mctrl_opp["is_hairpin"])
    if len(tgt_opp_vec) >= 10:
        matched_vs_target = paired_mcnemar(tgt_opp_vec, mctrl_opp_vec)
        matched_vs_target["target_effect"] = float(np.mean(tgt_opp_vec))
        matched_vs_target["matched_effect"] = float(np.mean(mctrl_opp_vec))
        matched_vs_target["n_chains"] = len(tgt_opp_vec)

    # Random direction control at β=+3: hairpin rate should be near baseline
    random_control = None
    rnd_vec = []
    for cid, c in per_chain.items():
        rnd = c["conditions"].get(("steer_random", "same", "v_random", 3.0))
        if rnd:
            rnd_vec.append(rnd["is_hairpin"])
    if rnd_vec:
        random_control = {"n_chains": len(rnd_vec), "hairpin_rate": float(np.mean(rnd_vec))}

    # pLDDT collapse detection — has structure collapsed under any β?
    plddt_by_beta_config = defaultdict(list)
    for cid, c in per_chain.items():
        base_plddt = c["baseline"]["plddt_target"] if c["baseline"] else None
        if base_plddt is None:
            continue
        for key, rec in c["conditions"].items():
            plddt_by_beta_config[(key[0], key[1], key[2], key[3])].append(rec["plddt_target"] - base_plddt)
    plddt_summary = {}
    for k, vals in plddt_by_beta_config.items():
        plddt_summary[f"{k[0]}__{k[1]}__{k[2]}__beta{k[3]}"] = {
            "n": len(vals), "mean_delta_plddt": float(np.mean(vals)),
        }

    # Verdict predicates for Claim 3b
    passes = {
        "spearman_same_>=0.7": _mean_if(spearman_same) is not None and abs(_mean_if(spearman_same)) >= 0.7,
        "spearman_opp_>=0.7": _mean_if(spearman_opp) is not None and abs(_mean_if(spearman_opp)) >= 0.7,
        "same_vs_opp_p_<0.05":
            same_vs_opp is not None and same_vs_opp.get("p", 1.0) < 0.05,
        "same_vs_opp_delta_>=0.15":
            same_vs_opp is not None and abs(same_vs_opp.get("effect_size", 0.0)) >= 0.15,
        "matched_ctrl_<50pct":
            matched_vs_target is None
            or (same_vs_opp is not None
                and abs(matched_vs_target["matched_effect"] - float(np.mean(tgt_opp_vec)) + float(np.mean(mctrl_opp_vec)))
                    < 0.5 * abs(same_vs_opp.get("effect_size", 0.0) or 1.0)),
        "random_ctrl_near_baseline":
            random_control is None or (abs(random_control["hairpin_rate"] - (float(np.mean([c["baseline"]["is_hairpin"]
                                                                                            for c in per_chain.values()
                                                                                            if c["baseline"]])))) < 0.15),
    }
    supported_count = sum(passes.values())
    verdict = "supported" if supported_count >= 5 else \
              ("partial" if supported_count >= 3 else "not-supported")

    summary = {
        "milestone": "M3b", "claim": "C3b",
        "betas_sweep": betas_sorted,
        "spearman_same_mean": _mean_if(spearman_same),
        "spearman_opp_mean": _mean_if(spearman_opp),
        "n_chains_dose_response": {"same": len(spearman_same), "opp": len(spearman_opp)},
        "same_vs_opp_paired": same_vs_opp,
        "matched_vs_target_paired": matched_vs_target,
        "random_control": random_control,
        "plddt_delta_by_condition": plddt_summary,
        "predicate_checks": passes,
        "verdict": verdict,
    }
    write_json(out_dir / "summary_stats.json", summary)
    print(f"[agg-M3b] verdict={verdict}", flush=True)
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--milestone", choices=["M1", "M2", "M3a", "M3b", "all"], required=True)
    args = p.parse_args()
    if args.milestone in ("M1", "all"):
        aggregate_m1()
    if args.milestone in ("M2", "all"):
        aggregate_m2()
    if args.milestone in ("M3a", "all"):
        aggregate_m3a()
    if args.milestone in ("M3b", "all"):
        aggregate_m3b()
