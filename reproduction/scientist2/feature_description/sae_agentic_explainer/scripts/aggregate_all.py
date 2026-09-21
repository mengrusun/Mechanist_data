"""
Aggregate M1 + M2 results into per-claim verdicts. Reads:
  results/m1/features_L{4,12,20}.jsonl  (M1)
  results/m2/features.jsonl             (M2 predictive-only)

Writes:
  results/m1/summary.json
  results/m2/summary.json
  results/aggregate.json      (headline: per-claim verdict)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


def load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    records = []
    with open(p) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def paired_bootstrap_ci(a: list[float], b: list[float], n_resamples: int = 10000, ci: float = 95.0, rng_seed: int = 0):
    if not a or not b or len(a) != len(b):
        return None
    a_ = np.asarray(a, dtype=np.float32)
    b_ = np.asarray(b, dtype=np.float32)
    d = a_ - b_
    rng = np.random.default_rng(rng_seed)
    boots = np.zeros(n_resamples, dtype=np.float32)
    for i in range(n_resamples):
        boots[i] = rng.choice(d, size=len(d), replace=True).mean()
    lo = float(np.percentile(boots, (100 - ci) / 2))
    hi = float(np.percentile(boots, 100 - (100 - ci) / 2))
    return {
        "n": len(a),
        "mean_a": float(a_.mean()),
        "mean_b": float(b_.mean()),
        "mean_delta": float(d.mean()),
        "ci_low": lo,
        "ci_high": hi,
        "significant_positive": bool(lo > 0),
        "significant_negative": bool(hi < 0),
    }


def paired_wilcoxon(a: list[float], b: list[float]) -> dict | None:
    if not a or not b or len(a) != len(b):
        return None
    from scipy.stats import wilcoxon
    a_ = np.asarray(a, dtype=np.float32)
    b_ = np.asarray(b, dtype=np.float32)
    d = a_ - b_
    if float(np.abs(d).max()) < 1e-9:
        return {"stat": 0.0, "pvalue": 1.0}
    try:
        w = wilcoxon(a_, b_, zero_method="wilcox", nan_policy="omit")
        return {"stat": float(w.statistic), "pvalue": float(w.pvalue)}
    except Exception as e:
        return {"error": str(e)}


def _extract(records, m, key):
    return [r["methods"].get(m, {}).get(key) for r in records if m in r["methods"] and r["methods"][m].get(key) is not None]


def _paired(records, m1, m2, key):
    a, b = [], []
    for r in records:
        if m1 in r["methods"] and m2 in r["methods"]:
            va = r["methods"][m1].get(key)
            vb = r["methods"][m2].get(key)
            if va is not None and vb is not None:
                a.append(va); b.append(vb)
    return a, b


def m1_summary(records, methods, layers, out_path: Path):
    summary: dict[str, Any] = {"n_features_total": len(records), "per_layer": {}, "overall": {}, "pairwise_deltas": {}}
    for L in layers:
        recs = [r for r in records if r["layer"] == L]
        entry = {"n_features": len(recs)}
        for m in methods:
            entry[m] = {
                "mean_gen_acc": float(np.mean(_extract(recs, m, "gen_acc") or [0])),
                "mean_pearson": float(np.mean(_extract(recs, m, "pred_pearson") or [0])),
                "mean_auroc": float(np.mean(_extract(recs, m, "pred_auroc") or [0.5])),
                "n": len(_extract(recs, m, "gen_acc")),
            }
        summary["per_layer"][f"L{L}"] = entry
    for m in methods:
        summary["overall"][m] = {
            "mean_gen_acc": float(np.mean(_extract(records, m, "gen_acc") or [0])),
            "mean_pearson": float(np.mean(_extract(records, m, "pred_pearson") or [0])),
            "mean_auroc": float(np.mean(_extract(records, m, "pred_auroc") or [0.5])),
            "n": len(_extract(records, m, "gen_acc")),
        }
    pairs = [("sage", "neuronpedia"), ("sage", "gpt5_1shot"), ("gpt5_1shot", "neuronpedia")]
    metrics = ["gen_acc", "pred_pearson", "pred_auroc"]
    for (m1, m2) in pairs:
        pkey = f"{m1}_vs_{m2}"
        summary["pairwise_deltas"][pkey] = {"overall": {}, "per_layer": {}}
        for k in metrics:
            a, b = _paired(records, m1, m2, k)
            if a:
                ci = paired_bootstrap_ci(a, b, rng_seed=abs(hash((m1, m2, k))) & 0xffff)
                w = paired_wilcoxon(a, b)
                summary["pairwise_deltas"][pkey]["overall"][k] = {**(ci or {}), "wilcoxon": w}
            for L in layers:
                recs_L = [r for r in records if r["layer"] == L]
                a, b = _paired(recs_L, m1, m2, k)
                if a:
                    ci = paired_bootstrap_ci(a, b, rng_seed=abs(hash((m1, m2, k, L))) & 0xffff)
                    w = paired_wilcoxon(a, b)
                    summary["pairwise_deltas"][pkey]["per_layer"].setdefault(f"L{L}", {})[k] = {**(ci or {}), "wilcoxon": w}
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def m2_summary(records, methods, layers, out_path: Path):
    summary: dict[str, Any] = {"n_features_total": len(records), "per_layer": {}, "overall": {}, "pairwise_deltas": {}}
    for L in layers:
        recs = [r for r in records if r["layer"] == L]
        entry = {"n_features": len(recs)}
        for m in methods:
            entry[m] = {
                "mean_pearson": float(np.mean(_extract(recs, m, "pred_pearson") or [0])),
                "mean_auroc": float(np.mean(_extract(recs, m, "pred_auroc") or [0.5])),
                "n": len(_extract(recs, m, "pred_pearson")),
            }
        summary["per_layer"][f"L{L}"] = entry
    for m in methods:
        summary["overall"][m] = {
            "mean_pearson": float(np.mean(_extract(records, m, "pred_pearson") or [0])),
            "mean_auroc": float(np.mean(_extract(records, m, "pred_auroc") or [0.5])),
            "n": len(_extract(records, m, "pred_pearson")),
        }
    pairs = [("sage_lite", "neuronpedia"), ("sage_lite", "gpt5_1shot"), ("gpt5_1shot", "neuronpedia")]
    metrics = ["pred_pearson", "pred_auroc"]
    for (m1, m2) in pairs:
        pkey = f"{m1}_vs_{m2}"
        summary["pairwise_deltas"][pkey] = {"overall": {}, "per_layer": {}}
        for k in metrics:
            a, b = _paired(records, m1, m2, k)
            if a:
                ci = paired_bootstrap_ci(a, b, rng_seed=abs(hash((m1, m2, k))) & 0xffff)
                w = paired_wilcoxon(a, b)
                summary["pairwise_deltas"][pkey]["overall"][k] = {**(ci or {}), "wilcoxon": w}
            for L in layers:
                recs_L = [r for r in records if r["layer"] == L]
                a, b = _paired(recs_L, m1, m2, k)
                if a:
                    ci = paired_bootstrap_ci(a, b, rng_seed=abs(hash((m1, m2, k, L))) & 0xffff)
                    w = paired_wilcoxon(a, b)
                    summary["pairwise_deltas"][pkey]["per_layer"].setdefault(f"L{L}", {})[k] = {**(ci or {}), "wilcoxon": w}
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    root = Path("/data/zhenqian/Reproduction1/mechanica/feature_description/sae_agentic_explainer/results")
    m1_files = sorted(list((root / "m1").glob("features_L*.jsonl")))
    m1_records = []
    for f in m1_files:
        m1_records.extend(load_jsonl(f))
    print(f"M1: loaded {len(m1_records)} records")
    m1_layers = sorted(set(r["layer"] for r in m1_records))
    m1_methods = ["sage", "neuronpedia", "gpt5_1shot"]
    m1_s = m1_summary(m1_records, m1_methods, m1_layers, root / "m1" / "summary.json") if m1_records else {}

    m2_records = load_jsonl(root / "m2" / "features.jsonl")
    print(f"M2: loaded {len(m2_records)} records")
    m2_layers = sorted(set(r["layer"] for r in m2_records))
    m2_methods = ["sage_lite", "neuronpedia", "gpt5_1shot"]
    m2_s = m2_summary(m2_records, m2_methods, m2_layers, root / "m2" / "summary.json") if m2_records else {}

    # Per-claim verdict
    verdict = {"C1": {}, "C2": {}, "C3": {}, "C4": {}, "under_power": {}}
    # C1: generative accuracy on main pair — mean gain SAGE-vs-Neuronpedia > 0 with 95% CI lower > 0
    if m1_s and "pairwise_deltas" in m1_s:
        vs_np = m1_s["pairwise_deltas"].get("sage_vs_neuronpedia", {}).get("overall", {}).get("gen_acc")
        vs_gpt = m1_s["pairwise_deltas"].get("sage_vs_gpt5_1shot", {}).get("overall", {}).get("gen_acc")
        verdict["C1"] = {
            "sage_vs_neuronpedia": vs_np,
            "sage_vs_gpt5_1shot": vs_gpt,
            "n": vs_np["n"] if vs_np else 0,
            "supported_vs_neuronpedia": bool(vs_np and vs_np.get("significant_positive")),
            "supported_vs_gpt5_1shot": bool(vs_gpt and vs_gpt.get("significant_positive")),
        }
        # C2: predictive accuracy
        vs_np2 = m1_s["pairwise_deltas"].get("sage_vs_neuronpedia", {}).get("overall", {}).get("pred_pearson")
        vs_gpt2 = m1_s["pairwise_deltas"].get("sage_vs_gpt5_1shot", {}).get("overall", {}).get("pred_pearson")
        verdict["C2"] = {
            "sage_vs_neuronpedia": vs_np2,
            "sage_vs_gpt5_1shot": vs_gpt2,
            "n": vs_np2["n"] if vs_np2 else 0,
            "supported_vs_neuronpedia": bool(vs_np2 and vs_np2.get("significant_positive")),
            "supported_vs_gpt5_1shot": bool(vs_gpt2 and vs_gpt2.get("significant_positive")),
        }
        # C3: layer-stratified holds
        c3_layers = {}
        for L in m1_layers:
            per = m1_s["pairwise_deltas"]["sage_vs_neuronpedia"]["per_layer"].get(f"L{L}", {})
            per_gpt = m1_s["pairwise_deltas"]["sage_vs_gpt5_1shot"]["per_layer"].get(f"L{L}", {})
            c3_layers[f"L{L}"] = {
                "gen_acc_vs_neuronpedia": per.get("gen_acc"),
                "pred_pearson_vs_neuronpedia": per.get("pred_pearson"),
                "gen_acc_vs_gpt5_1shot": per_gpt.get("gen_acc"),
                "pred_pearson_vs_gpt5_1shot": per_gpt.get("pred_pearson"),
            }
        verdict["C3"] = c3_layers
    # C4: cross-pair generalization
    if m2_s and "pairwise_deltas" in m2_s:
        vs_np_m2 = m2_s["pairwise_deltas"].get("sage_lite_vs_neuronpedia", {}).get("overall", {}).get("pred_pearson")
        vs_gpt_m2 = m2_s["pairwise_deltas"].get("sage_lite_vs_gpt5_1shot", {}).get("overall", {}).get("pred_pearson")
        verdict["C4"] = {
            "sage_lite_vs_neuronpedia_pearson": vs_np_m2,
            "sage_lite_vs_gpt5_1shot_pearson": vs_gpt_m2,
            "n": vs_np_m2["n"] if vs_np_m2 else 0,
            "supported_vs_neuronpedia": bool(vs_np_m2 and vs_np_m2.get("significant_positive")),
            "supported_vs_gpt5_1shot": bool(vs_gpt_m2 and vs_gpt_m2.get("significant_positive")),
            "note": "M2 scoped to predictive-accuracy only (no target-LLM generative test) due to Qwen3-4B + transcoder-hp integration cost within budget. Full generative test deferred to /auto-verify. SAGE-lite (Explainer + Reviewer only, no empirical activation feedback) used here — the full 4-role SAGE requires target-LLM forward.",
        }

    # Underpower flags per plan
    verdict["under_power"] = {
        "planned_used_n": {"m1_per_layer": 100, "m2_per_layer": 50},
        "realized_used_n": {
            "m1_L4": sum(1 for r in m1_records if r["layer"] == 4),
            "m1_L12": sum(1 for r in m1_records if r["layer"] == 12),
            "m1_L20": sum(1 for r in m1_records if r["layer"] == 20),
            "m2_per_layer_mean": len(m2_records) // max(len(m2_layers), 1),
        },
        "under_powered_claims_if_null": ["C1", "C2", "C3", "C4"] if len(m1_records) < 200 else [],
    }

    with open(root / "aggregate.json", "w") as f:
        json.dump(verdict, f, indent=2)
    print(f"[aggregate] wrote {root / 'aggregate.json'}")
    print(json.dumps(verdict, indent=2)[:3000])


if __name__ == "__main__":
    main()
