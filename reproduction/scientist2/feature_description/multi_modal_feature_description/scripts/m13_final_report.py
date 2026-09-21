"""
M13_final_report — Aggregated results table + human-readable summary for /result-to-claim.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


def load_json(p: Path) -> dict:
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default="runs/")
    p.add_argument("--out", default="runs/M13_final_report/results_table.json")
    p.add_argument("--out_md", default="runs/M13_final_report/FINAL_RESULTS.md")
    args = p.parse_args()

    root = Path(args.results_root)

    # Load all predicate outputs
    m6 = {p.name: load_json(p) for p in sorted((root / "M6_C1_last_layer").glob("*.json"))}
    m7 = {p.name: load_json(p) for p in sorted((root / "M7_C1_hidden").glob("*.json"))}
    m8 = {p.name: load_json(p) for p in sorted((root / "M8_C2_queryability").glob("*.json"))}
    m9 = {p.name: load_json(p) for p in sorted((root / "M9_C2_stability").glob("*.json"))}
    m10 = {p.name: load_json(p) for p in sorted((root / "M10_C2_separation").glob("*.json"))}
    m11 = load_json(root / "M11_layer_granularity" / "layer_summary.json")
    m12 = {p.name: load_json(p) for p in sorted((root / "M12_cross_model_verify").glob("*.json"))}

    # Predicate-level headline results (headline setting: k=16, pool=mean)
    def get_purity_headline(): return m6.get("purity__k16__mean.json", {})
    def get_sep_hidden_headline(): return m7.get("sep__k16__mean.json", {})
    def get_mrr_headline(pool="mean"): return m8.get(f"mrr__k16__{pool}.json", {})
    def get_stab_headline(pool="mean"): return m9.get(f"stability__hk16__{pool}.json", {})
    def get_sep_last_headline(pool="mean"): return m10.get(f"sep__k16__{pool}.json", {})

    P1a = get_purity_headline()
    P1b_l4 = get_sep_hidden_headline().get("per_layer", {}).get("layer4", {})
    P1b_l3 = get_sep_hidden_headline().get("per_layer", {}).get("layer3", {})

    p1c_l4 = m11.get("P1c_plateau_analysis", {}).get("layer4", {}) if m11 else {}
    p1c_l3 = m11.get("P1c_plateau_analysis", {}).get("layer3", {}) if m11 else {}

    # C1 decision:
    # C1 supported iff P1a passes AND P1b passes on >= layer4 AND P1c plateau <= 16
    c1_p1a = bool(P1a.get("passes"))
    c1_p1b = bool(P1b_l4.get("passes_P1b"))
    c1_p1c = bool(p1c_l4.get("passes_P1c", False))
    c1_verdict = "supported" if (c1_p1a and c1_p1b and c1_p1c) else (
        "partial" if (int(c1_p1a) + int(c1_p1b) + int(c1_p1c) >= 2) else "not-supported"
    )

    # C2 decision:
    # C2 supported iff P2a passes AND P2b median >= 0.5 AND P2c positive gap d >= 0.5 AND P2d qualitative consistency across all 4 pool ops
    P2a_mean = get_mrr_headline("mean")
    P2a_awm = get_mrr_headline("act_weighted_mean")
    P2a_max = get_mrr_headline("max")
    P2a_med = get_mrr_headline("medoid")
    P2b_mean = get_stab_headline("mean")
    P2c_mean = get_sep_last_headline("mean")

    c2_p2a = bool(P2a_mean.get("significant"))
    c2_p2b = bool(P2b_mean.get("passes"))
    c2_p2c = bool(P2c_mean.get("fc", {}).get("passes")) or bool(P2c_mean.get("layer4", {}).get("passes"))
    # P2d: sign consistency of P2a MRR across 4 pool ops (all significant OR all not-significant is OK; sign flip is a fail)
    p2a_sig_mean = P2a_mean.get("significant")
    all_sigs = [P2a_mean.get("significant"), P2a_awm.get("significant"),
                P2a_max.get("significant"), P2a_med.get("significant")]
    c2_p2d = all(s is not None for s in all_sigs) and (all(s for s in all_sigs) or all(not s for s in all_sigs))
    # Also require gap sign consistency for P2c across pool operators
    p2c_gaps = [get_sep_last_headline(pool).get("fc", {}).get("gap") for pool in ("mean", "act_weighted_mean", "max", "medoid")]
    p2c_gap_signs_consistent = all(g is not None and g > 0 for g in p2c_gaps) or all(g is not None and g <= 0 for g in p2c_gaps)
    c2_p2d = c2_p2d and p2c_gap_signs_consistent

    c2_verdict = "supported" if (c2_p2a and c2_p2b and c2_p2c and c2_p2d) else (
        "partial" if (int(c2_p2a) + int(c2_p2b) + int(c2_p2c) + int(c2_p2d) >= 3) else "not-supported"
    )

    # P3 cross-model
    p3_results = {}
    for name, d in m12.items():
        if d and "significant" in d:
            p3_results[d.get("inspected_model", name)] = {"mrr": d["mrr"], "significant": d["significant"], "perm95_upper": d.get("permutation_mrr_ci_95_upper")}
    n_sig_swap = sum(1 for v in p3_results.values() if v["significant"])
    p3_passes = n_sig_swap >= 2  # soft: >= 2 of 3

    result_table = {
        "headline_setting": "k=16, pool=mean",
        "C1_evaluation": {
            "P1a_last_layer_purity": P1a,
            "P1b_layer4_matched_control": P1b_l4,
            "P1b_layer3_matched_control": P1b_l3,
            "P1c_plateau_layer4": p1c_l4,
            "P1c_plateau_layer3": p1c_l3,
            "verdict": c1_verdict,
            "gate_details": {"P1a": c1_p1a, "P1b_layer4": c1_p1b, "P1c_layer4": c1_p1c},
        },
        "C2_evaluation": {
            "P2a_MRR_mean": P2a_mean,
            "P2a_MRR_act_weighted_mean": P2a_awm,
            "P2a_MRR_max": P2a_max,
            "P2a_MRR_medoid": P2a_med,
            "P2b_stability_mean": P2b_mean,
            "P2c_separation_mean": P2c_mean,
            "verdict": c2_verdict,
            "gate_details": {"P2a": c2_p2a, "P2b": c2_p2b, "P2c": c2_p2c, "P2d_consistency": c2_p2d},
        },
        "P3_cross_model": {
            "per_model": p3_results,
            "n_significant_of_3": n_sig_swap,
            "soft_passes_p3": p3_passes,
        },
        "M11_layer_granularity": m11,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result_table, indent=2))
    print(f"[M13] wrote {args.out}")

    # Human-readable MD
    lines = [
        "# SemanticLens Component -> CLIP Semantic-Vector — Main Results",
        "",
        "**Setting**: ResNet-50 (torchvision IMAGENET1K_V2) inspected on ImageNet-val; frozen CLIP ViT-B/32 (openai) as foundation encoder.",
        "**Method (routed via Multi-Modal / CLIP-Dissect)**: per-component top-k activation-driven reference inputs -> CLIP image tower -> pool -> v_c; cosine similarity vs CLIP text embeddings for scoring.",
        "**Headline setting**: k=16, pool=mean (SemanticLens default).",
        "",
        "## Claim C1 — Concept-faithful summary",
        f"**Verdict: {c1_verdict}**",
        "",
        "| Predicate | Value | Baseline | Pass? |",
        "|-----------|-------|----------|-------|",
        f"| P1a — Last-layer top-1 purity | {P1a.get('top1_purity'):.4f} | random-input {P1a.get('random_baseline_top1_purity'):.4f} (Δ={P1a.get('delta_pure'):.4f}, p={P1a.get('p_value_mcnemar'):.3g}) | {c1_p1a} |" if P1a else "| P1a | ??? | ??? | ??? |",
        f"| P1b — layer4 matched-control gap | Δ_sep={P1b_l4.get('delta_sep_mean'):.4f} (CI {P1b_l4.get('delta_sep_ci_95')}, p={P1b_l4.get('p_value_paired_greater'):.3g}) | vs. best-vs-second | {c1_p1b} |" if P1b_l4 else "",
        f"| P1c — k-plateau on layer4 | plateau_k={p1c_l4.get('plateau_k')} monotone={p1c_l4.get('monotone_nondecreasing')} | (curve: {p1c_l4.get('delta_sep_by_k')}) | {c1_p1c} |" if p1c_l4 else "",
        "",
        "## Claim C2 — v_c places c in joint CLIP image-text space",
        f"**Verdict: {c2_verdict}**",
        "",
        "| Predicate | Value | Baseline | Pass? |",
        "|-----------|-------|----------|-------|",
        f"| P2a — text-query MRR (mean) | {P2a_mean.get('mrr'):.4f} | perm95 upper {P2a_mean.get('permutation_mrr_ci_95_upper'):.4f} | {c2_p2a} |" if P2a_mean else "",
        f"| P2b — disjoint-half stability median (mean pool) | {P2b_mean.get('median_cosine'):.4f} | τ_stable=0.5 | {c2_p2b} |" if P2b_mean else "",
        f"| P2c — within-vs-between gap (fc) | gap={P2c_mean.get('fc', {}).get('gap'):.4f} d={P2c_mean.get('fc', {}).get('cohens_d'):.3f} | vs random cross-pair | {P2c_mean.get('fc', {}).get('passes')} |" if P2c_mean else "",
        f"| P2c — within-vs-between gap (layer4) | gap={P2c_mean.get('layer4', {}).get('gap'):.4f} d={P2c_mean.get('layer4', {}).get('cohens_d'):.3f} |  | {P2c_mean.get('layer4', {}).get('passes')} |" if P2c_mean else "",
        f"| P2d — sign consistency across pool ops | {c2_p2d} |  |  |",
        "",
        "## P3 — Cross-model transfer (SHOULD-RUN)",
    ]
    if p3_results:
        lines.append("| Model | MRR | perm95 upper | Sig? |")
        lines.append("|-------|-----|--------------|------|")
        for m, v in p3_results.items():
            lines.append(f"| {m} | {v['mrr']:.4f} | {v['perm95_upper']:.4f} | {v['significant']} |")
        lines.append(f"\n**Soft-pass ({n_sig_swap}/3)**: {p3_passes}")
    else:
        lines.append("_M12 not run yet or no results collected._")
    lines.append("")

    Path(args.out_md).write_text("\n".join(l for l in lines if l))
    print(f"[M13] wrote {args.out_md}")


if __name__ == "__main__":
    main()
