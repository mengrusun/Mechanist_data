#!/usr/bin/env python3
"""
M6 — Aggregate M1..M5 into a final report table + figures.

Reads results/*.json and produces:
  - results/final_report.md
"""

import argparse
import json
from pathlib import Path


def _load(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    r = Path(args.results)
    m0 = _load(r / "M0_setup.json")
    m1 = _load(r / "M1_attribution.json")
    m2 = _load(r / "M2_necessity.json")
    m3 = _load(r / "M3_sufficiency.json")
    m4 = _load(r / "M4_roles.json")
    m4stab = _load(r / "M4stab.json")
    m5 = _load(r / "M5_gemma9b.json")

    lines = []
    lines.append("# Final Report — Sparse Modular Circuit for Propositional-Logic Reasoning\n")
    lines.append("## Setup (M0)\n")
    if m0:
        lines.append(f"- model: `{m0.get('model_path')}` (fallback used: {m0.get('fallback_used')})")
        lines.append(f"- anchor accuracy: **{m0.get('anchor_accuracy'):.3f}** (top1 T/F share: {m0.get('anchor_top1_TF_share'):.3f})")
        lines.append(f"- sanity criterion (>=0.75) met: **{m0.get('sanity_criterion_met')}**\n")
    else:
        lines.append("- (M0 result missing)\n")

    lines.append("## C1 Location — attribution-patching screen (M1)\n")
    if m1:
        lines.append(f"- shortlist size: **{m1['shortlist_size']}** ({m1['sparsity_fraction']*100:.1f}% of components)")
        lines.append(f"- cumulative effect: {m1['cumulative_effect']:.3f}")
        lines.append(f"- completeness (recovery LD): {m1['completeness']:.3f}")
        lines.append(f"- minimality avg single-removal drop: {m1['minimality']['avg_single_removal_drop']:.3f}")
        lines.append(f"- **success_C1**: {m1['success_C1']}\n")
    else:
        lines.append("- (M1 result missing)\n")

    lines.append("## C3 Necessity — path patching (M2)\n")
    if m2:
        lines.append(f"- recovery LD/PD/KL: **{m2['recovery']['logit_diff']:.3f}** / "
                     f"{m2['recovery']['prob_diff']:.3f} / {m2['recovery']['KL']:.3f}")
        lines.append(f"- control recovery LD: {m2['control_recovery']['logit_diff']:.3f}")
        lines.append(f"- specificity_gap: {m2['specificity_gap']:.3f}")
        lines.append(f"- **success_C3_necessity**: {m2['success_C3_necessity']}\n")
        lines.append(f"- dose-response ({len(m2['dose_response'])} points):")
        for d in m2["dose_response"]:
            lines.append(f"  - k={d['k_patched']}: LD={d['recovery_logit_diff']:.3f}, "
                         f"PD={d['recovery_prob_diff']:.3f}, KL={d['recovery_KL']:.3f}")
        lines.append("")
    else:
        lines.append("- (M2 result missing)\n")

    lines.append("## C3 Sufficiency — reinsertion (M3)\n")
    if m3:
        lines.append(f"- sufficient_recovery LD/PD/KL: **{m3['sufficient_recovery']['logit_diff']:.3f}** / "
                     f"{m3['sufficient_recovery']['prob_diff']:.3f} / {m3['sufficient_recovery']['KL']:.3f}")
        lines.append(f"- control LD: {m3['control_sufficient_recovery']['logit_diff']:.3f}")
        lines.append(f"- specificity_gap: {m3['specificity_gap']:.3f}")
        lines.append(f"- per-seed std (LD): {m3['resample_seed_distribution']['logit_diff']['std']:.3f}")
        lines.append(f"- **success_C3_sufficiency**: {m3['success_C3_sufficiency']}\n")
    else:
        lines.append("- (M3 result missing)\n")

    lines.append("## C2 Role dissociation (M4)\n")
    if m4:
        for cell, mod in m4["per_cell"].items():
            lines.append(f"### cell {cell}")
            lines.append(f"- median dominance ratio: {mod['median_dominance_ratio']:.2f}")
            lines.append(f"- dissociation: {mod['dissociation']}")
            lines.append(f"- null-shuffle p-values: {mod['null_shuffle_p_value']}")
            lines.append(f"- block partition sizes: "
                         f"fact={len(mod['block_partition']['fact'])}, "
                         f"rule={len(mod['block_partition']['rule'])}, "
                         f"answer={len(mod['block_partition']['answer'])}")
        lines.append(f"- **success_C2**: {m4.get('success_C2')}\n")
    else:
        lines.append("- (M4 result missing)\n")

    lines.append("## C2 stability across cells (M4.stab)\n")
    if m4stab:
        stab = m4stab.get("stability", {})
        for r_, v in stab.items():
            lines.append(f"- role={r_}: mean_jaccard={v['mean_jaccard']:.3f}")
        lines.append(f"- **success_C2_stability**: {m4stab.get('success_C2_stability')}\n")
    else:
        lines.append("- (M4.stab result missing)\n")

    lines.append("## Cross-family verify (M5 — Gemma-2-9B)\n")
    if m5:
        lines.append(f"- anchor accuracy: {m5['anchor_accuracy']:.3f}")
        lines.append(f"- sparsity fraction: {m5['sparsity_fraction']*100:.1f}%")
        lines.append(f"- necessity LD: {m5['necessity_recovery']['logit_diff']:.3f}")
        lines.append(f"- sufficiency LD: {m5['sufficiency_recovery']['logit_diff']:.3f}")
        lines.append(f"- role dissociation: {m5['role_dissociation']}")
        lines.append(f"- **success_cross_family_recurrence**: {m5['success_cross_family_recurrence']}\n")
    else:
        lines.append("- (M5 result missing)\n")

    with open(args.out, "w") as fh:
        fh.write("\n".join(lines))
    print(f"[M6] wrote {args.out}")


if __name__ == "__main__":
    main()
