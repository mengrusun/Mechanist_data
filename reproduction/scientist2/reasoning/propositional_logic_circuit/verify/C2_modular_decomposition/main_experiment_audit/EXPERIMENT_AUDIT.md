# Experiment Audit Report — Claim C2

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via dmxapi, llm-chat-equivalent)
**Project**: Sparse Modular Circuit for Propositional-Logic Reasoning
**Claim**: C2 — "The shortlisted components decompose into three modular sub-circuits with distinct functional roles (fact identification, rule application, answer projection), evidenced by a role-assignment matrix S with median dominance ratio ≥ 2× and per-role dissociation d ≥ 0.1, stable across ≥ 2 additional cells (Jaccard ≥ 0.6)."
**Linked milestones**: M4, M4.stab

## Overall Verdict: WARN

*This is C2's integrity verdict — whether C2's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
GT for role-dissociation (C2) is loaded directly from the synthetic dataset JSONL files — one for each of the three corruption types (corrupt_fact, corrupt_rule, corrupt_answer). The `answer` field in each JSONL is set deterministically by `build_propositional_dataset.py` (50/50 True/False balance). Corruption labels (fact/rule/answer role) are structural properties of the corruption generator, not derived from model output. The modularity metric S[c,r] = Recovery_r(c) is a function of patch recovery, where Recovery uses `(patch_ld - corr_ld) / (clean_ld - corr_ld)` — denominator is the clean-corrupt logit-difference delta, not model max/mean.

Code: `answers = torch.tensor([1 if r["answer"] == "True" else 0 for r in chunk], device=device)` (prop_circuit_lib.py line ~263).

### B. Score Normalization: PASS
The role-assignment matrix S is computed from recovery scores: `S[c, r] = Recovery_r(c) = (patch_ld_after - corr_ld) / (clean_ld - corr_ld)`. Denominator is the clean-corrupt logit-diff delta, never the model's own max/mean output. The modularity and dissociation metrics are all functions of S. No normalization fraud.

### C. Result File Existence: PASS
Both result files cited for C2 exist on disk and match reported values:
- `results/M4_roles.json`: exists. Reported values match file — `median_dominance_ratio=1.197` (reported 1.20), `dissociation={fact:0.089, rule:0.027, answer:0.026}`, `null_shuffle_p_value={fact:0.05, rule:1.0, answer:1.0}`, `success_C2=false`. ✓
- `results/M4stab.json`: exists. Reported Jaccard values match file — `fact=0.750`, `answer=0.769`, `rule=0.364`. `success_C2=false`, `success_C2_stability=false`. ✓
- `results/M1_top40_for_M4.json`: referenced as the shortlist input to M4/M4.stab. Exists on disk. ✓
Tracker status for M4 and M4.stab: DONE. ✓

### D. Dead Code Detection: PASS
All key functions used for C2 — `compute_S_matrix`, `compute_modularity`, `null_shuffle_pvalue`, `jaccard` (in role_dissociation.py) — are called in the main pipeline and produce outputs that appear in `M4_roles.json` and `M4stab.json`. The `block_partition`, `S_matrix`, `dominance_ratios`, `dissociation`, and `null_shuffle_p_value` fields all appear in the output files. No dead or decorative metric code identified.

### E. Scope Assessment: WARN
**Shortlist reduction (M4/M4.stab) — disclosed but affects claim evidential scope.** The plan specifies C2 should run role-dissociation on the C1 shortlist (158 components). Both M4 and M4.stab used only the top-40 of 158 components (25% of shortlist), selected by |attribution score| from `results/M1_top40_for_M4.json`. This reduction is **explicitly disclosed** in:
- EXPERIMENT_RESULTS.md §M4 "Shortlist reduction" paragraph
- EXPERIMENT_RESULTS.md §Notes "Shortlist scope was reduced"
- EXPERIMENT_TRACKER.md M4 row "done (top-40 shortlist)"
- EXPERIMENT_RESULTS.md key_stats "Shortlist restricted to top-40 of 158 for compute-budget reasons"

The justification (top-40 covers ~0.95 of causal effect per M2 dose-response) is documented and defensible. However, the **claim statement for C2 in the plan references "the shortlisted components"** (all 158), not the top-40. Testing only 25% of the shortlist introduces a representativeness concern: the lower-attribution components (ranks 41–158) may have different role-assignment profiles, and their exclusion means C2's verdict is technically conditioned on the top-40 sublist. This is a scope mismatch between the plan's claim predicate and the executed experiment — hence WARN rather than FAIL (the scope reduction is defensible and disclosed, not fraudulent).

The claim verdict (C2 FAIL) is unlikely to reverse on the remaining 118 components, as those have lower causal effect and the top-40 already shows near-uniform smearing across roles. But the mismatch is real.

### F. Evaluation Type: synthetic_proxy
Task uses a fully programmatically generated synthetic propositional-logic dataset. Role corruptions (fact-swap, rule-swap, answer-swap) are structural modifications of the generator's template. GT is deterministic and not human-annotated. Classification: `synthetic_proxy`.

## Action Items
1. **Clarify scope**: state explicitly in EXPERIMENT_RESULTS.md and any write-up that C2's verdict is conditioned on the top-40 of 158 shortlist components, not the full shortlist. Add a qualifier to the C2 verdict: "conditioned on top-40 of 158-component shortlist (by |attribution score|)".
2. Optional: re-run M4 on the full 158-component shortlist using a cached-forward implementation to confirm the FAIL verdict extends to the full shortlist (very likely given uniform smearing in top-40, but would remove the scope caveat).
