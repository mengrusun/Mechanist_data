# Experiment Audit Report — Claim C2

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (gpt-5.4 via dmxapi, cross-model)
**Project**: Unified Verification of the Four-Claim Language-Agnostic/Specific Subspace Hypothesis on Qwen-3-4B-Thinking + MGSM
**Claim**: C2 — Suppressing the language-specific subspace at inference time via null-space projection at non-upper layers raises MGSM mean accuracy by ≥ 3 pp across 11 target languages, with GlotLID output-language fidelity drop ≤ 5 pp when the top-k layers are left intact.
**Linked milestones**: M2

## Overall Verdict: WARN

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
Gold answers sourced from MGSM parquet dataset, not model outputs. Grading: `grade() = numeric_equal(extract_answer(generation), gold_answer)` with `gold_answer` loaded from `/data/zhenqian/data/mgsm/{lang}/test-*.parquet`. Completely real dataset GT; no self-referential scoring.

### B. Score Normalization: PASS
`macro_acc` computed as mean of per-language accuracies; each accuracy = `correct / n_problems` with fixed `n_problems=25` (dataset size). No model-output-derived denominator. No normalization fraud.

### C. Result File Existence: PASS
All claimed results verified to exist with matching numbers:
- `results/m2/baseline_no_intervention_summary.json`: macro_acc=0.7618 (claimed 0.762 ✓)
- `results/m2/screen_mid_r2_ktop12_s42_summary.json`: macro_acc=0.0655 (claimed 0.065 ✓)
- `results/m2/screen_mid_r8_ktop12_s42_summary.json`: macro_acc=0.0291 (claimed 0.029 ✓)
- `results/m2/random_ctrl_mid_ktop12_s42_summary.json`: macro_acc=0.4291 (exists ✓)
- M2A run directories: exist with logs

### D. Dead Code Detection: PASS
`extract_answer()`, `numeric_equal()`, `grade()` in `mgsm_eval.py` all called in `m2_projection_eval.py` evaluation loop. No deceptive dead code affecting reported results.

### E. Scope Assessment: WARN
Reported not-supported verdict is honestly grounded but based on reduced scope vs plan:
- Only `mid` layer group tested (plan covered `early`, `mid`, `all_non_upper`)
- Only rank_r ∈ {2, 8} tested (plan: {2, 4, 8})
- Only 1 seed and n=25/lang (plan: 3 seeds, n=250/lang)
- Stage B verify (3-seed verification at winning config) aborted after screen
- This is documented in EXPERIMENT_TRACKER.md as intentional: "screen sufficient — Δ signal is strong enough to reject Claim 2 at n=25/lang"
- Not a fraud signal; limitation is documented and conclusion is grounded on observed 65-73 pp collapse. But scope is narrower than the claim's "across diverse models, languages, and reasoning tasks."

### F. Evaluation Type: real_gt
MGSM gold answers from dataset parquet files; GlotLID fidelity on model-generated continuations (real downstream measurement, not GT-proxy).

## Action Items
- [WARN-E] Scope limitation: only mid layer group and 2 rank_r values tested. Re-run early/all_non_upper groups in iteration to complete the planned scope.
