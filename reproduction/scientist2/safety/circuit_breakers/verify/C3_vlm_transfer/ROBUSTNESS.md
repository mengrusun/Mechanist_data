# Robustness Report — Claim C3

**Claim**: C3 — The representation-rerouting intervention transfers to vision-language models (LLaVA-NeXT / Mistral backbone): after RR fine-tune on text-only paired data, the model's harmful-image ASR under PGD adversarial visual prompts is materially reduced while VQA capability is preserved.
**Date**: 2026-07-15
**verdict**: INCONCLUSIVE
**stage2_skip_reason**: —
**robustness**: — (Stage 2 never ran — Phase 2 gate FAIL)
**n_eligible**: 0
**n_run**: 0
**n_pass**: 0
**n_fail**: 0
**integrity_clean**: N/A (no variants)

## Verdict: INCONCLUSIVE

### Phase 2 combined verdict: FAIL

- Experiment audit: WARN
  - Full PGD adversarial-image optimisation and VLM assembly skipped (budget-gated; honestly disclosed in c3_verdict.json and EXPERIMENT_RESULTS.md)
  - M6 substep files (m1_mistral, m3_mistral, m4_mistral) exist with correct reported values
  - Scope is partial (Mistral text-only mechanism only; VLM visual-attack component not run)
  - Negative partial result honestly reported: delta_cos_harmful=-0.0116 (Mistral), reroute_passed=false

- Mechanism audit: FAIL (Check A — Steering Coefficient Sweep)
  - M6T (Mistral RR fine-tune): same Check A violations as M3 — alpha=10.0 single value (no sweep), no capability metric logged at sweep points, no sigma_proj scaling
  - Random-direction control n=1 (requires ≥30)
  - Reroute not demonstrated: delta_cos_harmful=-0.012, L_rr_ema=0.000222 (near-zero throughout)
  - Training instability mirrors M3: L_rr near-zero from step 0

### inconclusive_reason

main-experiment mechanism rigor broken — see verify/C3_vlm_transfer/main_experiment_audit/MECHANISM_AUDIT.md

### Iteration guidance

To convert INCONCLUSIVE to a testable state:
1. Fix M6T RR mechanism (same as M3 fix): run alpha sweep [0.1, 0.5, 1.0, 5.0, 10.0, 50.0], add capability metric, increase random-direction control to n≥30; this is prerequisite before the VLM visual-attack evaluation makes sense.
2. Once mechanism activates: complete the PGD adversarial-image optimisation and VLM assembly steps (budget-permitting).
3. Re-run Phase 2 audits; if combined verdict upgrades, C3 can proceed to Stage 2.

### Stage 2 plan (if C3 were admitted)

C3 was not in the Stage-2 pick pool (all claims rejected by Phase 2 gate). Had C1 been admitted and picked, C3 would remain INTEGRITY_ONLY (cap). No independent swap-variant plan for C3.
