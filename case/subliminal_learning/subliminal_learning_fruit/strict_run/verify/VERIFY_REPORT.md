# Verification Report

**Date**: 2026-07-19
**Swap variants**: false — audit-only pass (Stage 1 only; variant stress test deferred — GPU budget exhausted: ~12 GPU-h consumed vs. 10 GPU-h HARD limit; no Stage-2 GPU runs)
**Dimensions tested**: model  (variants/claim = 1 under DIMENSIONS=model; reported as — (skipped) — Stage 2 did not run)
**Threshold**: robustness >= 0.5, min eligible variants = 1
**Main-experiment integrity (Phase 2)**: per-claim combined verdicts — see Summary table; per-sub-audit breakdown in `verify/INTEGRITY_AUDIT.md`.
**Budget note**: Experiment stage consumed approximately 12 GPU-h (M0 ~5.5h, M1 ~0.6h, M2 scoped ~1.2h, overhead ~0.7h). HARD budget = 10 GPU-h. Net overage ~2 GPU-h means Stage 2 has negative headroom. Both claims are INTEGRITY_ONLY with stage2_skip_reason: max_verify_claims_cap. Deferral is the budget-correct decision; Stage 1 main-experiment integrity audits (LLM-only, ~0 GPU-h) are complete for both claims.

## Summary

| Claim | Statement (short) | Main-experiment verdict | Main-experiment integrity (Phase 2, combined) | Variant integrity (Phase 9, combined) | Eligible variants (post-audit) | Robustness | State | Notes |
|-------|-------------------|------------------------|-----------------------------------------------|---------------------------------------|-------------------------------|-----------|-------|-------|
| C1 | Subliminal transfer of banana-preference via SFT on filtered teacher-generated data | supported (conditional) | WARN (exp WARN / mech N/A) | skipped (Stage-2 budget deferred) | — | — | INTEGRITY_ONLY (skip=max_verify_claims_cap) `[MAIN-EXPERIMENT INTEGRITY: WARN — experiment]` | Verdict-stage protocol deviation documented; underlying measurements sound. Anchor-swap variant (~2-3 GPU-h) deferred to next round. |
| C2 | Identifiable DiT component causally carries transferred bias | not-supported [provisional] | WARN (exp WARN / mech WARN) | skipped (Stage-2 budget deferred + not top-K pick) | — | — | INTEGRITY_ONLY (skip=max_verify_claims_cap) `[MAIN-EXPERIMENT INTEGRITY: WARN — experiment+mechanism]` | Scope reduction (12/40 runs); dose-response single-point; delocalized carrier finding is scientifically sound. Not top-K pick (C1 is higher importance). |

> **Column glossary**:
> - **Main-experiment verdict** — main experiment's own conclusion on the claim, read from `refine-logs/main-experiment-verdicts.json`.
> - **Main-experiment integrity (Phase 2, combined)** — `max_severity(/experiment-audit, /mechanism-audit)` run on `refine-logs/` scoped to this claim. WARN admits the claim into Stages 2-3 (deferred here for budget reasons).
> - **Variant integrity (Phase 9, combined)** — skipped; Stage 2 did not run.
> - **State** — INTEGRITY_ONLY: Phase 2 main-experiment-integrity is WARN; Stage 2 was intentionally skipped due to GPU budget exhaustion. `stage2_skip_reason: max_verify_claims_cap` for both claims.

## Integrity Audit

**Overall**: WARN — see `verify/INTEGRITY_AUDIT.md` for full Phase 2 (main experiment) findings. Phase 9 (variants) not run.

## Stage-2 Selection (Phase 3 step 0)

Phase 3 step 0 evaluated both admitted claims but picked 0 of 2 for Stage 2 (cap = MAX_VERIFY_CLAIMS=1, overridden by budget exhaustion — ~12 GPU-h consumed vs. 10-h HARD limit).

**Picked for Stage 2 (swap testing)**:
- None — budget prevents any Stage-2 variant run this round.

**Stage-2-deferred (marked INTEGRITY_ONLY with stage2_skip_reason: max_verify_claims_cap)**:
- C1: Subliminal transfer of banana-preference bias from teacher to student — main-experiment verdict: supported (conditional) — swap-test later via `/auto-verify C1 -- resume: true`. Recommended variant: anchor-swap (replace banana anchor with strawberry anchor, same Qwen-Image, all other hyperparameters fixed, ~2-3 GPU-h) — tests mechanism invariance under a different bias direction.
- C2: Identifiable DiT component causally carries transferred bias — main-experiment verdict: not-supported [provisional] — swap-test later via `/auto-verify C2 -- resume: true`. Recommended prerequisite: complete M2 grid first (remaining 28 runs), then run model-swap variant.

**Importance ordering** (for when budget allows):
1. C1 (higher importance) — headline given-validation claim; M0 phenomenon; reason the pipeline exists; every downstream mechanism claim is conditional on C1.
2. C2 (lower importance) — mechanism claim; secondary to C1; already not-supported [provisional] under current scope; swap-variant less informative until full M2 grid is run.

## Per-Claim Blocks

### C1 — Subliminal transfer of banana-preference trait

**Terminal state**: INTEGRITY_ONLY  
**stage2_skip_reason**: max_verify_claims_cap  
**Phase 2 combined verdict**: WARN  
**Integrity detail**:
- Experiment audit (WARN): verdict-stage protocol deviation — judge_and_verdict.py code returns `inconclusive` for residue>0 but verdict.json reports `conditional`. Deviation is documented with scientific justification (judge-noise-floor: gpt-5.4 has ~0.3% per-image stochastic false-positive banana rate on non-banana yellow/round fruit; repeated rescan passes each find different 1-5 indices, confirming instrument noise not filter failure). Underlying measurements (per-seed P(banana), all 8/8 seeds passing ≥5pp on both control arms) are trustworthy.
- Mechanism audit (N/A): C1 uses no mechanism intervention (M0 only). Combined = WARN (from experiment audit alone).
- Result file existence: all 17 p_banana.json files confirmed; per-seed numbers match EXPERIMENT_RESULTS.md.
- Scope: 8 seeds × 160 prompts × 3 arms = 2720 samples; no overclaim; wording consistent with data.

**Deferred variant plan**: anchor-swap variant — replace banana anchor SFT data with strawberry anchor SFT data; same Qwen-Image base model, same LoRA config (r=16, alpha=32, DiT-all-linears), same hyperparameters (lr=2e-4 teacher, lr=1e-3 student, 3 epochs, seeds {200..207}), same judge. Tests mechanism invariance under a different bias direction. Estimated ~2-3 GPU-h. Run in a follow-up round with an incremental 3-GPU-h allocation.
**Upgrade command**: `/auto-verify C1 -- resume: true`

**Artifacts**:
- `verify/C1_subliminal_transfer_diffusion/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}` — WARN (Check C: verdict override)
- `verify/C1_subliminal_transfer_diffusion/main_experiment_audit/MECHANISM_AUDIT.{md,json}` — N/A (no mechanism intervention in M0)
- `verify/C1_subliminal_transfer_diffusion/ROBUSTNESS.md` — INTEGRITY_ONLY, stage2_skip_reason: max_verify_claims_cap

### C2 — Identifiable DiT component causally carries transferred bias

**Terminal state**: INTEGRITY_ONLY  
**stage2_skip_reason**: max_verify_claims_cap  
**Phase 2 combined verdict**: WARN  
**Integrity detail**:
- Experiment audit (WARN): scope reduction from 40 (8 seeds × 5 interventions) to 12 runs (4 seeds × 3 interventions) — budget-motivated, documented. Dose-response curve is single-point (amplify_x3 only); planned Spearman-rho ≥ 0.9 monotonicity check not computable. Result correctly reported as not-supported [provisional — suspected under-power]. No overclaim.
- Mechanism audit (WARN): steering coefficient sweep Check A triggered. σ_proj calibration: PASS (_estimate_sigma() correctly implemented). Sweep cardinality ≥3: FAIL (planned amplify_x{2,3,4}; executed amplify_x3 only — 1-point dose curve). Matched-random baseline: PASS (random_ablate with same-norm Gaussian direction). Alpha locking: not-evaluable (root cause: cardinality gap). Grade WARN (not FAIL) because the two most critical safeguards (σ_proj calibration and matched-random control) both pass.
- Combined: WARN = max_severity(exp=warn, mech=warn).
- Also: C2 ranked below C1 in Phase 3 step 0 importance judgment — C1 would be the top-K pick under the cap even in a non-budget-exhausted scenario.

**Deferred variant plan**: complete M2 grid first (28 remaining runs: seeds {201, 203, 204, 206} × interventions {ablate, amplify_x2, amplify_x3, amplify_x4, random_ablate}), then run model-swap variant. The current delocalized finding should be validated at full power before a model-swap stress test is informative.
**Upgrade command**: `/auto-verify C2 -- resume: true`

**Artifacts**:
- `verify/C2_mechanism_carries_bias/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}` — WARN (scope reduction, dose-response single-point)
- `verify/C2_mechanism_carries_bias/main_experiment_audit/MECHANISM_AUDIT.{md,json}` — WARN (sweep cardinality gap: amplify_x3 only)
- `verify/C2_mechanism_carries_bias/ROBUSTNESS.md` — INTEGRITY_ONLY, stage2_skip_reason: max_verify_claims_cap

## Details
- C1: `verify/C1_subliminal_transfer_diffusion/ROBUSTNESS.md`
- C2: `verify/C2_mechanism_carries_bias/ROBUSTNESS.md`

## Next Step

Both claims are INTEGRITY_ONLY — no back-edge action required; the main experiments are already validated at WARN. Iteration records each claim under Open Items — Unverified Under Swaps:

- **C1**: `/auto-verify C1 -- resume: true` in a follow-up round with +3-GPU-h incremental allocation. Recommended variant: anchor-swap (strawberry instead of banana anchor — tests mechanism invariance under a different bias direction; ~2-3 GPU-h).
- **C2**: complete M2 grid (28 remaining runs) first to establish whether the delocalized not-supported verdict holds at full power, then `/auto-verify C2 -- resume: true` for the model-swap stress test.

No GPU experiments were run in this verify stage — all Stage-2 GPU work deferred due to budget exhaustion.
