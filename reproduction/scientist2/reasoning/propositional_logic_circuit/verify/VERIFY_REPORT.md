# Verification Report

**Date**: 2026-07-15
**Swap variants**: true (full Stage 1–3 pipeline)
**Dimensions tested**: model  (variants/claim = 1; model-swap only)
**Threshold**: robustness ≥ 0.5, min eligible variants = 1
**Main-experiment integrity (Phase 2)**: per-claim combined verdict — see Summary table; per-sub-audit breakdown in `INTEGRITY_AUDIT.md`.
**M5 reuse**: the C3 model-swap variant reused `results/M5_gemma9b.json` (M5 milestone) — identical activation-patching pipeline, identical anchor cell, Gemma-2-9B model. No fresh GPU run dispatched (~0.67 GPU-h saved).

## Summary

| Claim | Statement (short) | Main-experiment verdict | Main-experiment integrity (Phase 2, combined) | Variant integrity (Phase 9, combined) | Eligible variants (post-audit) | Robustness | State | Notes |
|-------|-------------------|------------------------|----------------------------------------------|---------------------------------------|--------------------------------|------------|-------|-------|
| C1 | sparse component set | not-supported | WARN (exp WARN / mech N/A) | skipped (MAX_VERIFY_CLAIMS cap) | — | — | INTEGRITY_ONLY (skip=max_verify_claims_cap) | admitted by Phase 2 but not top-K=1 picked — run `/auto-verify C1 — resume: true` to swap-test |
| C2 | modular decomposition | not-supported | WARN (exp WARN / mech N/A) | skipped (MAX_VERIFY_CLAIMS cap) | — | — | INTEGRITY_ONLY (skip=max_verify_claims_cap) | admitted by Phase 2 but not top-K=1 picked — run `/auto-verify C2 — resume: true` to swap-test |
| C3 | necessity sufficiency | not-supported | WARN (exp WARN / mech N/A) | clean (1/1 PASS) | 1/1 | 1.00 | PASS `[MAIN-EXPERIMENT INTEGRITY: WARN — experiment]` | necessity-yes/sufficiency-no pattern is cross-family robust (Mistral-7B + Gemma-2-9B); M5 result reused as model-swap variant |

> **Column glossary**:
> - **Main-experiment verdict**: main experiment's own conclusion on the claim (`supported` / `not-supported`), from `refine-logs/main-experiment-verdicts.json`.
> - **Main-experiment integrity (Phase 2, combined)**: `max_severity(/experiment-audit, /mechanism-audit)` on `refine-logs/` scoped to this claim. All three claims: exp=WARN, mech=N/A (no additive steering coefficient in any claim's pipeline — all use replacement activation patching). Combined=WARN for all.
> - **Variant integrity (Phase 9, combined)**: only C3 had variants run; model-swap-gemma2-9b passed integrity cleanly (PASS). C1/C2 skipped.
> - **Eligible variants (post-audit)**: C3: 1/1 (model-swap-gemma2-9b integrity=pass). C1/C2: — (no variants ran).
> - **Robustness**: C3: 1.0 (1 pass / 1 eligible). C1/C2: — (no variants).
> - **State**: C3=PASS (robustness 1.0 ≥ 0.5 threshold); C1=INTEGRITY_ONLY, C2=INTEGRITY_ONLY (both admitted by Phase 2 WARN, both deferred by MAX_VERIFY_CLAIMS=1 cap — C3 was the top-K pick).

## Integrity Audit

**Overall**: WARN — see `verify/INTEGRITY_AUDIT.md` for full Phase 2 (main experiment) + Phase 9 (variants) findings.

**Phase 2 WARN sources (per claim)**:
- C1: undisclosed minimality sampling (20/158 components sampled in minimality sweep; reported as average but sampling not disclosed in per-claim verdict text).
- C2: shortlist scope reduction (top-40 of 158 components used in M4/M4.stab; disclosed in Notes but claim predicate references "the shortlisted components" i.e. all 158).
- C3: KL recovery scaling artifact (KL baseline=0.043 too small; recovery_KL=-5.618 in M2 is an artifact; documented in Notes but not elevated to per-claim verdict box; plan criterion requires all three metrics).

**Phase 9 PASS** — the model-swap variant for C3 is integrity-clean on all checks.

## Stage-2 Selection

Phase 3 step 0 picked 1 of 3 admitted claims for Stage 2 (cap = MAX_VERIFY_CLAIMS = 1).

**Picked** (with importance rationale):
- C3: The necessity + sufficiency claim is the most scientifically central and surprising finding — the necessity-yes/sufficiency-no asymmetry is the study's publishable core result, and the model-swap stress-test directly tests whether it is a Mistral-specific artifact or a cross-family property. Highest scientific stakes.

**Stage-2-deferred (marked INTEGRITY_ONLY with stage2_skip_reason: max_verify_claims_cap)**:
- C1: sparse component set — main-experiment verdict: not-supported — swap-test later via `/auto-verify C1 — resume: true`
- C2: modular decomposition — main-experiment verdict: not-supported — swap-test later via `/auto-verify C2 — resume: true`

## Details

- C1: `verify/C1_sparse_component_set/ROBUSTNESS.md`
- C2: `verify/C2_modular_decomposition/ROBUSTNESS.md`
- C3: `verify/C3_necessity_sufficiency/ROBUSTNESS.md` (full directory: `C3_necessity_sufficiency/`)

## Next Step

→ **C3 PASSes** — the necessity-yes/sufficiency-no conclusion is robust across the model dimension (Mistral-7B → Gemma-2-9B). The robustly-negative finding (C3 not-supported across both families) goes into the next-round draft as the cross-family established result.

→ **C1 and C2 INTEGRITY_ONLY** (not swap-tested this pass) — iteration records them under Open Items:
  - C1: `/auto-verify C1 — resume: true` (Phase 2 audit reused via RESUME; model-swap with Gemma-2-9B would test whether sparsity + completeness pattern recurs)
  - C2: `/auto-verify C2 — resume: true` (Phase 2 audit reused via RESUME; model-swap with Gemma-2-9B would test whether the absence-of-modularity pattern recurs)
  INTEGRITY_ONLY does not consume iteration budget and does not block next-round entry.

→ Proceed to: `/auto-iteration-loop "sparse modular circuit for propositional-logic reasoning"` with C3 PASS, C1+C2 INTEGRITY_ONLY (no back-edge action needed for iteration; open items only).
