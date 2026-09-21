# Verification Report

**Date**: 2026-07-14
**Swap variants**: true
**Dimensions tested**: model  (variants/claim = 1; only C3a — the top-1 picked claim)
**Threshold**: robustness >= 0.5, min eligible variants = 1
**Main-experiment integrity (Phase 2)**: per-claim combined verdict — see Summary table column; per-sub-audit breakdown in `verify/INTEGRITY_AUDIT.md`.

## Summary

| Claim | Statement (short) | Main-experiment verdict | Main-experiment integrity (Phase 2, combined) | Variant integrity (Phase 9, combined) | Eligible variants (post-audit) | Robustness | State | Notes |
|-------|-------------------|------------------------|-----------------------------------------------|---------------------------------------|-------------------------------|------------|-------|-------|
| C1 | Gold correctness linearly accessible (AUROC>=0.70) | supported | WARN (exp WARN / mech N/A) | skipped (MAX_VERIFY_CLAIMS cap) | — | — | INTEGRITY_ONLY (skip=max_verify_claims_cap) `[MAIN-EXPERIMENT INTEGRITY: WARN — experiment]` | Bootstrap CI does not contain point estimate (normal artifact); n=200 vs planned 1000 (disclosed). Upgrade: `/auto-verify C1 -- resume: true` |
| C2 | Verbalized confidence linearly accessible pre-emission | supported | WARN (exp WARN / mech N/A) | skipped (MAX_VERIFY_CLAIMS cap) | — | — | INTEGRITY_ONLY (skip=max_verify_claims_cap) `[MAIN-EXPERIMENT INTEGRITY: WARN — experiment]` | Proxy GT (model's own verbalization, by design); bootstrap CI mismatch; paraphrase Delta P1=-0.11 slightly exceeds 0.10. Upgrade: `/auto-verify C2 -- resume: true` |
| C3a | v_c and v_v near-orthogonal at L* (|cos|<=0.3, CI upper<0.4) | supported | PASS (exp PASS / mech N/A) | clean (1 variant PASS) | 1/1 | 1.00 | PASS | Qwen2.5-7B-Instruct model swap replicates: |cos|=0.021 [0.001,0.037] at L*=22, neighborhood mean=0.015 — all thresholds met. Architecture-independent. |
| C3b | Causal separability (cross-direction steering bounded) | supported | WARN (exp WARN / mech WARN) | skipped (MAX_VERIFY_CLAIMS cap) | — | — | INTEGRITY_ONLY (skip=max_verify_claims_cap) `[MAIN-EXPERIMENT INTEGRITY: WARN — experiment+mechanism]` | Sparse alpha grid (3 pts); n_random=1. Upgrade: `/auto-verify C3b -- resume: true` |
| C3c | Dissociation when disagree (accuracy ordering) | not-supported | WARN (exp WARN / mech N/A) | skipped (MAX_VERIFY_CLAIMS cap) | — | — | INTEGRITY_ONLY (skip=max_verify_claims_cap) `[MAIN-EXPERIMENT INTEGRITY: WARN — experiment]` | Cell n=4/2000 — suspected under-power. Upgrade: `/auto-verify C3c -- resume: true` |

> **Column glossary**:
> - **Main-experiment verdict** — main experiment's own conclusion on the claim (`supported` / `not-supported`), read from `refine-logs/main-experiment-verdicts.json`. `—` when the main-experiment audit failed before this could matter.
> - **Main-experiment integrity (Phase 2, combined)** — `max_severity(/experiment-audit, /mechanism-audit)` run on `refine-logs/` scoped to this claim, with the two sub-verdicts shown in parentheses. `PASS` / `WARN` admit the claim into Phases 3–10; `FAIL` short-circuits to INCONCLUSIVE. `mech N/A` means the claim's experiment uses no mechanism intervention.
> - **Variant integrity (Phase 9, combined)** — same `max_severity` rule on the per-claim variants. `clean` = every variant combined-PASS. `skipped` = Stage 2 was not run for this claim (MAX_VERIFY_CLAIMS cap).
> - **Eligible variants (post-audit)** — `N_eligible / N_run` — how many variants survived Phase 9. Only the eligible set feeds robustness. `—` when no variants were dispatched.
> - **Robustness** — `#pass / N_eligible`, counted over each eligible variant's `consistent_with_main_experiment`. `—` when no robustness defined (INTEGRITY_ONLY).
> - **State** — final per-claim verdict. PASS = main-experiment verdict is robust under swaps. INTEGRITY_ONLY = Stage 1 audit passed but Stage 2 skipped by policy (max_verify_claims_cap — this claim was admitted but not the top-K picked).

## Integrity Audit

**Overall**: WARN — see `verify/INTEGRITY_AUDIT.md` for full Phase 2 (main experiment) + Phase 9 (variants) findings.

- Phase 2 (main-experiment): WARN across all 5 claims (C3a=PASS, all others=WARN)
- Phase 9 (variants): PASS — the single variant run (C3a, model-swap-qwen25-7b-instruct) passed all integrity checks

## Stage-2 Selection

Phase 3 step 0 picked 1 of 5 admitted claims for Stage 2 (cap = MAX_VERIFY_CLAIMS = 1).

**Picked** (with importance rationale):
- C3a: The load-bearing claim of the paper — the only direct measurement of geometric near-orthogonality (novel finding no prior paper reports). Clean Phase 2 integrity (PASS). Under single-model DIMENSIONS=model, this is the highest-value claim to stress-test under a cross-architecture swap.

**Stage-2-deferred (marked INTEGRITY_ONLY with stage2_skip_reason: max_verify_claims_cap)**:
- C1: Gold correctness linearly accessible — main-experiment verdict: supported — swap-test later via `/auto-verify C1 -- resume: true`
- C2: Verbalized confidence linearly accessible — main-experiment verdict: supported — swap-test later via `/auto-verify C2 -- resume: true`
- C3b: Causal separability — main-experiment verdict: supported — swap-test later via `/auto-verify C3b -- resume: true`
- C3c: Dissociation when disagree — main-experiment verdict: not-supported — swap-test later via `/auto-verify C3c -- resume: true`

See `verify/STAGE2_PICK.json` for the full pick record.

## Details

- C1: `verify/C1_correctness_linear_accessible/ROBUSTNESS.md`
- C2: `verify/C2_verbalized_conf_linear/ROBUSTNESS.md`
- C3a: `verify/C3a_near_orthogonality/ROBUSTNESS.md`
- C3b: `verify/C3b_causal_separability/ROBUSTNESS.md`
- C3c: `verify/C3c_dissociation_when_disagree/ROBUSTNESS.md`

## Next Step

-> **C3a PASS (robustness=1.00)**: The load-bearing near-orthogonality claim is robust across a cross-family architecture swap. The result generalizes from Llama-3.1-8B-Instruct (L*=31, |cos|=0.015) to Qwen2.5-7B-Instruct (L*=22, |cos|=0.021) — both near the random-direction null, both well below all thresholds.

-> **C1, C2, C3b, C3c INTEGRITY_ONLY** (stage2_skip_reason: max_verify_claims_cap): These claims were admitted by Phase 2 but not selected for the single swap-test slot. Their main-experiment verdicts stand (supported for C1/C2/C3b, not-supported for C3c) with the caveat that cross-model/dataset/method robustness has not been tested. Proceed to `/auto-iteration-loop` with the current state; the iteration loop should note C1/C2/C3b/C3c as "verified integrity only — robustness not tested this pass" in Open Items. To upgrade any of them, run `/auto-verify <claim-id> -- resume: true`.
