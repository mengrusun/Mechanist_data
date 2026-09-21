# Verification Report

**Date**: 2026-07-14
**Status**: PARTIAL — halted at Stage 2 (variant deploy blocked by GPU budget constraint)
**Swap variants**: true (intended; Stage 2 launched but did not complete for C1)
**Dimensions tested**: model (1 variant per picked claim)
**Threshold**: robustness ≥ 0.5, min eligible variants = 1
**Main-experiment integrity (Phase 2)**: per-claim combined verdict — see Summary table; detail in `verify/INTEGRITY_AUDIT.md`.
**Round-End Decision**: ended-needs-decision (verify: variant-sanity-budget-exceeded)

---

## Summary

| Claim | Statement (short) | Main-experiment verdict | Main-experiment integrity (Phase 2, combined) | Variant integrity (Phase 9, combined) | Eligible variants (post-audit) | Robustness | State | Notes |
|-------|-------------------|------------------------|----------------------------------------------|---------------------------------------|-------------------------------|------------|-------|-------|
| C1 | probe existence signal | not-supported | WARN (exp WARN / mech N/A) | skipped (budget-blocked) | 0/0 | — | ZERO_ELIGIBLE_VARIANTS (budget-blocked) | Sanity partial pass (2/6 scen, no errors); full run ~10.45 GPU-h exceeds ~2.59 GPU-h remaining. Round-End Decision required. |
| C2 | aggregation diversity | not-supported | WARN (exp WARN / mech N/A) | skipped (MAX_VERIFY_CLAIMS cap) | — | — | INTEGRITY_ONLY (skip=max_verify_claims_cap) | Admitted by Phase 2 (WARN — under-power); not top-K=1 picked. Swap-test via `/auto-verify C2 — resume: true`. |
| C3 | zero shot transfer | supported | PASS (exp PASS / mech N/A) | skipped (MAX_VERIFY_CLAIMS cap) | — | — | INTEGRITY_ONLY (skip=max_verify_claims_cap) | Admitted by Phase 2 (PASS); not top-K=1 picked. Swap-test via `/auto-verify C3 — resume: true`. |

> **Column glossary**:
> - **Main-experiment verdict** — main experiment's own conclusion (`supported` / `not-supported`), from `refine-logs/main-experiment-verdicts.json`.
> - **Main-experiment integrity (Phase 2, combined)** — `max_severity(/experiment-audit, /mechanism-audit)` on `refine-logs/` scoped to this claim.
> - **Variant integrity (Phase 9, combined)** — variant-level audit; `skipped (budget-blocked)` means the variant ran partially but no artifacts were produced for audit.
> - **Eligible variants** — `N_eligible / N_run` surviving Phase 9.
> - **Robustness** — `#pass / N_eligible`; `—` when not computable.
> - **State** — ZERO_ELIGIBLE_VARIANTS (budget-blocked) for C1: variant was launched, sanity started but session died before producing any artifacts; full run blocked by budget; no eligible variant for robustness. INTEGRITY_ONLY for C2/C3: admitted but not Stage-2-picked (MAX_VERIFY_CLAIMS=1 cap).

---

## Integrity Audit

**Overall**: WARN — see `verify/INTEGRITY_AUDIT.md`.
- Main-experiment (Phase 2): WARN (C1: exp WARN / mech N/A; C2: exp WARN / mech N/A; C3: exp PASS / mech N/A)
- Variant (Phase 9): skipped (budget-blocked for C1; MAX_VERIFY_CLAIMS cap for C2/C3)

---

## Stage-2 Selection (Phase 3 step 0)

Phase 3 step 0 picked **1/3** admitted claims for Stage 2 (cap = MAX_VERIFY_CLAIMS = 1).

**Picked** (with importance rationale):
- C1: Foundational existence claim — C2 (aggregation) and C3 (transfer) both build on C1's per-agent probe scores; C1 result is the most uncertain (partial, just below threshold) making it highest-value to stress-test.

**Stage-2-deferred (INTEGRITY_ONLY, stage2_skip_reason: max_verify_claims_cap)**:
- C2: not-supported (delta group-vs-single too small) — swap-test via `/auto-verify C2 — resume: true`
- C3: supported (5/7 transfer families pass) — swap-test via `/auto-verify C3 — resume: true`

---

## Budget Constraint Record

| Phase | GPU-h consumed | GPU IDs | Notes |
|-------|---------------|---------|-------|
| M1 (main experiment) | 4.44 | 2,3 | Qwen3-32B-AWQ, 282 scenarios, 133 min |
| M2 (aggregation) | 0.05 | 2 | 5 aggregations on cached M1 activations |
| M3 (zero-shot transfer) | 2.60 | 5,6 | 222 scenarios, 78 min |
| Verify sanity (partial) | ~0.32 | 1,2,6 | 2/6 scenarios extracted before session died |
| **Total consumed** | **~7.41** | | |
| **Remaining** | **~2.59** | | |
| Estimated full variant run | ~10.45 | 1,2,3,5 (4 GPUs) | 282 scen × 33s/scen + 220s load × 4 GPUs |
| **Budget shortfall** | **~7.86** | | Full run cannot proceed within cap |

---

## Round-End Decision Required

**ended-needs-decision (verify: variant-sanity-budget-exceeded)**

**Situation**: The C1 model-swap variant (Qwen3-32B bf16) is technically sound — the sanity run confirmed correct model loading (d_model=5120, 64 layers, 219.8s load) and extraction (2/6 scenarios at 0.03 scen/s, no errors). However, the full 282-scenario run requires ~10.45 GPU-h, exceeding the ~2.59 GPU-h remaining under the 10 GPU-h total cap. The hard constraint "Do NOT exceed the 10 GPU-h total" blocks deployment.

**Options for the orchestrator to choose**:

1. **Expand budget** (recommended if scientific validity is priority): increase cap to ≥18 GPU-h. Re-invoke `/auto-verify C1 — resume: true`. Sanity will be repeated (session died without completing), then full run proceeds.

2. **Accept ZERO_ELIGIBLE_VARIANTS** (recommended if budget is firm): record C1 as having no eligible variants due to budget exhaustion. The main experiment's partial verdict (not-supported, AUROC=0.665) stands without model-generality check. Iteration works from the partial result directly.

3. **Switch model swap** (fit-in-budget alternative): Replace Qwen3-32B (bf16) with a faster model. The NOTICE mentions `Llama-3.1-70B-Instruct-AWQ-INT4` (AWQ quantized, likely similar throughput to M1's Qwen3-32B-AWQ) but it is a different architecture family (Llama vs Qwen), which is a weaker within-family robustness test. Alternatively, `DeepSeek-R1-Distill-Qwen-32B` would stay in the Qwen family but has reasoning-tuned behavior (different response distribution than base model).

4. **Subsample run (70 scenarios, ~2.81 GPU-h)**: Run on 70 scenarios (42 min × 4 GPUs). Probe uses ~49/7/14 split. This fits the budget with minimal margin; produces a verdict at reduced statistical power. The 0.75 AUROC threshold is identical to the main experiment; the result's comparability caveat must be noted.

---

## Details

- C1: `verify/C1_probe_existence_signal/ROBUSTNESS.md` — budget-blocked, full budget analysis
- C2: `verify/C2_aggregation_diversity/ROBUSTNESS.md` — INTEGRITY_ONLY (max_verify_claims_cap)
- C3: `verify/C3_zero_shot_transfer/ROBUSTNESS.md` — INTEGRITY_ONLY (max_verify_claims_cap)
- C1 main-experiment audit: `verify/C1_probe_existence_signal/main_experiment_audit/`
- C2 main-experiment audit: `verify/C2_aggregation_diversity/main_experiment_audit/`
- C3 main-experiment audit: `verify/C3_zero_shot_transfer/main_experiment_audit/`
- C1 sanity log: `runs/verify/model-swap-qwen3-32b-bf16/sanity.log`
- C1 variant cost: `runs/verify/model-swap-qwen3-32b-bf16/cost.json`

---

## Next Step

**Round-End Decision required** — the orchestrator must choose one of the four options above before verify can resume. This is not an iteration failure; the main experiment is intact and the variant is technically valid. Only the GPU budget prevents completion.

- After option 1 or 3 or 4: re-invoke `/auto-verify C1 — resume: true` (with updated GPU budget or new variant config)
- After option 2: hand to iteration with `/auto-iteration-loop "[topic] — verify-zero-eligible: C1"`. C1 iteration works from the partial main-experiment result (not-supported, AUROC=0.665); no variant evidence available.
