# Verify Report — exp18 (Belief-Circuit Reproduction on Pythia)

**Date:** 2026-07-22  
**Dimensions:** model  
**MAX_VERIFY_CLAIMS:** 1  
**ROBUSTNESS_THRESHOLD:** 0.50  
**MIN_VARIANTS_FOR_VERDICT:** 1  
**TARGET_CLAIMS:** all (C1, C2, C3, C4)

---

## Per-Claim Verdicts

| Claim | Short text | Baseline verdict | Robustness | Eligible | Variants | Integrity | Final |
|-------|------------|------------------|------------|----------|----------|-----------|-------|
| C1 | Scale-dependent emergence (≥1B) | supported | — | — | — | PASS | INTEGRITY_ONLY |
| C2 | Belief heads localizable | supported | 0.00 (0/1) | 1/1 | 0 pass / 1 fail | WARN | **FAIL** |
| C3 | Formation window (step 40k-80k) | supported | — | — | — | WARN | INTEGRITY_ONLY |
| C4 | Dynamic controllability | supported | — | — | — | PASS | INTEGRITY_ONLY |

### Detailed per-claim entries

**C1 (Scale-dependent emergence):** baseline-verdict supported — robustness — (threshold 0.50) — eligible —/— — variants — — integrity clean — INTEGRITY_ONLY (skip=max_verify_claims_cap)

**C2 (Belief heads localizable):** baseline-verdict supported — robustness 0.00 (threshold 0.50) — eligible 1/1 — variants 0 pass / 1 fail — integrity 1 warn — FAIL

**C3 (Formation window 40k-80k):** baseline-verdict supported — robustness — (threshold 0.50) — eligible —/— — variants — — integrity 1 warn (scope: only 24/154 checkpoints audited) — INTEGRITY_ONLY (skip=max_verify_claims_cap)

**C4 (Dynamic controllability):** baseline-verdict supported — robustness — (threshold 0.50) — eligible —/— — variants — — integrity clean — INTEGRITY_ONLY (skip=max_verify_claims_cap)

---

## Stage-2 Selection

Phase 3 step 0 applied MAX_VERIFY_CLAIMS=1 cap to the admitted pool of {C1, C2, C3, C4}.

**Picked (Stage 2 proceeds):** C2 — selected as highest-importance claim (localization is the mechanistic core of the belief-circuit hypothesis; robustness of the head-specific localization finding is the primary risk for the scientific contribution).

**Stage-2 deferred (INTEGRITY_ONLY, stage2_skip_reason: max_verify_claims_cap):** C1, C3, C4.

See `verify/STAGE2_PICK.json` for full rationale.

---

## Summary

**0 PASS, 1 FAIL, 0 INCONCLUSIVE, 0 ZERO_ELIGIBLE_VARIANTS, 3 INTEGRITY_ONLY** (of 4 target claims; INTEGRITY_ONLY breakdown: 0 stage2_skip_reason=swap_variants_false + 3 stage2_skip_reason=max_verify_claims_cap).

**C2 failure analysis:** OLMo-1B (16L/16H/2048D) failed to localize belief heads for both personal_belief and attributed_belief targets:

- **personal_belief**: Fisher-ranked heads cause accuracy drops meeting C2a (≥30%) but simultaneously violate C2c (off-target beliefs also drop ≥10%) and C2d (PPL degrades catastrophically, up to 46× clean). The heads that encode personal_belief in OLMo-1B appear to be entangled with general LM computation.

- **attributed_belief**: All 30 greedy steps show *negative* accuracy drops (ablating top Fisher heads *increases* attributed_belief accuracy). The Fisher-selected heads function as suppression heads in OLMo-1B rather than encoding heads, inverting the causal direction assumed by C2a. Jackknife rho=0.954 (very consistent Fisher estimate) confirms this is a stable finding.

**Interpretation:** The localization finding appears to be Pythia-architecture-specific and does not generalize to OLMo-1B in the same size class. Architectural differences in how attention heads are organized (dense vs. split projections, different pre-norm configurations, different training data) may explain the divergence. C2 is **fragile under model swap**.

---

## Artifacts

- `verify/VERIFY_REPORT.md` — this file
- `verify/INTEGRITY_AUDIT.md` — Phase 2 baseline + Phase 9 variant sections
- `verify/STAGE2_PICK.json` — Phase 3 step 0 pick record
- `verify/C2_belief_heads_localization/baseline_audit/EXPERIMENT_AUDIT.{md,json}`
- `verify/C2_belief_heads_localization/baseline_audit/MECHANISM_AUDIT.{md,json}`
- `verify/C2_belief_heads_localization/variant_audit/EXPERIMENT_AUDIT.{md,json}`
- `verify/C2_belief_heads_localization/variant_audit/MECHANISM_AUDIT.{md,json}`
- `verify/C2_belief_heads_localization/ROBUSTNESS.md`
- `verify/C2_belief_heads_localization/variants/model-swap-olmo-1b/result.json`
- `verify/C2_belief_heads_localization/variants/model-swap-olmo-1b/run.log`
- `runs/model-swap-olmo-1b/cost.json` — 4730s wall clock, 1.314 GPU-hours (GPU 1, A800-SXM4-80GB)
- `verify/C1_scale_dependent_emergence/ROBUSTNESS.md` — INTEGRITY_ONLY
- `verify/C3_formation_window/ROBUSTNESS.md` — INTEGRITY_ONLY
- `verify/C4_dynamic_controllability/ROBUSTNESS.md` — INTEGRITY_ONLY

---

## Upgrade Commands

To swap-test the deferred claims (one at a time, reusing Phase 2 audits):

```
/auto-verify C1 -- resume: true, max_verify_claims: 1, dimensions: model
/auto-verify C3 -- resume: true, max_verify_claims: 1, dimensions: model
/auto-verify C4 -- resume: true, max_verify_claims: 1, dimensions: model
```
