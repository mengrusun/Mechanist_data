# Verification Report

**Date**: 2026-07-15
**Swap variants**: true (full 3-stage pipeline; Stage 1 audits all 5 claims; Stage 2/3 run on C1 only)
**Dimensions tested**: model (1 variant per picked claim — 1 total variant run)
**Threshold**: robustness >= 0.5, min eligible variants = 1
**Main-experiment integrity (Phase 2)**: per-claim combined verdict — see Summary table; per-sub-audit breakdown in `verify/INTEGRITY_AUDIT.md`.

## Summary

| Claim | Statement (short) | Main-experiment verdict | Main-experiment integrity (Phase 2, combined) | Variant integrity (Phase 9, combined) | Eligible variants (post-audit) | Robustness | State | Notes |
|-------|-------------------|------------------------|----------------------------------------------|--------------------------------------|-------------------------------|------------|-------|-------|
| C1 | h and r directions distinct | not-supported | WARN (exp WARN / mech N/A) | WARN (1 variant: auroc_r=NaN expected) | 1/1 | 1.00 | PASS | Robustly not-supported — Qwen2 replicates geometry (cos_ratio=0.069) + refusal-side NaN; partial finding holds cross-model |
| C2 | Position dissociation | not-supported | WARN (exp WARN / mech N/A) | skipped (MAX_VERIFY_CLAIMS cap) | — | — | INTEGRITY_ONLY (skip=max_verify_claims_cap) | Admitted by Phase 2; not top-K pick; upgrade via `/auto-verify C2 — resume: true` |
| C3 | Causal steering dissociation | supported | FAIL (exp WARN / mech FAIL) | — | — | — | INCONCLUSIVE | main-experiment mechanism rigor broken — steering alpha not in sigma_proj units; no plateau; n_random=1; see verify/C3_causal_steering_dissociation/main_experiment_audit/MECHANISM_AUDIT.md |
| C4 | Jailbreak signature | not-supported | WARN (exp WARN / mech N/A) | skipped (MAX_VERIFY_CLAIMS cap) | — | — | INTEGRITY_ONLY (skip=max_verify_claims_cap) | Admitted by Phase 2; not top-K pick; ASR=0 so claim untestable; upgrade via `/auto-verify C4 — resume: true` |
| C5 | Probe beats Llama Guard | supported | FAIL (exp FAIL / mech N/A) | — | — | — | INCONCLUSIVE | main-experiment integrity broken — scope mismatch: claim says "jailbreaks" but test set has 0 jailbreaks (M4 ASR=0); see verify/C5_probe_beats_llamaguard/main_experiment_audit/EXPERIMENT_AUDIT.md |

> **Column glossary:**
> - **Main-experiment verdict**: main experiment's own conclusion (`supported`/`not-supported`), from `refine-logs/main-experiment-verdicts.json`.
> - **Main-experiment integrity (Phase 2, combined)**: `max_severity(exp, mech)` with sub-verdicts shown in parentheses. `mech N/A` = claim uses no steering intervention.
> - **Variant integrity (Phase 9, combined)**: WARN = variant passed integrity but with minor caveat. `skipped` = variants not dispatched (Stage 2 not reached for this claim).
> - **Eligible variants (post-audit)**: variants with integrity_status ∈ {pass, warn}. Integrity-FAIL excluded from both numerator and denominator.
> - **Robustness**: #pass / N_eligible over `consistent_with_main_experiment`. `—` when no robustness is computable.
> - **State**: PASS / FAIL / INCONCLUSIVE / ZERO_ELIGIBLE_VARIANTS / INTEGRITY_ONLY.

## Integrity Audit

**Overall**: FAIL — see `verify/INTEGRITY_AUDIT.md` for full Phase 2 + Phase 9 findings.
- Phase 2 (main-experiment): FAIL (C3 mechanism broken; C5 experiment scope broken)
- Phase 9 (variant integrity): WARN (1 variant, clean with minor NaN caveat)

## Stage-2 Selection

Phase 3 step 0 picked 1 of 3 admitted claims for Stage 2 (cap = MAX_VERIFY_CLAIMS = 1).

**Admitted pool**: C1 (WARN), C2 (WARN), C4 (WARN)
**Rejected pool (INCONCLUSIVE)**: C3 (FAIL — mechanism), C5 (FAIL — experiment scope)

**Picked** (with importance rationale):
- C1: The foundational geometric claim — existence of distinct h and r directions — is the necessary precondition for every downstream claim (C2 position-crossover, C3 causal dissociation, C4 jailbreak signature, C5 probe). Among the admitted claims, C1 is the most scientifically central. The partial main-experiment verdict (geometry passes, refusal-side unmeasurable at 98.7% bare-refusal) is precisely the case where a model-swap to Qwen2-Instruct-7B provides decisive new information about cross-model generalization.

**Stage-2-deferred (INTEGRITY_ONLY with stage2_skip_reason: max_verify_claims_cap)**:
- C2: Position dissociation — not-supported; informative negative; upgrade via `/auto-verify C2 — resume: true`
- C4: Jailbreak signature — not-supported (ASR=0); upgrade via `/auto-verify C4 — resume: true`

## Details

- C1: `verify/C1_h_r_directions_distinct/ROBUSTNESS.md`
- C2: `verify/C2_position_dissociation/ROBUSTNESS.md`
- C3: `verify/C3_causal_steering_dissociation/ROBUSTNESS.md`
- C4: `verify/C4_jailbreak_signature/ROBUSTNESS.md`
- C5: `verify/C5_probe_beats_llamaguard/ROBUSTNESS.md`

## Next Step

→ **C1 PASS** — the not-supported (partial) result is robust across Llama-3 and Qwen2 model families. The geometry finding (distinct h and r directions) generalizes; the refusal-side unmeasurability at high bare-refusal rate is a dataset+model alignment issue, not a model-specific artifact.

→ **C3 INCONCLUSIVE** (main-experiment mechanism rigor broken) → hand to iteration: fix the steering sweep:
  - Re-sweep alpha in sigma_proj units spanning >=3 orders of magnitude (e.g., [0.03, 0.1, 0.3, 1.0, 3.0] sigma_proj)
  - Run n_random >= 30 matched-norm control directions
  - Identify and lock alpha mid-plateau (both target effect stable and capability metric within tolerance)
  - Then re-invoke `/auto-verify C3 — resume: true`

→ **C5 INCONCLUSIVE** (main-experiment experiment scope broken) → hand to iteration: fix the scope mismatch:
  - Option A: re-scope C5 claim to "flagging bare-harmful content" (not jailbreaks) — AUROC=1.000 vs LG 0.9992 does support this narrower claim
  - Option B: obtain successful jailbreak instances (requires model/attack where ASR>0) and re-run M5 on the jailbreak-inclusive test set
  - Then re-invoke `/auto-verify C5 — resume: true`

→ **C2, C4 INTEGRITY_ONLY** (max_verify_claims_cap) → no back-edge action needed; record in Open Items:
  - `/auto-verify C2 — resume: true` (model-swap to Qwen2 may reveal position crossover if Qwen2 has measurable refusal-side)
  - `/auto-verify C4 — resume: true` (model-swap unlikely to help unless attack family changes; core issue is ASR=0 on instruction-tuned models with published transferable suffixes)
