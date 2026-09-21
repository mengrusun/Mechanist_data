# Verify Report

**Project**: Steerable Social-Variable Directions in an LLM Dictator (llm_social_decision)
**Verify round**: 1 (Phases 1–7 completed prior; Phases 8–10 completed this session)
**Date**: 2026-07-13
**DIMENSIONS**: model
**MAX_VERIFY_CLAIMS**: 1
**ROBUSTNESS_THRESHOLD**: 0.5
**MIN_VARIANTS_FOR_VERDICT**: 1
**COMPACT**: false
**GPU_ID**: 1,2,3,5,6
**RESUME**: true (Phases 8–10 only)

---

## Per-Claim Verdicts

| Claim | Short text | Baseline verdict | Robustness | Threshold | N_eligible/N_run | n_pass/n_fail | Integrity | Terminal state |
|-------|-----------|-----------------|------------|-----------|-----------------|----------------|-----------|----------------|
| C1 | linear encoding of social variables | supported | — | 0.5 | 0/0 | 0/0 | WARN (Phase 2) | INTEGRITY_ONLY (skip=max_verify_claims_cap) |
| C2 | purity via decorrelation (GS) | supported | — | 0.5 | 0/0 | 0/0 | WARN (Phase 2) | INTEGRITY_ONLY (skip=max_verify_claims_cap) |
| C3 | bidirectional causal steering at L=16 | supported | 1.00 | 0.5 | 1/1 | 1/0 | WARN (Phase 2 + 9) | PASS |
| C4 | selectivity matrix (4×4) | not-supported | — | 0.5 | 0/0 | 0/0 | WARN (Phase 2) | INTEGRITY_ONLY (skip=max_verify_claims_cap) |

---

## Detailed Per-Claim Summaries

### C1 — Linear Encoding of Social Variables

- **Baseline verdict**: supported (probe cv_acc=1.0 for all V; proj-transfer β significant for 3/4 V)
- **Phase 2 integrity**: WARN — proxy GT not labeled; 48 unique prompt texts appear in both train/held splits; shallow layer pick (L=2–6)
- **Stage 2 status**: SKIPPED — INTEGRITY_ONLY (`stage2_skip_reason: max_verify_claims_cap`)
- **Robustness**: null (no variants run)
- **Terminal state**: INTEGRITY_ONLY
- **Detail**: `verify/C1_linear_encoding_social_vars/ROBUSTNESS.md`

### C2 — Purity via Decorrelation

- **Baseline verdict**: supported by GS; not supported by LEACE
- **Phase 2 integrity**: WARN — GS off-diagonal 0.506 (borderline vs 0.55 threshold); 1-D scalar-projection cross-leakage probe is conservative; LEACE norm inflation not formally addressed
- **Stage 2 status**: SKIPPED — INTEGRITY_ONLY (`stage2_skip_reason: max_verify_claims_cap`)
- **Robustness**: null (no variants run)
- **Terminal state**: INTEGRITY_ONLY
- **Detail**: `verify/C2_purity_via_decorrelation/ROBUSTNESS.md`

### C3 — Bidirectional Causal Steering

- **Baseline verdict**: supported at L=16 supplementary run; inconclusive/under-powered at ell_V*
- **Phase 2 integrity**: WARN — supp run uses raw v_hat_V (not pure GS/LEACE); 5-point alpha grid; no L=16 random-direction control; plateau lock post-hoc
- **Stage 2**: PICKED (1/4 admitted claims; cap=1). Variant: `model-swap-meta-llama3-8b` (Meta-Llama-3-8B-Instruct)
- **Phase 8 result**: `consistent_with_main_experiment = true` — V=M sign-inverts at alpha=+2 at L=16 in swap model (matching main experiment); V=G, V=I, V=A all show bidirectional modulation at L=16
- **Phase 9 integrity**: WARN — n_baseline=10 (compact sweep; planned n=200); coherence format_ok_rate=1.0 all cells; hook site verified; no L=16 random-direction control in variant
- **N_eligible**: 1 (0 excluded; WARN does not exclude from N_eligible)
- **Robustness**: 1/1 = 1.00 >= 0.50 threshold
- **Terminal state**: **PASS**
- **Detail**: `verify/C3_bidirectional_causal_steering/ROBUSTNESS.md`

### C4 — Selectivity Matrix (4×4)

- **Baseline verdict**: not supported at ell_V* (permutation test p=0.84; min-diagonal=0 degenerate ratio for V=G)
- **Phase 2 integrity**: WARN — synthetic_proxy GT; min-diagonal=0 artifact; inherits C3's missing L=16 random-direction control; C4 at L=16 deferred to iteration
- **Stage 2 status**: SKIPPED — INTEGRITY_ONLY (`stage2_skip_reason: max_verify_claims_cap`)
- **Robustness**: null (no variants run)
- **Terminal state**: INTEGRITY_ONLY
- **Detail**: `verify/C4_selectivity_matrix/ROBUSTNESS.md`

---

## Stage-2 Selection (Phase 3 Step 0)

**Admitted pool** (Phase 2 PASS/WARN): C1, C2, C3, C4 (4 claims)
**Cap**: MAX_VERIFY_CLAIMS=1
**Picked** (1): C3 — selected as the mechanistic crux claim; positive evidence based on supplementary L=16 run (deviation from pre-registration) making model-swap the highest-value robustness test
**Stage-2 deferred** (INTEGRITY_ONLY, `stage2_skip_reason: max_verify_claims_cap`): C1, C2, C4
**Rejected pool**: none

To run Stage 2 on deferred claims:
- `/auto-verify C1 --resume true` (Phase 2 baseline audit reused)
- `/auto-verify C2 --resume true`
- `/auto-verify C4 --resume true`

See `verify/STAGE2_PICK.json` for full rationale.

---

## Variants Run

**Total variants run**: 1 (one per dimension; DIMENSIONS=model; only under picked claim C3)

| Variant | Claim | Dimension | Swap | N_cells | N_pass | N_fail | Phase 9 | Eligible |
|---------|-------|-----------|------|---------|--------|--------|---------|---------|
| model-swap-meta-llama3-8b | C3 | model | Meta-Llama-3-8B-Instruct | 40 | 1 (consistent) | 0 | WARN | YES |

---

## Baseline Integrity (Phase 2)

**Overall**: WARN (max severity across all 4 target claims)
**All 4 claims admitted** (WARN gate: continue-with-warn). No claims marked INCONCLUSIVE.

| Claim | Exp. audit | Mech. audit | Combined | Gate |
|-------|------------|-------------|----------|------|
| C1 | WARN | N/A | WARN | continue |
| C2 | WARN | N/A | WARN | continue |
| C3 | WARN | WARN | WARN | continue |
| C4 | WARN | WARN | WARN | continue |

**Details**: `verify/INTEGRITY_AUDIT.md` (Phase 2 section) + per-claim `verify/<claim_dir>/main_experiment_audit/`

---

## Variant Integrity (Phase 9)

**Overall**: WARN — 1 variant audited; 0 FAIL; 3 WARN findings; 1 eligible
**Findings**:
1. n_baseline=10 per cell (compact sweep; planned n=200) — WARN
2. No random-direction control at L=16 in swap variant — WARN
3. Alpha range ±2σ only (not ±4σ) — WARN

Integrity-FAIL variants excluded from N_eligible and robustness computation: **0** (none excluded).

**Details**: `verify/INTEGRITY_AUDIT.md` (Phase 9 section) + `verify/C3_bidirectional_causal_steering/variant_audit/`

---

## Summary Counts

| State | Count | Claims |
|-------|-------|--------|
| PASS | 1 | C3 |
| FAIL | 0 | — |
| INCONCLUSIVE | 0 | — |
| ZERO_ELIGIBLE_VARIANTS | 0 | — |
| INTEGRITY_ONLY | 3 | C1, C2, C4 |
| **Total** | **4** | C1, C2, C3, C4 |

INTEGRITY_ONLY breakdown: 0 `stage2_skip_reason=swap_variants_false` + 3 `stage2_skip_reason=max_verify_claims_cap`

---

## Artifacts

| File | Status |
|------|--------|
| `verify/VERIFY_REPORT.md` | COMPLETE |
| `verify/INTEGRITY_AUDIT.md` | COMPLETE (Phase 2 + Phase 9) |
| `verify/STAGE2_PICK.json` | COMPLETE (from Phase 3 step 0) |
| `verify/C1_linear_encoding_social_vars/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}` | COMPLETE |
| `verify/C1_linear_encoding_social_vars/main_experiment_audit/MECHANISM_AUDIT.{md,json}` | COMPLETE (N/A verdict) |
| `verify/C1_linear_encoding_social_vars/ROBUSTNESS.md` | COMPLETE (INTEGRITY_ONLY) |
| `verify/C2_purity_via_decorrelation/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}` | COMPLETE |
| `verify/C2_purity_via_decorrelation/main_experiment_audit/MECHANISM_AUDIT.{md,json}` | COMPLETE (N/A verdict) |
| `verify/C2_purity_via_decorrelation/ROBUSTNESS.md` | COMPLETE (INTEGRITY_ONLY) |
| `verify/C3_bidirectional_causal_steering/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}` | COMPLETE |
| `verify/C3_bidirectional_causal_steering/main_experiment_audit/MECHANISM_AUDIT.{md,json}` | COMPLETE |
| `verify/C3_bidirectional_causal_steering/variant_audit/EXPERIMENT_AUDIT.{md,json}` | COMPLETE (Phase 9) |
| `verify/C3_bidirectional_causal_steering/variant_audit/MECHANISM_AUDIT.{md,json}` | COMPLETE (Phase 9) |
| `verify/C3_bidirectional_causal_steering/ROBUSTNESS.md` | COMPLETE (PASS) |
| `verify/C3_bidirectional_causal_steering/variants/model-swap-meta-llama3-8b/RESULT_TO_CLAIM.json` | COMPLETE (Phase 8) |
| `verify/C3_bidirectional_causal_steering/variants/` | COMPLETE (artifacts on disk) |
| `verify/C4_selectivity_matrix/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}` | COMPLETE |
| `verify/C4_selectivity_matrix/main_experiment_audit/MECHANISM_AUDIT.{md,json}` | COMPLETE |
| `verify/C4_selectivity_matrix/ROBUSTNESS.md` | COMPLETE (INTEGRITY_ONLY) |

---

## Notes

1. **C3 PASS — n=10 caveat**: The swap variant ran at n=10 per cell (compact sanity sweep), not the planned n=200. The PASS verdict is based on qualitative pattern-match (V=M sign-inverts at alpha=+2 in both models; all 4 V show bidirectionality at L=16). A future full-scale re-run (n=200) should confirm quantitative effect size stability.

2. **C3 no L=16 random-direction control**: Neither the main experiment supp run nor the swap variant includes a random-direction control at L=16. The V=M sign-inversion result could in principle arise from a large-magnitude perturbation in any direction. This is the single most important methodological gap for the C3 claim and should be addressed in the iteration loop.

3. **C1, C2, C4 deferred**: These three claims are INTEGRITY_ONLY due to the MAX_VERIFY_CLAIMS=1 cap. Their baseline Phase 2 verdicts (C1: supported WARN, C2: supported-GS/not-supported-LEACE WARN, C4: not-supported WARN) stand unverified by swap testing. Run `/auto-verify <C1|C2|C4> --resume true` to add model-swap coverage.

4. **C4 at L=16 not evaluated**: The main experiment did not run the full 4×4 selectivity matrix at L=16 (only per-V dose-response from M4-supp). C4's iteration-loop priority is to run M5 selectivity at L=16.

5. **GPU pin propagation**: No new GPU dispatches were made in Phases 8–10 (resume-only). GPU pin (1,2,3,5,6) confirmed in config.yaml for the pre-existing swap variant run.

---

## Auto-proceed log line

`AUTO_PROCEED: verified 4 claim(s) across model — 1 PASS / 0 FAIL / 0 INCONCLUSIVE / 0 ZEV / 3 INTEGRITY_ONLY; main-experiment-integrity: WARN, variant-integrity: WARN`
