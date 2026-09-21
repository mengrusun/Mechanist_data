# Verify Report

**Date**: 2026-07-18
**Run config**: TARGET_CLAIMS=all DIMENSIONS=method MAX_VERIFY_CLAIMS=1 ROBUSTNESS_THRESHOLD=0.5 MIN_VARIANTS_FOR_VERDICT=1 GPU_ID=0,1,2,3 RESUME=true

---

## Summary Table

| Claim | Short description | Baseline verdict | Robustness | Threshold | Eligible | Variants pass/fail | Integrity | Final verdict |
|-------|-------------------|-----------------|------------|-----------|----------|--------------------|-----------|---------------|
| C1 | >=3pp QA_I drop (treated vs both controls, per seed) | supported | 1.00 | 0.50 | 1/1 | 1/0 | clean | PASS |
| C2 | BOUNDED_NULL mechanism (CAA steering) | not-supported | — | 0.50 | — | — | — | INCONCLUSIVE |

**Overall**: 1 PASS, 0 FAIL, 1 INCONCLUSIVE, 0 ZERO_ELIGIBLE_VARIANTS, 0 INTEGRITY_ONLY (of 2 target claims).

---

## Stage-2 Selection (Phase 3 Step 0)

**Admitted pool** (Phase 2 PASS or WARN): C1  
**Rejected pool** (Phase 2 FAIL → INCONCLUSIVE): C2  
**Cap**: MAX_VERIFY_CLAIMS=1  
**Picked**: C1 (trivial selection — only one admitted claim)  
**Stage-2-deferred** (INTEGRITY_ONLY, `stage2_skip_reason: max_verify_claims_cap`): none

Rationale for C1: C1 is the behavioral phenomenon anchor of the project — the primary quantitative claim that text-only teacher-distilled data impairs multimodal QA. It is the necessary precondition for C2 (mechanism claim). Robustness-testing C1 takes priority over a mechanism claim that failed the baseline integrity gate.

See `verify/STAGE2_PICK.json` for the full pick record.

---

## Claim Details

### C1 — covert_transfer_phenomenon — PASS

**Claim**: In the fixed Qwen3.5-9B -> Qwen3.5-9B multimodal transfer setup and the exact task.md recipe, text-only tuned-teacher-generated filtered data causes a >=3pp drop in QA_I accuracy vs BOTH Ctrl-A and Ctrl-B, per seed across all 3 pre-registered seeds {42, 123, 2026}.

**Phase 2 baseline integrity**: PASS (exp=PASS, mech=N/A)

**Variant run**: 1 variant dispatched (DIMENSIONS=method → 1 swap axis)

| Variant | Dimension | Swap description | claim_supported | consistent_w_main | Integrity | Eligible |
|---------|-----------|-----------------|-----------------|-------------------|-----------|----------|
| method-swap-rule-based-scorer | method | LLM judge (gpt-5.4) → deterministic regex letter extractor | pass | pass | pass | yes |

**Phase 9 variant integrity**: PASS (exp=PASS, mech=N/A)

**Robustness**: 1.0 (1/1 eligible variants pass) — threshold 0.50 — **PASS**

**Key metrics (variant)**:
- rule_based_acc: treated_mean=0.101, ctrlb_mean=0.261, ctrl_a=0.233
- Per-seed gaps (gap_A, gap_B): seed42=(+15.8pp, +16.5pp), seed123=(+15.0pp, +21.1pp), seed2026=(+9.0pp, +10.5pp)
- All 3 seeds pass both >=3pp inequalities; min gap = 9.0pp (3x threshold)

**Delta**: Absolute accuracy lower under regex (conservative scorer); relative inter-arm gaps preserved and larger. Result is not an artifact of LLM judge leniency.

**Artifacts**:
- `verify/C1_covert_transfer_phenomenon/main_experiment_audit/`
- `verify/C1_covert_transfer_phenomenon/variant_audit/`
- `verify/C1_covert_transfer_phenomenon/variants/method-swap-rule-based-scorer/`
- `verify/C1_covert_transfer_phenomenon/ROBUSTNESS.md`

---

### C2 — bounded_null_mechanism — INCONCLUSIVE

**Claim**: The BOUNDED_NULL mechanism verdict holds — CAA steering on the L-Core direction fails to monotonically recover QA_I performance as alpha scales from negative to positive (median Spearman rho >= -0.5 not achieved; recovery fraction < 30% in >= 2/3 seeds).

**Phase 2 baseline integrity**: FAIL

- Experiment audit: WARN (executor post-hoc verification resolved C/D to PASS; reviewer WARN preserved)
- Mechanism audit: FAIL — Check A triggered (CAA steering with alpha sweep):
  - (1) No independent capability metric logged at any alpha point — only acc_steered; cannot distinguish mechanism failure from OOD collapse
  - (2) Sign pattern broken: median Spearman rho = +0.371 (2/3 seeds positive); expected <= -0.5 for BOUNDED_NULL direction confirmation

**Gate decision**: INCONCLUSIVE — Stages 2-3 skipped for C2.

**Inconclusive reason**: main-experiment mechanism rigor broken — see `verify/C2_bounded_null_mechanism/main_experiment_audit/MECHANISM_AUDIT.md`

**Iteration instruction**: To convert C2 from INCONCLUSIVE to a verifiable claim:
1. Add an independent capability metric (e.g., perplexity on held-out text, random-arm eval accuracy) logged at every alpha sweep point in `scripts/steer_and_eval.py`
2. Increase n_random >= 30 (current: unclear from code) to establish a stable null distribution
3. Re-run M2 sweeps with the augmented evaluation harness
4. Re-invoke `/auto-verify C2 --resume false` after the fix

**Artifacts**:
- `verify/C2_bounded_null_mechanism/main_experiment_audit/`
- `verify/C2_bounded_null_mechanism/ROBUSTNESS.md`

---

## Integrity Summary

See `verify/INTEGRITY_AUDIT.md` for full audit tables.

**Phase 2 (baseline integrity)**:
- C1: PASS (exp=PASS, mech=N/A)
- C2: FAIL (exp=WARN, mech=FAIL → combined FAIL)

**Phase 9 (variant integrity)**:
- C1 / method-swap-rule-based-scorer: PASS (exp=PASS, mech=N/A)
- C2: skipped (INCONCLUSIVE from Phase 2)

---

## GPU Pin Verification

All variant runs for C1 (method-swap-rule-based-scorer) are CPU-only: the script reads saved JSONL outputs and runs deterministic regex scoring — no model inference, no CUDA. No `cost.json` GPU field to verify. GPU constraint (CUDA_VISIBLE_DEVICES=0,1,2,3) is set in `run.sh` as a precautionary export and was not violated.

---

## Artifact Index

| File | Status |
|------|--------|
| `verify/VERIFY_REPORT.md` | this file |
| `verify/INTEGRITY_AUDIT.md` | complete (Phase 2 + Phase 9 sections) |
| `verify/STAGE2_PICK.json` | complete |
| `verify/C1_covert_transfer_phenomenon/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}` | complete |
| `verify/C1_covert_transfer_phenomenon/main_experiment_audit/MECHANISM_AUDIT.{md,json}` | complete |
| `verify/C1_covert_transfer_phenomenon/variant_audit/EXPERIMENT_AUDIT.{md,json}` | complete |
| `verify/C1_covert_transfer_phenomenon/variant_audit/MECHANISM_AUDIT.{md,json}` | complete |
| `verify/C1_covert_transfer_phenomenon/ROBUSTNESS.md` | complete |
| `verify/C1_covert_transfer_phenomenon/variants/method-swap-rule-based-scorer/` | complete |
| `verify/C2_bounded_null_mechanism/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}` | complete |
| `verify/C2_bounded_null_mechanism/main_experiment_audit/MECHANISM_AUDIT.{md,json}` | complete |
| `verify/C2_bounded_null_mechanism/ROBUSTNESS.md` | complete (INCONCLUSIVE stub) |
