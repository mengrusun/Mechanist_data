# Verify Report
# Generated: 2026-07-15

## Parameters
- TARGET_CLAIMS: all
- DIMENSIONS: model
- MAX_VERIFY_CLAIMS: 1
- ROBUSTNESS_THRESHOLD: 0.50
- MIN_VARIANTS_FOR_VERDICT: 1
- GPU_ID: 0,1,2,3
- AUTO_PROCEED: true
- AUTO_DEPLOY: true
- CODE_REVIEW: true
- SANITY_FIRST: true
- COMPACT: false
- RESUME: false

---

## Per-Claim Verdicts

| Claim | Short Text | Baseline Verdict | Robustness | Eligible/Run | Variants (pass/fail) | Integrity | Final State |
|-------|-----------|-----------------|------------|-------------|---------------------|-----------|-------------|
| C2a | aligned_spearman_gain | supported | 1.00 | 1/1 | 1/0 | clean | PASS |
| C1a | teacher_beats_baselines | supported | — | —/— | —/— | PASS (Phase 2) | INTEGRITY_ONLY (max_verify_claims_cap) |
| C1b | teacher_hierarchy_monotonic | not-supported | — | —/— | —/— | WARN (Phase 2) | INTEGRITY_ONLY (max_verify_claims_cap) |
| C2b | perlevel_spearman_gain | supported | — | —/— | —/— | WARN (Phase 2) | INTEGRITY_ONLY (max_verify_claims_cap) |
| C2c | gain_alignment_specific | supported | — | —/— | —/— | PASS (Phase 2) | INTEGRITY_ONLY (max_verify_claims_cap) |
| C3 | behavioural_uncertainty_match | supported | — | —/— | —/— | PASS (Phase 2) | INTEGRITY_ONLY (max_verify_claims_cap) |
| C4a | downstream_noninferior | not-supported | — | —/— | —/— | WARN (Phase 2) | INTEGRITY_ONLY (max_verify_claims_cap) |
| C4b | ood_strict_improve | not-supported | — | —/— | —/— | WARN (Phase 2) | INTEGRITY_ONLY (max_verify_claims_cap) |

---

## C2a — Detailed Verdict: PASS

**Claim**: Aligned DINOv2 ViT-B improves aggregate Spearman correlation with human THINGS-triplet similarity judgments compared to unaligned DINOv2 ViT-B (delta_rho >= 0.05).

**Main experiment result**: supported (spearman_aggregate: aligned=0.5554, unaligned=0.1891; delta_rho=+0.366, 7.3x threshold)

**Variant**: model-swap-dinov2-vits — student changed from DINOv2 ViT-B (86M, 768-dim) to DINOv2 ViT-S (21M, 384-dim); teacher SigLIP-So400m fixed (hard constraint); all hyperparameters identical to M3.

**Variant result**:
- Aligned ViT-S: spearman_aggregate = 0.5397
- Unaligned ViT-S: spearman_aggregate = 0.2157
- delta_rho = +0.3240 (6.5x threshold)
- consistent_with_main_experiment: pass (same direction, same order of magnitude)

**Robustness**: 1 pass / 1 eligible = 1.00 >= 0.50 threshold

**Phase 2 integrity**: EXPERIMENT=pass, MECHANISM=n/a; combined=PASS
**Phase 9 integrity**: EXPERIMENT=pass, MECHANISM=n/a; combined=PASS; variant eligible

**Verdict: PASS**

---

## Stage-2 Selection

**Stage-2 pick (Phase 3 step 0):** picked 1/8 admitted claims — C2a. Stage-2-deferred (INTEGRITY_ONLY, stage2_skip_reason: max_verify_claims_cap): C1a, C1b, C2b, C2c, C3, C4a, C4b.

**Selection rationale for C2a:** Largest effect size (delta_rho=+0.366, 7.3x threshold), cleanest Phase 2 audit (PASS with no caveats), most central to the task-vector composition pipeline (Verify step), and aligns with orchestrator priority hint in task.md. C2b and C2c are downstream consequences of C2a.

**To swap-test deferred claims later:**
- `/auto-verify C1a — resume: true` (Stage 1 audit reused via RESUME)
- `/auto-verify C1b — resume: true`
- `/auto-verify C2b — resume: true`
- `/auto-verify C2c — resume: true`
- `/auto-verify C3 — resume: true`
- `/auto-verify C4a — resume: true`
- `/auto-verify C4b — resume: true`

---

## Counts

- 1 PASS (C2a)
- 0 FAIL
- 0 INCONCLUSIVE
- 0 ZERO_ELIGIBLE_VARIANTS
- 7 INTEGRITY_ONLY (all: stage2_skip_reason=max_verify_claims_cap)
- **Total: 8 target claims**

---

## Phase 2 Baseline Integrity Summary

- PASS: C1a, C2a, C2c, C3 (4 claims)
- WARN: C1b (numeric citation mismatch in fine-level prose), C2b (fine-level 0 pairs), C4a (dataset substitution), C4b (dataset substitution + OOD definition issue) (4 claims)
- FAIL: 0 claims — no INCONCLUSIVE claims
- All 8 claims admitted to the Stage-1 pool

See: verify/INTEGRITY_AUDIT.md (Phase 2 baseline section)

## Phase 9 Variant Integrity Summary

- PASS: model-swap-dinov2-vits / C2a (1 variant)
- FAIL: 0 variants
- N_eligible: 1 / N_run: 1

See: verify/INTEGRITY_AUDIT.md (Phase 9 variant section)

---

## GPU Pin Verification

cost.json gpu_ids: [0, 1, 2, 3] — within task.md hard constraint {0, 1, 2, 3}. Pin propagation: PASS.

## Compute Used

- runs/verify/C2a_model_swap_dinov2_vits/cost.json: gpu_hours~0.50, wall_clock~8.4h (includes queue), gpu_ids=[0,1,2,3]
- Total verify GPU-hours: ~0.50 (alignment: ~0.48h, eval: ~0.02h)

---

## Artifacts

- verify/VERIFY_REPORT.md (this file)
- verify/INTEGRITY_AUDIT.md (Phase 2 baseline + Phase 9 variant sections)
- verify/STAGE2_PICK.json (Phase 3 step 0 pick record)
- verify/C2a_aligned_spearman_gain/baseline_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}
- verify/C2a_aligned_spearman_gain/variant_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}
- verify/C2a_aligned_spearman_gain/ROBUSTNESS.md
- verify/C2a_aligned_spearman_gain/variants/model-swap-dinov2-vits/ (variant code + results)
- runs/verify/C2a_model_swap_dinov2_vits/ (variant run output: aligned+unaligned eval JSONs, cost.json)
- verify/C[1-4]*/ROBUSTNESS.md (INTEGRITY_ONLY stubs for 7 deferred claims)
