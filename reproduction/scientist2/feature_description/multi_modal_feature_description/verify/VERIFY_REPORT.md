# Verify Report

**Date**: 2026-07-14  
**Pipeline**: /auto-verify  
**Target claims**: all (C1, C2)  
**Dimensions**: method  
**MAX_VERIFY_CLAIMS**: 1  
**ROBUSTNESS_THRESHOLD**: 0.5  
**MIN_VARIANTS_FOR_VERDICT**: 1  
**GPU_ID**: 1,2,3,5,6  

---

## Per-Claim Verdicts

### C1 — "SemanticLens generates faithful, concept-discriminating feature summaries"

- **Baseline verdict** (main experiment): supported
- **Stage 2**: SKIPPED — C1 not selected by Phase 3 step 0 importance pick (MAX_VERIFY_CLAIMS=1 cap; C2 was selected as higher-importance claim)
- **Stage 2 skip reason**: max_verify_claims_cap
- **Robustness**: — (no variants run)
- **Verdict**: INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap)

Main-experiment integrity (Phase 2): WARN  
- Experiment audit overall_verdict: WARN (scope overstatement — "every component c in a trained vision model" but only ResNet-50 tested; P3 cross-model SKIPPED; P1b docstring mismatch)
- Mechanism audit overall_verdict: N/A (SemanticLens is observational)
- Combined: WARN → admitted to Stage 1; Stage 2 skipped by cap

Upgrade command: `/auto-verify C1 — resume: true` (Phase 2 audit reused; Stage 2 runs fresh)

---

### C2 — "Pooled frozen-CLIP vector v_c places c in the joint image-text semantic space"

- **Baseline verdict** (main experiment): supported (MRR=0.898, sig=True, stability_cos=0.970, layer4 d=1.96)
- **Variant run**: method-swap-crp-compose (CRP-crop compose; method dimension)
- **Variant result**: pass_fail=pass (MRR=0.9024, sig=True, stability_cos=0.9581, delta_mrr=+0.0044)
- **Variant integrity (Phase 9)**: WARN (exp=WARN scope-narrowing expected; mech=N/A)
- **N_eligible**: 1 (WARN is not FAIL; variant is eligible)
- **N_pass**: 1
- **Robustness**: 1.00 (1/1)
- **Threshold**: 0.50
- **Verdict**: PASS (robustness 1.00 ≥ 0.50; baseline supported and stable under method swap)

Main-experiment integrity (Phase 2): WARN  
- Experiment audit overall_verdict: WARN (scope — fc P2c fails under heuristic grouping; P3 SKIPPED)
- Mechanism audit overall_verdict: N/A (observational)
- Combined: WARN → admitted to Stage 2; picked by Phase 3 step 0

---

## Stage-2 Selection

| Claim | Admitted | Picked | Stage-2 status | stage2_skip_reason |
|-------|----------|--------|----------------|--------------------|
| C1    | YES (WARN) | NO   | INTEGRITY_ONLY | max_verify_claims_cap |
| C2    | YES (WARN) | YES  | Variants run   | — |

**Picked**: C2 (1/2 admitted claims, under MAX_VERIFY_CLAIMS=1 cap)  
**Rationale**: C2 is the more scientifically central claim — it operationalizes the core SemanticLens capability (text-queryable joint semantic space placement, P2a MRR=0.898). The method-swap axis is directly applicable to C2's core metric. C1's main-experiment evidence (top-1 purity, layer separation) is more indirect and the within-family method swap is less cleanly defined for C1.  
**Stage-2 deferred**: C1 (INTEGRITY_ONLY, stage2_skip_reason: max_verify_claims_cap)

See: `verify/STAGE2_PICK.json`

---

## Cross-Claim Summary

| Claim | Baseline verdict | Robustness | Eligible | Variants | Integrity | Final verdict |
|-------|-----------------|------------|----------|----------|-----------|---------------|
| C1    | supported       | —          | 0        | 0/0      | Phase2=WARN | INTEGRITY_ONLY (max_verify_claims_cap) |
| C2    | supported       | 1.00       | 1/1      | 1 pass / 0 fail | Phase9=WARN | **PASS** |

**Counts**: 1 PASS, 0 FAIL, 0 INCONCLUSIVE, 0 ZERO_ELIGIBLE_VARIANTS, 1 INTEGRITY_ONLY (of 2 target claims; INTEGRITY_ONLY breakdown: 0 swap_variants_false + 1 max_verify_claims_cap)

---

## GPU Pin Verification

- Variant run: CUDA_VISIBLE_DEVICES="1,2,3,5,6"
- Active compute GPU: physical GPU 1 (UUID GPU-6d73793d-0247-8c38-a691-4f59dd2281cf)
- GPU 1 ∈ allowed set {1,2,3,5,6}: PASS
- No forbidden GPUs (0, 4, 7) observed in compute app table
- Pin propagation: PASS (no violation to report)

---

## Budget Usage

- Main experiment (pre-verify): ~0.4 GPU-h
- CRP-compose LRP loop (Phase 7, original run): 3256s ≈ 0.90 GPU-h on GPU 1
- CRP-compose recovery (P2a + P2b, Phase 7 continued): 2690s ≈ 0.75 GPU-h on GPU 1
- Total GPU-h used: ~2.05 GPU-h (budget: 10 GPU-h; remaining: ~7.95 GPU-h)

---

## Artifacts

- `verify/VERIFY_REPORT.md` — this file
- `verify/INTEGRITY_AUDIT.md` — Phase 2 baseline + Phase 9 variant sections
- `verify/STAGE2_PICK.json` — Phase 3 step 0 pick record
- `verify/C1_concept_faithful_summary/main_experiment_audit/` — C1 Phase 2 audits
- `verify/C1_concept_faithful_summary/ROBUSTNESS.md` — C1 INTEGRITY_ONLY record
- `verify/C2_clip_joint_semantic_space/main_experiment_audit/` — C2 Phase 2 audits
- `verify/C2_clip_joint_semantic_space/variant_audit/` — C2 Phase 9 audits
- `verify/C2_clip_joint_semantic_space/variants/method-swap-crp-compose/` — CRP-compose variant artifacts
- `verify/C2_clip_joint_semantic_space/ROBUSTNESS.md` — C2 robustness detail
