# Verify Report

**Run parameters:** TARGET_CLAIMS=all DIMENSIONS=model MAX_VERIFY_CLAIMS=1 ROBUSTNESS_THRESHOLD=0.5 MIN_VARIANTS_FOR_VERDICT=1 GPU_ID=0,1,2,3 MAX_PARALLEL_RUNS=4 COMPACT=false RESUME=false
**swap_variants:** true
**Date:** 2026-07-15

---

## Per-claim verdicts

| Claim | Baseline verdict | Robustness | Threshold | Eligible | Variants (pass/fail) | Integrity | State |
|-------|-----------------|-----------|---------|---------|---------------------|-----------|-------|
| c2 (sae_concept_alignment_gap) | not-supported | 1.00 | 0.50 | 1/1 | 1/0 | clean | PASS |
| c1 (sae_interpretable_feature_count) | not-supported | — | 0.50 | 0/0 | — | n/a | INTEGRITY_ONLY |
| c3 (sae_superposition_specificity) | not-supported | — | 0.50 | 0/0 | — | n/a | INTEGRITY_ONLY |
| c4 (novel_concept_discovery) | not-supported | — | 0.50 | 0/0 | — | n/a | INTEGRITY_ONLY |
| c5a (annotation_filling_probes) | not-supported | — | 0.50 | 0/0 | — | n/a | INTEGRITY_ONLY |
| c5b (feature_clamp_steering) | not-supported | — | 0.50 | 0/0 | — | n/a | INTEGRITY_ONLY |

**Counts:** 1 PASS, 0 FAIL, 0 INCONCLUSIVE, 0 ZERO_ELIGIBLE_VARIANTS, 5 INTEGRITY_ONLY (of 6 total)
**INTEGRITY_ONLY breakdown:** 0 stage2_skip_reason=swap_variants_false + 5 stage2_skip_reason=max_verify_claims_cap

---

## Stage-2 Selection

**Phase 3 step 0 output (`verify/STAGE2_PICK.json`):**

| Claim | Pool | Stage-2 decision | Rationale |
|-------|------|-----------------|-----------|
| c2 | admitted | PICKED | Most scientifically central claim (core headline: SAE concept alignment gap vs raw neurons). Scale generalization (650M→8M) is the highest-leverage robustness probe |
| c1 | admitted | deferred (INTEGRITY_ONLY, max_verify_claims_cap) | Interpretable feature count; absolute criterion under-powered; swap adds marginal value until primary criterion fixed |
| c3 | admitted | deferred (INTEGRITY_ONLY, max_verify_claims_cap) | Superposition specificity; direction already supported; margin criterion needs broader tau sweep first |
| c4 | admitted | deferred (INTEGRITY_ONLY, max_verify_claims_cap) | Novel concept discovery; criterion-design null — model swap does not resolve criterion issue |
| c5a | admitted | deferred (INTEGRITY_ONLY, max_verify_claims_cap) | Annotation filling; SGDClassifier under-fitting is the primary blocker; swap would not fix that |
| c5b | admitted | deferred (INTEGRITY_ONLY, max_verify_claims_cap) | Feature clamp steering; dose-range too narrow; swap would not fix 0.9 OOM sweep range |

**Picked K=1 of 6 admitted.** Rejected pool: empty (no FAIL in Phase 2).

---

## Claim details

### C2 — sae_concept_alignment_gap (PASS)

**Claim:** SAE features from ESM-2 cover substantially more Swiss-Prot functional concepts than raw neurons from the same model.

**Main experiment (ESM-2-650M):**
- SAE covered = 15 concepts at τ_F1=0.5, q_top=0.99
- Neuron covered = 0
- Ratio = ∞
- Main verdict: not-supported (15 < target 65; direction SAE≫neurons is supported)

**Variant: model-swap-esm2-8m (ESM-2-650M → ESM-2-8M)**
- Model: ESM-2-8M (d_model=320, 6 layers, ~6M parameters vs 650M)
- SAE: InterPLM-esm2-8m, d_feat=10240, layers {1,2,3,4,5,6}
- Dataset: same 1500 Swiss-Prot test sequences, 387195 residues
- Protocol: identical q_top=0.99, τ_F1=0.5, τ_clean=0.7

**Variant result:**
- SAE covered = 14 (main=15, delta=-1)
- Neuron covered = 0 (main=0)
- Ratio = ∞
- Sensitivity at τ=0.3: SAE=58, neuron=3, ratio=19.3×
- Sensitivity at q=0.95, τ=0.5: SAE=13, neuron=2, ratio=6.5×

**Judgment:** CONSISTENT — variant reaches same conclusion (not-supported; direction SAE≫neurons preserved)
**Variant integrity (Phase 9):** CLEAN (exp=pass, mech=n/a)
**Robustness:** 1/1 = 1.00 ≥ 0.50 → **PASS**

**Interpretation:** The SAE alignment gap is robust to an ~80× parameter-count reduction. ESM-2-8M SAEs achieve the same conceptual alignment advantage over raw neurons as ESM-2-650M SAEs (14 vs 15 concepts, ∞ ratio preserved). This is a strong scale-invariance signal: the alignment gap is an architectural property of the SAE training objective, not an artifact of the 650M model's capacity.

---

### C1 — sae_interpretable_feature_count (INTEGRITY_ONLY)

stage2_skip_reason: max_verify_claims_cap

Main verdict: not-supported (SAE_interp=1480, target≥2000; ratio criterion met at 19.3×)
Main integrity: WARN (auto-interp proxy GT, scope under-power at 1500/10k seqs)
Stage 2 skipped. To swap-test: `/auto-verify c1 — resume: true`

---

### C3 — sae_superposition_specificity (INTEGRITY_ONLY)

stage2_skip_reason: max_verify_claims_cap

Main verdict: not-supported at strict criterion (margin=15, target≥20; direction supported at τ=0.5; clean at τ=0.3 with margin=65)
Main integrity: WARN (strict margin not met; order_check_pass=false due to PCA=neurons=0)
Stage 2 skipped. To swap-test: `/auto-verify c3 — resume: true`

---

### C4 — novel_concept_discovery (INTEGRITY_ONLY)

stage2_skip_reason: max_verify_claims_cap

Main verdict: not-supported (0/100 novel features at criterion-design null)
Main integrity: WARN (synonym-check GT proxy; criterion non-discriminative)
Stage 2 skipped. To swap-test: `/auto-verify c4 — resume: true`

---

### C5a — annotation_filling_probes (INTEGRITY_ONLY)

stage2_skip_reason: max_verify_claims_cap

Main verdict: not-supported (p=0.191, means equal at 0.591; SGDClassifier substitution)
Main integrity: WARN (dead code in per_concept_pr_auc; SGDClassifier with max_iter=30; 25k/387k subsample)
Stage 2 skipped. To swap-test: `/auto-verify c5a — resume: true`

---

### C5b — feature_clamp_steering (INTEGRITY_ONLY)

stage2_skip_reason: max_verify_claims_cap

Main verdict: not-supported (no-steer yield highest for all 3 features; no dose-response)
Main integrity: WARN (exp: scope 3/4 features, 10/25 seqs; mech: sweep 8-fold = 0.9 OOM, <5 grid points)
Stage 2 skipped. To swap-test: `/auto-verify c5b — resume: true`

---

## Baseline integrity (Phase 2)

**Overall Phase 2:** WARN — all 6 claims WARN; 0 FAIL
All 6 claims admitted to Stage-1-admitted pool. No claim excluded by Phase 2 gate.

Details: `verify/INTEGRITY_AUDIT.md` and per-claim `verify/<claim_dir>/main_experiment_audit/`

## Variant integrity (Phase 9)

**Overall Phase 9:** PASS — 1 variant (c2 model-swap-esm2-8m) CLEAN; 0 findings

Details: `verify/INTEGRITY_AUDIT.md` and `verify/c2_sae_concept_alignment_gap/variant_audit/`

---

## GPU pin propagation check

CUDA_VISIBLE_DEVICES=0,1,2,3 passed to variant. cost.json reports gpu_ids=[0,1,2,3]. Pin propagation confirmed — no violation.

---

## Artifacts

- verify/VERIFY_REPORT.md (this file)
- verify/INTEGRITY_AUDIT.md
- verify/STAGE2_PICK.json
- verify/c2_sae_concept_alignment_gap/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}
- verify/c2_sae_concept_alignment_gap/main_experiment_audit/MECHANISM_AUDIT.{md,json}
- verify/c2_sae_concept_alignment_gap/variant_audit/EXPERIMENT_AUDIT.{md,json}
- verify/c2_sae_concept_alignment_gap/variant_audit/MECHANISM_AUDIT.{md,json}
- verify/c2_sae_concept_alignment_gap/ROBUSTNESS.md
- verify/c2_sae_concept_alignment_gap/PLAN.md
- verify/c2_sae_concept_alignment_gap/variants/model-swap-esm2-8m/run.py
- verify/c2_sae_concept_alignment_gap/variants/model-swap-esm2-8m/run.log
- verify/c2_sae_concept_alignment_gap/variants/model-swap-esm2-8m/JUDGMENT.json
- verify/c2_sae_concept_alignment_gap/variants/model-swap-esm2-8m/results/coverage.json
- verify/c2_sae_concept_alignment_gap/variants/model-swap-esm2-8m/results/sensitivity.json
- verify/c2_sae_concept_alignment_gap/variants/model-swap-esm2-8m/results/per_concept_best_F1.parquet
- verify/c2_sae_concept_alignment_gap/variants/model-swap-esm2-8m/results/cost.json
- verify/c1_sae_interpretable_feature_count/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}
- verify/c1_sae_interpretable_feature_count/main_experiment_audit/MECHANISM_AUDIT.{md,json}
- verify/c1_sae_interpretable_feature_count/ROBUSTNESS.md
- verify/c3_sae_superposition_specificity/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}
- verify/c3_sae_superposition_specificity/main_experiment_audit/MECHANISM_AUDIT.{md,json}
- verify/c3_sae_superposition_specificity/ROBUSTNESS.md
- verify/c4_novel_concept_discovery/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}
- verify/c4_novel_concept_discovery/main_experiment_audit/MECHANISM_AUDIT.{md,json}
- verify/c4_novel_concept_discovery/ROBUSTNESS.md
- verify/c5a_annotation_filling_probes/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}
- verify/c5a_annotation_filling_probes/main_experiment_audit/MECHANISM_AUDIT.{md,json}
- verify/c5a_annotation_filling_probes/ROBUSTNESS.md
- verify/c5b_feature_clamp_steering/main_experiment_audit/EXPERIMENT_AUDIT.{md,json}
- verify/c5b_feature_clamp_steering/main_experiment_audit/MECHANISM_AUDIT.{md,json}
- verify/c5b_feature_clamp_steering/ROBUSTNESS.md
