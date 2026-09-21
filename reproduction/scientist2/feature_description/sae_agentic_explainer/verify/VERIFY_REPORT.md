# Verify Report

**Pipeline**: auto-verify (Workflow 1.75)
**Run date**: 2026-07-14
**TARGET_CLAIMS**: all (C1, C2, C3, C4)
**DIMENSIONS**: model
**MAX_VERIFY_CLAIMS**: 1 (C4 picked; C1/C2/C3 deferred — INTEGRITY_ONLY)
**ROBUSTNESS_THRESHOLD**: 0.5
**MIN_VARIANTS_FOR_VERDICT**: 1
**GPU_ID**: 1 (from allowlist {1,2,3,5,6})
**Total variants run**: 1 (model-swap-gpt-oss-20b for C4)

---

## Stage-2 Selection

**Phase 3 step 0 pick**: 1/4 admitted claims → C4 selected (max_verify_claims=1)

| Claim | Stage 1 gate | Stage 2 decision | stage2_skip_reason |
|-------|-------------|-----------------|-------------------|
| C1 | ADMITTED (WARN) | deferred | max_verify_claims_cap |
| C2 | ADMITTED (PASS) | deferred | max_verify_claims_cap |
| C3 | ADMITTED (WARN) | deferred | max_verify_claims_cap |
| C4 | ADMITTED (WARN) | **PICKED** | — |

**Rationale for picking C4**: C4 has the weakest main-experiment evidence (inconclusive-positive-trend; only 2 of 3 planned depths tested; only one cross-pair tested), making it the highest-priority claim to stress-test with a model swap. The deferred GPT-OSS-20B pair had never been evaluated before.

See `verify/STAGE2_PICK.json` for full rationale.

---

## Per-claim Verdicts

| Claim | Short description | Baseline verdict | Robustness | Threshold | N_eligible / N_run | n_pass / n_fail | Integrity | Terminal state |
|-------|------------------|-----------------|------------|-----------|-------------------|-----------------|-----------|----------------|
| C1 | gen_acc main pair | supported | — | 0.5 | —/0 | —/— | — | INTEGRITY_ONLY (max_verify_claims_cap) |
| C2 | pred_acc main pair | not-supported | — | 0.5 | —/0 | —/— | — | INTEGRITY_ONLY (max_verify_claims_cap) |
| C3 | layer generalization | not-supported | — | 0.5 | —/0 | —/— | — | INTEGRITY_ONLY (max_verify_claims_cap) |
| C4 | cross-pair generalization | not-supported | **1.00** | 0.5 | 1/1 | 1/0 | clean | **PASS** |

---

## Detailed Claim Reports

### C1 — Generative Accuracy (Main Pair: Gemma-2-2B + JumpReLU SAE)

**Baseline verdict**: supported (partial-support — Wilcoxon p=0.0455, n_eff=4)
**Stage 1 integrity**: WARN (Exp: coarse gen_acc metric, n_eff=4; Mech: N/A)
**Stage 2**: deferred (INTEGRITY_ONLY, stage2_skip_reason=max_verify_claims_cap)
**Terminal state**: INTEGRITY_ONLY

Upgrade later: `/auto-verify C1 — resume: true, swap-variants: true` (Phase 2 audit reused).
Evidence: n=44 features, delta_gen_acc=+0.018, CI=[0.005,0.036], Wilcoxon p=0.0455, but n_eff=4.

---

### C2 — Predictive Accuracy (Main Pair: Gemma-2-2B + JumpReLU SAE)

**Baseline verdict**: not-supported (delta_pearson=-0.018, CI spans zero)
**Stage 1 integrity**: PASS (Exp: no issues; Mech: N/A)
**Stage 2**: deferred (INTEGRITY_ONLY, stage2_skip_reason=max_verify_claims_cap)
**Terminal state**: INTEGRITY_ONLY

Upgrade later: `/auto-verify C2 — resume: true, swap-variants: true`.
Evidence: n=44, delta_pearson=-0.018 (SAGE-lite worse than Neuronpedia on main pair).

---

### C3 — Layer Depth Generalization (Gemma-2-2B, L4/L12/L20)

**Baseline verdict**: not-supported (inconsistent across layers; L20 gen_acc all-zero)
**Stage 1 integrity**: WARN (Exp: Bonferroni gap, per-layer n=14-15; Mech: N/A)
**Stage 2**: deferred (INTEGRITY_ONLY, stage2_skip_reason=max_verify_claims_cap)
**Terminal state**: INTEGRITY_ONLY

Upgrade later: `/auto-verify C3 — resume: true, swap-variants: true`.
Evidence: No layer shows consistent advantage; L20 gen_acc differences all zero.

---

### C4 — Cross-Pair Generalization (M2: Qwen3-4B; Variant: GPT-OSS-20B)

**Baseline verdict**: not-supported (M2 inconclusive-positive-trend: delta_pearson=+0.153 CI=[-0.039,+0.391] on Qwen3-4B)
**Stage 1 integrity**: WARN (Exp: L28 missing from M2 data; scope narrowed to SAGE-lite+predictive)
**Stage 2**: PICKED — model-swap-gpt-oss-20b (GPT-OSS-20B + resid-post-aa, layers 3/11/19, n=45 features)

**Variant results (GPT-OSS-20B)**:
| Method | Mean Pearson r |
|--------|---------------|
| sage_lite | 0.1365 |
| neuronpedia | 0.1817 |
| gpt5_1shot | 0.1112 |

| Comparison | delta | 95% CI | significant |
|-----------|-------|--------|-------------|
| sage_lite vs neuronpedia | −0.0451 | [−0.210, +0.115] | No |
| sage_lite vs gpt5_1shot | +0.0254 | [−0.070, +0.119] | No |

**Phase 8 judgment**: SAGE-lite does NOT outperform Neuronpedia on GPT-OSS-20B (delta negative). Verdict: not-supported.
**Consistent with main experiment**: YES (both say "not-supported") → variant_verdict = **pass**

**Phase 9 integrity (variant)**: PASS — 7/7 checks pass (no fake GT, correct normalization, no phantom results, live metric code, correct scope, no leakage, valid statistics). N_eligible=1.

**Phase 10 robustness**: n_pass=1, N_eligible=1, robustness=1.00 ≥ 0.50 → **PASS**

**Terminal state**: PASS
**Robustness**: 1.00 (1/1 eligible variants agree with main experiment verdict)

**Per-layer breakdown**:
| Layer | sage_lite | neuronpedia | gpt5_1shot |
|-------|----------|-------------|-----------|
| L3 | 0.2307 | 0.3338 | 0.1220 |
| L11 | 0.0140 | 0.0112 | 0.0945 |
| L19 | 0.1649 | 0.2000 | 0.1169 |

---

## Cross-Claim Summary

| State | Claims |
|-------|--------|
| PASS | C4 |
| FAIL | (none) |
| INCONCLUSIVE | (none) |
| ZERO_ELIGIBLE_VARIANTS | (none) |
| INTEGRITY_ONLY (max_verify_claims_cap) | C1, C2, C3 |

**Counts**: 1 PASS, 0 FAIL, 0 INCONCLUSIVE, 0 ZERO_ELIGIBLE_VARIANTS, 3 INTEGRITY_ONLY (all cap).

---

## GPU Pin Verification

| Run | gpu_ids in cost.json | Within allowlist {1,2,3,5,6} |
|-----|---------------------|------------------------------|
| model-swap-gpt-oss-20b | [1] | YES |

No GPU pin propagation failures detected.

---

## Resource Usage

| Run | Wall clock | GPU hours |
|-----|------------|-----------|
| model-swap-gpt-oss-20b | 1701.8s (28.4 min) | 0.02 (no LLM forward pass) |

Main experiment total: 1.66 GPU-hours. Verify variant: 0.02 GPU-hours. Running total: 1.68 GPU-hours (budget: 10 GPU-hours; remaining: 8.32 GPU-hours for iteration).

---

## Artifacts

- `verify/VERIFY_REPORT.md` — this file
- `verify/INTEGRITY_AUDIT.md` — Phase 2 baseline + Phase 9 variant audit (one file)
- `verify/STAGE2_PICK.json` — Phase 3 step 0 pick record
- `verify/C4_cross_pair_generalization/main_experiment_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}` — baseline audit
- `verify/C4_cross_pair_generalization/variant_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}` — Phase 9 audit
- `verify/C4_cross_pair_generalization/ROBUSTNESS.md` — C4 robustness report (PASS, robustness=1.00)
- `verify/C4_cross_pair_generalization/variants/model-swap-gpt-oss-20b/` — variant run artifacts
  - `features.jsonl` (45 features)
  - `summary.json` (aggregate stats)
  - `cost.json` (gpu_ids=[1], gpu_hours=0.02)
  - `verdict.json` (variant_verdict=pass)
- `verify/C1_gen_acc_main_pair/ROBUSTNESS.md` — INTEGRITY_ONLY stub
- `verify/C2_pred_acc_main_pair/ROBUSTNESS.md` — INTEGRITY_ONLY stub
- `verify/C3_layer_depth_generalization/ROBUSTNESS.md` — INTEGRITY_ONLY stub

## Notes for Iteration

1. **C4 PASS**: The "not-supported" conclusion for C4 is robust. Both M2 (Qwen3-4B) and this variant (GPT-OSS-20B) independently fail to show a statistically significant SAGE-lite advantage in predictive accuracy over Neuronpedia's reference explanations. The C4 claim cannot be upheld as stated.

2. **C1 INTEGRITY_ONLY**: The "supported" conclusion (gen_acc, Wilcoxon p=0.0455, n_eff=4) needs a model-swap variant to verify robustness. To upgrade: `/auto-verify C1 — resume: true, swap-variants: true`. Caution: n_eff=4 is fragile.

3. **C2 INTEGRITY_ONLY**: The "not-supported" conclusion (pred_acc negative) needs a model-swap to verify. To upgrade: `/auto-verify C2 — resume: true, swap-variants: true`.

4. **C3 INTEGRITY_ONLY**: The "not-supported" conclusion (layer generalization) needs a model-swap to verify. To upgrade: `/auto-verify C3 — resume: true, swap-variants: true`.

5. **Upgrade order recommendation**: C2 (cleanest methodology), then C1 (significant but fragile), then C3 (secondary question given C1/C2 results).
