# Integrity Audit — All Claims

**Experiment**: Verifying Locatability, Causality, and Applied Control of Emotion Circuits in Llama-3.2-3B
**Date**: 2026-07-13
**Phase 2 (Baseline Integrity) + Phase 9 (Variant Integrity)**

---

## Baseline Integrity (Phase 2, per-claim)

### C1 — Localizability (verdict=supported)

**Overall**: PASS with WARN

| Check | Verdict | Detail |
|-------|---------|--------|
| Data splits | PASS | 10/5/5 scenario-level, asserted disjoint; used_n=available_n |
| Jaccard methodology | PASS | 3-fold 80% event-subsample, 200-draw permutation null, size-matched pool |
| Sparsity floor | PASS | (k_h*, k_n*) = (24, 2000), grid minimum |
| Stage-B signal | PASS | 2.136 nats macro gain, well above 0.01-nat warning |
| Jaccard 6/6 both axes | PASS | heads 0.947-1.000 vs CI-hi 0.265-0.277; neurons 0.972-0.983 vs CI-hi 0.045-0.047 |
| Random control | PASS | Within null CI as expected |
| **Flat kstar grid** | **WARN** | All 9 grid cells yield identical macro_gain (kstar selection non-discriminative) |
| **Head Stage-B layer-level** | **WARN** | Head enhancement hook adds full layer direction; per-head causal ranking degenerate |

**Admission**: ADMITTED to Stage-1 pool; INTEGRITY_ONLY (Stage 2 skipped, stage2_skip_reason: max_verify_claims_cap)

### C2 — Causal + Stability (verdict=not-supported)

**Overall**: WARN (admitted with warnings; not blocked)

| Check | Verdict | Detail |
|-------|---------|--------|
| Data splits | PASS | Eval fold 120 stems x 6 emotions, scenario-disjoint |
| Scenario stability | PASS | 6/6 S1/S2 Jaccard pass |
| Enhancement sweep | PASS | alpha in {0.5, 1.0, 2.0} fully implemented |
| **Ablation operator scope** | **WARN (HIGH)** | Global mean (all stems x other 5 emotions) instead of per-stem mean. Wrong-scope substitution may explain wrong-sign delta |
| **Same-emotion bug** | **CLEAR** | NOT present: code uses others=[e_off for e_off in EMOTIONS if e_off != e] |
| **Random-null degenerate** | **WARN (MED)** | null_delta_std=0 for all 6 emotions (100 draws identical). Random-null control invalid |
| Neuron std from eval vs train | WARN (LOW) | Minor deviation: std computed on eval fold not train fold |

**Admission**: ADMITTED to Stage-1 pool; INTEGRITY_ONLY (Stage 2 skipped, stage2_skip_reason: max_verify_claims_cap)
**Critical implication**: The "not-supported" verdict for C2 may be an operator implementation artifact. The
ablation operator does NOT substitute the same emotion (not the "substitute with SAME emotion mean" bug), but
it DOES use a global mean across all stems instead of per-stem, which can ADD rather than neutralize signal.
The wrong-sign ablation delta is a plausible consequence of this operator scope mismatch.

### C3 — Applied Control (verdict=not-supported)

**Overall**: PASS with minor WARN

| Check | Verdict | Detail |
|-------|---------|--------|
| Data splits | PASS | 120 eval stems x 6 emotions x 3 arms, judge gate PASS |
| Judge reliability | PASS | agreement=1.000 (180/180 gold items); gpt-5.4 hidden-target |
| OTHER counted as wrong | PASS | per_row_correct=0 for OTHER judgments |
| Matched budget | PASS | N=9 val configs per arm per emotion |
| Pre-eval-split freeze | PASS | Arm C direction on train fold; kstar frozen before eval |
| **Length secondary not run** | **WARN (LOW)** | Ratio 1.26 > 1.15 threshold; gap ~13 words, minor |
| **Val-eval metric gap** | **WARN (LOW)** | Val uses logprob gain; eval uses generation accuracy |

**Admission**: ADMITTED to Stage-1 pool; **PICKED for Stage 2** (max_verify_claims cap selects C3 as top-importance)

---

## Variant Integrity (Phase 9)

### C3 — Qwen2.5-7B-Instruct model-swap variant

**Source**: runs/A4_verify_qwen/ (M4 milestone, same experiment as main run)

| Check | Verdict | Detail |
|-------|---------|--------|
| Hyperparameter independence | PASS | Qwen configs re-tuned on Qwen val fold |
| C_e independence | PASS | Qwen C_e fit with Qwen Stage A+B |
| Judge reuse | PASS | Same gpt-5.4 judge, same gold gate |
| Within-family constraint | PASS | Model swap only; method unchanged |
| GPU pin | PASS | Ran on CUDA_VISIBLE_DEVICES=1,2,3,5,6 |
| Variant result integrity | PASS | 2160 judged continuations; metrics complete |

**Phase 9 verdict**: PASS — variant is integrity-clean, eligible for robustness computation.

### C1, C2 Variant Integrity

**Skipped** — Stage 2 not run for C1 and C2 (INTEGRITY_ONLY due to max_verify_claims_cap).
Variant integrity section for C1/C2: [skipped — Stage 2 not run for these claims]

---

## Summary Table

| Claim | Phase 2 Verdict | Phase 9 Verdict | Notes |
|-------|-----------------|-----------------|-------|
| C1 | PASS (WARN x2) | skipped (INTEGRITY_ONLY) | flat kstar grid; head Stage-B layer-level |
| C2 | WARN (not block) | skipped (INTEGRITY_ONLY) | ablation operator scope mismatch (HIGH WARN); random-null degenerate |
| C3 | PASS (WARN x2) | PASS | minor warns only; Qwen variant clean |
