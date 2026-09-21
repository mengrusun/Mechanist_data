# Verify Report

**Pipeline:** auto-verify Workflow 1.75 — Stage 1 → 2 → 3  
**Parameters:** TARGET_CLAIMS=all, DIMENSIONS=model, MAX_VERIFY_CLAIMS=1, ROBUSTNESS_THRESHOLD=0.5, MIN_VARIANTS_FOR_VERDICT=1, GPU_ID=0,1,2,3  
**Date:** 2026-07-15  
**Variant model:** DeepSeek-R1-Distill-Qwen-14B (model-swap-qwen-14b)

---

## Stage-2 Selection

**Phase 3 step 0 pick:** 1 / 3 admitted claims selected (MAX_VERIFY_CLAIMS=1)

| Claim | Stage 1 gate | Stage 2 status | Rationale |
|-------|-------------|----------------|-----------|
| C1 | WARN — admitted | INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap) | Deferred — C4 picked as top-1 |
| C2 | WARN — admitted | INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap) | Deferred — C4 picked as top-1 |
| C3 | FAIL — rejected | INCONCLUSIVE (mechanism FAIL) | Skipped Stages 2–3 |
| C4 | WARN — admitted | **PICKED** (top-1 by importance) | Most informative cross-model test: quantitative operating-point claim with clean model-swap hook |

Full pick rationale: `verify/STAGE2_PICK.json`

---

## Per-claim verdicts

### C1 — Linear behaviour directions exist and are extractable

- **Main-experiment verdict:** not-supported (AUC ≥ 0.75 passes; first-PC |cos| ≥ 0.70 fails for all four behaviours — joint predicate fails)
- **Phase 2 integrity:** WARN (exp=WARN — LLM-judge proxy annotation; mech=N/A — M1 probe-only, no additive steering)
- **Stage 2:** skipped (INTEGRITY_ONLY, stage2_skip_reason: max_verify_claims_cap)
- **robustness:** — (not computed)
- **State: INTEGRITY_ONLY** (stage2_skip_reason: max_verify_claims_cap)

Upgrade: `/auto-verify C1 — resume: true`

---

### C2 — Small pool is sufficient for direction extraction

- **Main-experiment verdict:** not-supported (split-half cos=0.588 < 0.70 floor for uncertainty at n=25; other 3 behaviours untestable at current corpus scale)
- **Phase 2 integrity:** WARN (exp=WARN — ratio_to_large_pool normalized by model-derived reference; mech=WARN — borrows M3's under-validated α_op)
- **Stage 2:** skipped (INTEGRITY_ONLY, stage2_skip_reason: max_verify_claims_cap)
- **robustness:** — (not computed)
- **State: INTEGRITY_ONLY** (stage2_skip_reason: max_verify_claims_cap)

Upgrade: `/auto-verify C2 — resume: true`

---

### C3 — Dose-response causal control by steering vector

- **Main-experiment verdict:** not-supported (partial for uncertainty, negative for other three behaviours)
- **Phase 2 integrity:** FAIL (exp=WARN, mech=FAIL → combined=FAIL)
  - Mechanism FAIL reasons: α sweep range ±2σ does not span ≥3 OOM; no random-direction control at n_random≥30; α_op=0.5σ placed at plateau edge; 3 of 4 behaviours show zero rate across all α
- **Stage 2:** skipped (INCONCLUSIVE — baseline mechanism rigor broken; swap-test on broken anchor uninformative)
- **robustness:** — (variants never ran)
- **State: INCONCLUSIVE** (inconclusive_reason: main-experiment mechanism rigor broken — see verify/C3_dose_response_causal_control/main_experiment_audit/MECHANISM_AUDIT.md)

---

### C4 — Steering offers more distinct operating points than prompt/TI

- **Main-experiment verdict:** not-supported (positive-partial — n_distinct(steering)=4 > n_distinct(prompt)=2=n_distinct(TI) for expressing_uncertainty; preservation misses 3-pt floor by 1 pt → joint claim fails)
- **Phase 2 integrity:** WARN (exp=WARN — LLM-judge proxy GT; mech=WARN — no random-direction control, no confirmed mid-plateau)
- **Stage 2:** RAN — model-swap-qwen-14b (DeepSeek-R1-Distill-Qwen-14B)
- **Variant result:**
  - n_distinct(steering)=0 (all 4 alpha values cause complete incoherence — sigma_proj=886 on Qwen-14B vs 10.52 on Llama-8B, 84x larger)
  - n_distinct(prompt)=2
  - n_distinct(TI)=2
  - Primary predicate: 0 > 2 = FALSE → INCONSISTENT
  - Binary judgment: **fail**
- **Phase 9 integrity:** WARN (exp=WARN + mech=WARN → combined=WARN → variant admitted to eligible pool)
- **N_run=1, N_eligible=1, N_pass=0, N_fail=1**
- **robustness = 0/1 = 0.00 < 0.50** → **FAIL**
- **State: FAIL**

---

## Summary

| Claim | Main-exp verdict | Phase 2 integrity | Variants run | Eligible | Pass | Robustness | State |
|-------|-----------------|-------------------|-------------|---------|------|-----------|-------|
| C1 | not-supported | WARN | 0 | — | — | — | INTEGRITY_ONLY (cap) |
| C2 | not-supported | WARN | 0 | — | — | — | INTEGRITY_ONLY (cap) |
| C3 | not-supported | FAIL | 0 | — | — | — | INCONCLUSIVE |
| C4 | not-supported | WARN | 1 | 1 | 0 | 0.00 | FAIL |

**Counts:** 0 PASS, 1 FAIL, 1 INCONCLUSIVE, 0 ZERO_ELIGIBLE_VARIANTS, 2 INTEGRITY_ONLY (of 4 target claims; INTEGRITY_ONLY breakdown: 0 swap_variants_false + 2 max_verify_claims_cap)

---

## Key scientific finding

**sigma_proj architectural incompatibility**: The CAA formula with sigma_proj scaling is not directly portable across the Llama and Qwen backbones within the R1-Distill family. sigma_proj on Qwen-14B for expressing_uncertainty = 886 (vs 10.52 for Llama-8B = 84x larger). At alpha=±1, the effective coefficient is ±886, well above any coherent operating range for Qwen-14B. All 4 steering controllers collapse to incoherent outputs or zero behaviour rate. By contrast, NL-prompt and thinking-intervention controllers remain fully coherent and produce 2 distinct operating points each on Qwen-14B.

This makes C4 **FAIL** — the claim that steering offers more distinct operating points than prompt/TI does not generalize cross-architecture under the same sigma_proj-scaled formula.

---

## Integrity summary

**Phase 2 (baseline integrity):** WARN overall (C1 WARN, C2 WARN, C3 FAIL, C4 WARN)  
**Phase 9 (variant integrity):** WARN overall (C4 / model-swap-qwen-14b: exp=WARN + mech=WARN → combined=WARN → admitted)

**GPU pin propagation:** VERIFIED — cost.json gpu_ids="0,1,2,3" for M1 run; CUDA_VISIBLE_DEVICES=0,1,2,3 in run.sh for all steps.

---

## Artifacts

- `verify/VERIFY_REPORT.md` — this file
- `verify/INTEGRITY_AUDIT.md` — Phase 2 baseline + Phase 9 variant sections (one file)
- `verify/STAGE2_PICK.json` — Phase 3 step 0 pick record
- `verify/C1_linear_behaviour_directions/main_experiment_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}`
- `verify/C2_small_pool_extractability/main_experiment_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}`
- `verify/C3_dose_response_causal_control/main_experiment_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}`
- `verify/C4_finer_than_prompt_engineering/main_experiment_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}`
- `verify/C4_finer_than_prompt_engineering/variant_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}`
- `verify/C1_linear_behaviour_directions/ROBUSTNESS.md`
- `verify/C2_small_pool_extractability/ROBUSTNESS.md`
- `verify/C3_dose_response_causal_control/ROBUSTNESS.md`
- `verify/C4_finer_than_prompt_engineering/ROBUSTNESS.md`
- `verify/C4_finer_than_prompt_engineering/variants/model-swap-qwen-14b/` (variant artifacts)
- `verify/C4_finer_than_prompt_engineering/variants/model-swap-qwen-14b/verdict.json`
- `verify/C4_finer_than_prompt_engineering/variants/model-swap-qwen-14b/m1_14b/` (M1 results)
- `verify/C4_finer_than_prompt_engineering/variants/model-swap-qwen-14b/m4_14b/` (M4 results)
