# Experiment Audit — C4 Variant (model-swap-qwen-14b)

**Claim:** C4 — steering-vector control offers more distinct (rate, accuracy) operating points than prompt/TI  
**Variant:** model-swap-qwen-14b (DeepSeek-R1-Distill-Qwen-14B)  
**Audit type:** Phase 9 variant-level experiment integrity audit  
**Date:** 2026-07-15  
**Auditor:** auto-verify Phase 9 (audit-by-construction, symmetric to Phase 2)

**Overall verdict: WARN**

---

## Check A — Ground-truth provenance

**Status: WARN**

Same LLM-judge proxy pipeline as the main experiment: GPT-5.4 generates gold answers, GPT-5.4 judges accuracy, and an LLM behaviour classifier annotates expressing_uncertainty. This is inherited by design — no human-annotation or dataset ground truth is used. The same provenance warning from Phase 2 applies here without amplification (the variant uses identical evaluation scaffolding).

## Check B — Score normalization

**Status: PASS**

n_distinct_operating_points is a count (integer). Accuracy is a proportion (mean over 60 tasks). Coherence_rate is a proportion. No normalization artefacts — consistent with main experiment.

## Check C — Result existence

**Status: PASS**

All 8 controllers completed successfully (8/8 entries in results_summary.json confirmed). Results:

| Controller | coherence_rate | behaviour_rate | accuracy |
|-----------|---------------|----------------|---------|
| steering_alpha_neg2 | 0.350 | 0.000 | 0.000 |
| steering_alpha_neg1 | 0.400 | 0.000 | 0.000 |
| steering_alpha_pos1 | 0.017 | 0.000 | 0.000 |
| steering_alpha_pos2 | 0.000 | nan | None |
| prompt_suppress | 1.000 | 0.133 | 0.800 |
| prompt_amplify | 1.000 | 0.350 | 0.700 |
| thinking_intervention_suppress | 1.000 | 0.667 | 0.767 |
| thinking_intervention_amplify | 1.000 | 0.633 | 0.717 |

Results are internally consistent: steering coherence_rate collapse (0.0–0.40) is mechanistically explained by sigma_proj=886 (M1 result), causing coef = alpha * 886 which produces near-total incoherence. Prompt and TI controllers unaffected (no steering applied).

## Check D — Dead code

**Status: WARN**

Matched-rate pairing algorithm not exercised at per-task level for this variant audit (per-task JSON exists but was not independently re-executed). Inherits same WARN as Phase 2 — algorithm exists in source code (src/run_M4_control_compare.py) but per-task verification not conducted. The n_distinct_operating_points computation was independently verified from summary-level data.

## Check E — Scope

**Status: PASS**

Variant scope: expressing_uncertainty only, n=60 tasks from the same 500-task benchmark. Documented in PLAN.md and config.yaml. No scope overreach. sigma_proj=886 finding is a legitimate scientific observation about cross-model generalization, not a scope violation.

---

## Variant-specific findings

### sigma_proj architectural difference (INFO)

sigma_proj for expressing_uncertainty on Qwen-14B = 886 (vs Llama-8B = 10.52, 84x larger). The sigma_proj-scaled CAA formula (coef = alpha * sigma_proj) produces steering coefficients ~84x larger on Qwen-14B than on Llama-8B at the same alpha values. This causes near-total output incoherence for all 4 steering alpha values. This is a scientifically significant finding about cross-backbone generalization of sigma_proj-scaled CAA — not an evaluation failure.

### Model swap fidelity (PASS)

M1 re-run faithfully on Qwen-14B: 200 chains × 48 layers, L*=45, sigma_proj=886 computed. M4 re-run with 8 controllers × 60 tasks, using Qwen-14B model and its extracted direction. Same auxiliary corpus and benchmark tasks. GPU pin: CUDA_VISIBLE_DEVICES=0,1,2,3 confirmed (cost.json gpu_ids="0,1,2,3").

---

**Audit conclusion:** WARN (inherited gt_provenance WARN + dead_code WARN; no FAIL findings). Variant admitted to robustness computation. Phase 9 integrity_status = WARN.
