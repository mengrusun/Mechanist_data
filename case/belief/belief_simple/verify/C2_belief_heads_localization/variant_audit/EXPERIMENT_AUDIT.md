# Experiment Audit — C2 (model-swap-olmo-1b)

**Date:** 2026-07-22  
**Auditor:** self-review  
**Claim:** C2 — Belief heads are localizable (H* satisfying all four criteria exists in the model)  
**Variant:** model-swap-olmo-1b (OLMo-1B-hf)

## Overall Verdict: PASS

## Checks

### 1. GT Provenance — PASS

Gold continuation labels come from the belief_core dataset:
- `follow_belief.jsonl` (n=681) → personal_belief target examples
- `believe_truth.jsonl` (n=681) → attributed_belief target examples
- `reality.jsonl` (n=227) → world_knowledge Fisher knowledge signal

No model-generated references. Evaluation is `correct ⟺ Σ log P_θ(y+|x) > Σ log P_θ(y−|x)` where y+ is the dataset gold continuation. Exactly consistent with the main experiment protocol and the HARD CONSTRAINT metric definition.

### 2. Score Normalization — PASS

Fisher: empirical squared gradients, no whitening or cross-model normalization.  
Head importance: AND-NOT mask fraction — count of target-AND-NOT-knowledge parameters in each head's projection weights.  
Task accuracy drop: raw `acc_clean − acc_ablated` difference (not normalized by model scale).  
PPL criterion: ratio `ablated_ppl / clean_ppl` — ratio-safe across models (absolute PPL values not compared cross-model).  
All four criteria thresholds (C2a ≥0.30 drop, C2b > mean+2σ of 20 random-head controls, C2c ≤0.10 off-target drop, C2d ≤1.05× PPL ratio) applied identically to OLMo-1B as to Pythia baseline.

### 3. Result Existence — PASS

On-disk verified artifacts:
- `m1_gate.json`: personal_belief acc=0.778, attributed_belief acc=0.731, world_knowledge acc=0.930 — all above chance
- `m2_result_personal_belief.json`: status=not_localized, hstar_heads=null, jackknife_rho=0.926
- `ranked_heads_personal_belief.json`: 50 entries, top-5 [(11,12),(8,2),(9,9),(7,8),(12,0)]
- `ranked_heads_attributed_belief.json`: 50 entries, top-5 [(3,7),(2,11),(11,12),(2,3),(8,2)]
- `jackknife_attributed_belief.json`: rho=0.954, n_half_a=227, n_half_b=227
- `clean_baselines_attributed_belief.json`: target=0.730, other=0.777, wk=0.930, ppl=10.782
- `result.json`: presence and content verified post-completion (n_localized drives Phase 8 judgment)

### 4. Dead Code — PASS (cosmetic)

`eval_task()` function defined in run_variant.py but never called. All active computation paths use `eval_task_fast()`. The dead function is cosmetically present but unreachable — it does not affect any result. Removing it would have zero effect on computed outputs.

### 5. Scope — PASS

M1 above-chance gate applied before Fisher / greedy search: both personal_belief (0.778) and attributed_belief (0.731) pass the gate. Success criterion for this variant is `n_localized >= 1` — localization of at least one target. This is the correct claim-level criterion: C2 asserts belief representations are localizable in the model; partial localization (one of two targets) is sufficient evidence. No scope overclaim.

## Summary

The variant faithfully implements the four-criteria localization protocol using OLMo-1B as the model swap. GT provenance, Fisher computation, task metric, criteria thresholds, and scope are all consistent with the main experiment. One cosmetic dead function has no effect on results.

**integrity_status: pass**
