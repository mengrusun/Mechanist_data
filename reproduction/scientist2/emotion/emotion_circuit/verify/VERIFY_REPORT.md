# Verify Report

**Experiment**: Verifying Locatability, Causality, and Applied Control of Emotion Circuits in Llama-3.2-3B
**Date**: 2026-07-13
**Configuration**: target_claims=all, max_verify_claims=1, dimensions=model,
robustness_threshold=0.5, min_variants_for_verdict=1, auto_proceed=true, compact=false

---

## Stage-2 Selection

**Admitted pool** (Phase 2 PASS/WARN): C1, C2, C3 (all 3 claims pass baseline integrity gate)
**Picked for Stage 2**: C3 (Applied Control — top importance per FINAL_PROPOSAL.md pay-off claim)
**Stage-2 deferred** (INTEGRITY_ONLY, stage2_skip_reason: max_verify_claims_cap): C1, C2
**Rejected pool** (Phase 2 FAIL): none

Rationale for C3 pick: C3 is the practical impact claim the entire Location->Causal->Applied
framework was built to earn. Additionally, M4 (the main experiment's Qwen verify-swap-lite) provides
a ready-made model-swap variant artifact at runs/A4_verify_qwen/, eliminating the need for a new run
within the GPU budget.

---

## Per-Claim Verdicts

### C1 — Localizability per-emotion component sets C_e

**Terminal state**: INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap)
**Main-experiment verdict**: supported
**Baseline integrity (Phase 2)**: PASS with WARN
**Stage 2**: skipped (max_verify_claims cap; deferred)
**Robustness**: — (not computed)

Key stats:
- (k_h*, k_n*) = (24, 2000); sparsity floor met
- Jaccard heads 0.947-1.000 vs. null CI-hi 0.265-0.277 (all 6/6 pass)
- Jaccard neurons 0.972-0.983 vs. null CI-hi 0.045-0.047 (all 6/6 pass)

Integrity warnings:
- WARN: Flat kstar grid (all 9 grid cells identical macro_gain; selection non-discriminative)
- WARN: Head Stage-B hook is layer-level not head-level (head causal ranking degenerate within layer)

Upgrade: `/auto-verify C1 --resume=true --dimensions=model --gpu_id=1,2,3,5,6`

---

### C2 — Causal + Stability of C_e

**Terminal state**: INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap)
**Main-experiment verdict**: not-supported
**Baseline integrity (Phase 2)**: WARN (not blocked)
**Stage 2**: skipped (max_verify_claims cap; deferred)
**Robustness**: — (not computed)

Key stats:
- Ablation delta: +0.510 to +0.928 nats (WRONG SIGN all 6 emotions)
- Enhancement Spearman: -1.0 (anger), -1.0 (surprise), -0.5 (disgust), +0.5 (fear), +1.0 (joy), +1.0 (sadness)
- Rubric: 0 full / 0 partial / 0 causal-only / 6 not-supported

CRITICAL INTEGRITY FINDING:
The "not-supported" verdict may be an operator artifact. The ablation operator does NOT contain the
same-emotion substitution bug (code correctly uses other 5 emotions), but it DOES use a GLOBAL mean
(over all 120 eval stems x 5 off-target emotions) instead of the plan-specified PER-STEM mean.
The global-mean substitution can RAISE activations if the emotion-specific value is below the global
mean, producing positive (wrong-sign) ablation delta. The random-null is also degenerate (std=0 for
all 6 emotions across 100 draws). Both are HIGH-priority integrity concerns.

Upgrade: `/auto-verify C2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6`
(Iteration should first fix the ablation operator to per-stem mean, then re-run M2.)

---

### C3 — Applied Circuit Control Beats Prompting + Steering

**Terminal state**: FAIL
**Main-experiment verdict**: not-supported
**Baseline integrity (Phase 2)**: PASS (minor warns)
**Variant**: Qwen2.5-7B-Instruct model-swap (from M4, runs/A4_verify_qwen/)
**Variant integrity (Phase 9)**: PASS
**N_eligible**: 1, **N_pass_agree**: 1 (variant also not-supported)
**Robustness**: 1.00 (threshold 0.50) — not-supported verdict is ROBUST

Key stats (Llama main):
- Macro A=0.053 (below 1/6=0.167 chance floor), B=0.669, C=0.357
- A>B: 0/6, A>C: 0/6 (95% CIs all exclude zero in wrong direction)
- "OTHER" dominates Arm A (97%+ on 5/6 emotions)

Key stats (Qwen model-swap variant):
- Macro A=0.076 (below chance), B=0.969, C=0.125
- A>B: 0/6, A>C: 0/6 — same failure pattern on different architecture

Root causes (both architectures):
1. Cumulative additive injection (24 heads + 2000 neurons at alpha) pushes model OOD during generation
2. Val metric (logprob gain) misaligned with eval metric (generation accuracy)
3. Enhancement operator designed for single-token scoring, not 100-token generation

Iteration priorities:
- Reduce cumulative alpha / number of components; test with k_h=5, k_n=200 at alpha=0.5
- Use generation accuracy on val (not logprob gain) for config selection
- Consider gated or partial-prefix injection rather than every-decode-step injection

---

## Cross-Claim Summary

| Claim | Main Verdict | Phase 2 | Stage 2 | Robustness | Terminal State |
|-------|-------------|---------|---------|------------|----------------|
| C1 (Localizability) | supported | PASS+WARN | skipped | — | INTEGRITY_ONLY (cap) |
| C2 (Causal + Stable) | not-supported | WARN | skipped | — | INTEGRITY_ONLY (cap) |
| C3 (Applied) | not-supported | PASS+minor WARN | RAN | 1.00 | FAIL |

**Counts**: 0 PASS, 1 FAIL, 0 INCONCLUSIVE, 0 ZERO_ELIGIBLE_VARIANTS, 2 INTEGRITY_ONLY
(of 3 target claims; INTEGRITY_ONLY breakdown: 0 stage2_skip_reason=swap_variants_false +
2 stage2_skip_reason=max_verify_claims_cap)

---

## Critical Findings for Iteration

1. **C3 FAIL (robust)**: The circuit-injection Arm A destroys generation quality on BOTH Llama-3.2-3B
   and Qwen2.5-7B (robustness=1.0). This is a systematic operator failure, not architecture-specific.
   Iteration must address the additive-injection operator's cumulative magnitude.

2. **C2 operator caveat (high priority)**: The ablation delta wrong-sign finding MAY be due to the
   global-vs-per-stem mean substitution bug. If the ablation operator is corrected to per-stem, C2's
   verdict could change from not-supported to causal-only or partial. This is the iteration loop's
   most important diagnostic: fix the ablation operator, re-run M2, re-verify C2.

3. **C1 kstar anomaly**: All 9 kstar grid cells are identical — the kstar selection was non-discriminative.
   The sparsity floor is technically met but the claim "smallest cell was best" is misleading.

---

## Artifacts

- verify/VERIFY_REPORT.md (this file)
- verify/INTEGRITY_AUDIT.md
- verify/STAGE2_PICK.json
- verify/C1_localizability_per_emotion_component_sets_Ce/baseline_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}
- verify/C1_localizability_per_emotion_component_sets_Ce/ROBUSTNESS.md
- verify/C2_causal_stability_of_Ce/baseline_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}
- verify/C2_causal_stability_of_Ce/ROBUSTNESS.md
- verify/C3_applied_circuit_control_beats_prompting_steering/baseline_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}
- verify/C3_applied_circuit_control_beats_prompting_steering/variant_audit/{EXPERIMENT,MECHANISM}_AUDIT.{md,json}
- verify/C3_applied_circuit_control_beats_prompting_steering/ROBUSTNESS.md
- verify/C3_applied_circuit_control_beats_prompting_steering/variants/model_Qwen2.5-7B-Instruct/README.md
- Source variant data: runs/A4_verify_qwen/ (M4 Qwen model-swap milestone)
