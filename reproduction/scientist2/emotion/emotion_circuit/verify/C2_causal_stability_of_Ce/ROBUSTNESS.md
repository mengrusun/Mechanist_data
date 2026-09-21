# Robustness Report — C2 (Causal + Stability of C_e)

**Terminal state**: INTEGRITY_ONLY (stage2_skip_reason: max_verify_claims_cap)
**Main-experiment verdict**: not-supported
**Baseline integrity (Phase 2)**: WARN (operator scope mismatch; random-null degenerate)

---

## Claim Statement

C_e causally carries emotion generation and is specific + scenario-stable on Llama-3.2-3B-Instruct:
ablation weakens target at alpha2, enhancement strengthens it with monotonic dose-response (Spearman >= 0.7
over 3 alpha); specificity holds against three complementary controls; scenario Jaccard(S1,S2) > null.

## Main Experiment Evidence

- Ablation delta: +0.510 to +0.928 nats target (WRONG SIGN on all 6 emotions; expected < 0).
- Enhancement at alpha=1.0: +1.34 to +5.83 nats (correct sign but non-monotonic on 4/6 emotions:
  Spearman(alpha, delta) = -1.0 for anger and surprise, -0.5 for disgust, 0.5 for fear).
- Specificity (b) 4/6, (c) 5/6, (d) 6/6 (scenario stability passes).
- Rubric: 0 full / 0 partial / 0 causal-only / 6 not-supported.

## Stage 2 Status

**INTEGRITY_ONLY — stage2_skip_reason: max_verify_claims_cap**

C2 was admitted to the Stage-1 pool (baseline integrity WARN — admitted, not blocked) but was not
selected for Stage 2 because max_verify_claims=1 and C3 was selected as the top-importance claim.
C2 should be re-verified with a CORRECTED ablation operator (per-stem mean, not global mean).

## Robustness Score

**Not computed** (Stage 2 not run for this claim).

## Critical Integrity Warnings

### Warning 1 (HIGH): Ablation operator scope mismatch

**Spec**: per-stem mean over OTHER 5 emotion variants of the SAME event stem.
**Implemented**: global mean over ALL 120 eval stems x OTHER 5 emotions (constant offset).

The same-emotion substitution bug (flagged in task.md) does NOT apply — the code correctly uses
`others = [e_off for e_off in EMOTIONS if e_off != e]`. However, the SCOPE bug (global vs. per-stem)
is real and is the most plausible explanation for the wrong-sign ablation delta. When the substitute
value is the global mean (which includes activation contributions from all emotions and all stems),
and the emotion-specific target-emotion activation at a given stem is below this global mean, the
substitution RAISES the activation, producing positive delta instead of negative.

**Implication for C2 verdict**: The "not-supported" verdict may be an operator artifact rather
than evidence of non-causality. A correct per-stem operator might produce negative ablation deltas
(the expected causal-necessity signature). The iteration loop should prioritize fixing this operator.

### Warning 2 (MEDIUM): Degenerate random-null distribution

All 100 random draws from the Stage-A shortlist pool produce identical logprob gain values
(std = 0 or machine epsilon). Root cause: the per-layer aggregation of random head/neuron subsets
produces the same aggregate layer delta regardless of which specific components are chosen.
This makes the random-null specificity control invalid (it cannot distinguish C_e from random sets).

## Upgrade Command

To run Stage 2 swap variants for C2 (requires corrected operator):
```
/auto-verify C2 --resume=true --dimensions=model --gpu_id=1,2,3,5,6
```
Note: iteration should first fix the ablation operator (per-stem mean) before re-running M2,
then re-verify C2 with the corrected experiment output.
