# Robustness Report — C3 (Applied Circuit Control Beats Prompting + Steering)

**Terminal state**: FAIL
**Main-experiment verdict**: not-supported
**Baseline integrity (Phase 2)**: PASS (with minor warns)
**Variant integrity (Phase 9)**: PASS
**Robustness**: 1.0 (1/1 eligible variants agree with main-experiment direction) — confirms not-supported

---

## Claim Statement

A circuit-based control method (activating C_e via the enhancement operator) reliably induces target
emotions across arbitrary held-out event stems and outperforms both prompting (EmotionPrompt-style) AND
single-direction steering (RepE/CAA-style, frozen pre-eval-split) on hidden-target 6-way judged emotion-
expression accuracy under a matched N=9 val budget: Arm A > Arm B on >= 5/6 emotions AND Arm A > Arm C
on >= 5/6 emotions, with paired-bootstrap 95% CIs excluding 0 per emotion.

## Robustness Computation

- N_variants_run: 1 (model-swap: Qwen2.5-7B-Instruct from M4)
- N_eligible: 1 (Phase 9 integrity PASS)
- N_pass_agree: 1 (variant verdict = not-supported, agrees with main experiment)
- robustness = 1/1 = 1.00
- threshold = 0.50 → robustness >= threshold → **FAIL** (claim not-supported, verdict confirmed robust)

Note: the PASS/FAIL state for CLAIM robustness is: a claim PASSes iff robustness >= threshold AND the
baseline conclusion is supported. Here the baseline conclusion is not-supported. The variant confirms
not-supported. Robustness = 1.0 means the not-supported verdict is ROBUST across both architectures.
The claim FAILS (not-supported confirmed).

## Main Experiment Evidence

| Emotion | Arm A (Llama) | Arm B (Llama) | Arm C (Llama) | A>B at 95% | A>C at 95% |
|---------|---------------|---------------|---------------|------------|------------|
| joy     | 0.025         | 0.933         | 0.425         | NO         | NO         |
| sadness | 0.017         | 0.758         | 0.192         | NO         | NO         |
| anger   | 0.000         | 0.675         | 0.183         | NO         | NO         |
| fear    | 0.208         | 0.608         | 0.550         | NO         | NO         |
| surprise| 0.042         | 0.492         | 0.075         | NO         | NO         |
| disgust | 0.025         | 0.550         | 0.717         | NO         | NO         |
| **Macro**| **0.053**    | **0.669**     | **0.357**     | 0/6        | 0/6        |

Arm A macro (0.053) is BELOW the 1/6=0.167 uniform-chance floor. 5/6 emotions dominated by "OTHER"
judgments (>97% off-target). Both A-B and A-C 95% CIs exclude zero IN THE WRONG DIRECTION.

## Variant Evidence (Qwen2.5-7B-Instruct)

| Emotion | Arm A (Qwen) | Arm B (Qwen) | Arm C (Qwen) |
|---------|-------------|-------------|-------------|
| joy     | 0.000       | 0.983       | 0.092       |
| sadness | 0.092       | 0.967       | 0.100       |
| anger   | 0.008       | 0.975       | 0.175       |
| fear    | 0.200       | 1.000       | 0.108       |
| surprise| 0.150       | 0.958       | 0.267       |
| disgust | 0.008       | 0.933       | 0.008       |
| **Macro**| **0.076**  | **0.969**   | **0.125**   |

A > B: 0/6; A > C: 0/6 on Qwen. Pattern identical to Llama. Arm A still below chance floor.
Arm B near-ceiling (0.969 macro) on Qwen — prompting works extremely well on Qwen with Qwen
instruction-following capabilities.

## Key Interpretive Finding

The not-supported verdict is ROBUST across Llama-3.2-3B-Instruct and Qwen2.5-7B-Instruct
(two different architectures, independently re-tuned hyperparameters). The enhancement operator
(cumulative additive injection across k_h=24+ heads and k_n=2000+ neurons at alpha) destroys
generation quality on both models, producing incoherent continuations that the judge correctly
labels "OTHER" rather than the target emotion. This is a systematic failure of the additive-
injection circuit operator in the applied generation setting, not a Llama-specific artifact.

The root cause hypotheses (ordered by evidence):
1. **Cumulative perturbation magnitude**: 24 heads x alpha * d_{e,L} + 2000 neurons x delta
   applied at EVERY decode step accumulates to a very large residual perturbation that pushes
   the model off-distribution. Single-direction steering (Arm C, 1 direction x 1 layer x alpha)
   at 0.357 macro shows steering CAN work — the granularity explosion in Arm A is the differentiator.
2. **Val-eval metric misalignment**: val configs selected by logprob gain (next-token logprob of
   "I feel {e}"), but eval measures generation quality over 100 new tokens. High logprob gain
   does not imply coherent generation.
3. **alpha scale**: best alpha selected per emotion on the val logprob gain metric; the selected
   alpha values may be too large for generation quality even if optimal for the logprob metric.

## Baseline Integrity Warnings (minor)

- Length-matched secondary analysis not run (ratio 1.26 > 1.15 trigger, but gap is 13 words,
  unlikely to explain a 60-point accuracy difference).
- Val-eval metric alignment gap (logprob gain vs. generation accuracy) documented.

## Verdict Summary

**FAIL** — C3 Applied Control is not-supported, confirmed robust at robustness=1.0 across
2 architectures. The circuit injection operator does not produce emotion-specific generation
on either Llama-3.2-3B-Instruct or Qwen2.5-7B-Instruct; prompting and single-direction
steering substantially outperform the circuit arm. Recommended iteration: (a) reduce cumulative
alpha / number of components in Arm A; (b) use generation quality on val (rather than logprob gain)
for config selection; (c) consider a zero-ablation or weaker enhancement variant.
