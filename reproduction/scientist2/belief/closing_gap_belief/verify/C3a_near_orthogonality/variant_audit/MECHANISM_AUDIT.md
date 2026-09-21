# Mechanism Audit Report — Claim C3a (Variant: model-swap-qwen25-7b-instruct)

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (gpt-5.4, via llm-chat API, cross-model)
**Project**: Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence
**Claim**: C3a — The gold-correctness direction v_c and the verbalized-confidence direction v_v at the primary reporting layer L* are geometrically near-orthogonal (|cos| <= 0.3 with 95% CI upper bound < 0.4), robust across L*+-2 neighborhood.
**Variant**: model-swap-qwen25-7b-instruct

## Overall Verdict: N/A

*C3a uses no mechanism intervention (no activation steering, no causal intervention). Mechanism audit checks are not triggered. This is consistent with the main-experiment mechanism audit for C3a (also N/A).*

## Triggered Checks: none

C3a is a purely geometric claim about the cosine angle between two probe weight vectors. The variant implements linear probing only — no steering coefficient, no alpha sweep, no cross-direction causal manipulation. All mechanism-audit checks (steering coefficient sweep and reserved checks B-F) do not apply to this variant.

## Combined Verdict: n/a → treated as pass in max_severity computation

Per the auto-verify skill: `n/a` (no mechanism intervention used) is treated as `pass` in the combined verdict computation `max_severity(exp, mech)`. The variant's integrity_status is therefore determined entirely by the experiment audit: PASS.
