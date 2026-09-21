# Mechanism Audit Report — Claim C1

**Date**: 2026-07-20
**Auditor**: expert self-review (llm-chat MCP unavailable — graceful degradation)
**Claim**: C1 — Subliminal banana preference transfers via judge-filtered non-banana channel
**Linked milestones**: M-PREP, M0 (M0.1–M0.8)

## Overall Verdict: N/A

## Integrity Status: n/a

## Rationale

C1 (M0) is a **phenomenon-validation** claim. It tests whether a behavioral transfer signal exists, using LoRA-SFT for anchoring the teacher, a judge-filter for channel construction, another LoRA-SFT for student training, and a vision-judge for scoring. It does NOT use any mechanism-intervention technique (activation patching, steering vectors, mean-ablation, causal attribution, or additive intervention on internal representations).

Per the mechanism-audit protocol: "Returns `n/a` when Cx's experiment uses no additive intervention on internal representations." C1 satisfies this condition — no steering coefficient is used, no site is intervened upon, and no representational direction is added or subtracted during the C1 evaluation pipeline.

## Check A: Steering Coefficient Sweep — N/A

C1's M0 pipeline (M0.1–M0.8) does not use a steering coefficient. The LoRA training uses a fixed learning rate (1e-4) that was swept in M0.5, but LR is a training hyperparameter, not a mechanism-intervention coefficient. The α-sweep requirement applies to M2 (C3), not M0.

**Verdict: n/a** — no steering coefficient present.

## Checks B–F: Reserved — N/A

Placeholders for future mechanism-rigor checks. Not applicable to C1's phenomenon-validation design.

## Summary

C1's experimental design has no mechanism-intervention component. Mechanism-audit finds no issues and returns n/a, which per the gate rule is treated as severity=0 (equivalent to pass for the combined verdict computation).
