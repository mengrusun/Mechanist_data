# Mechanism Audit Report — Claim C2

**Date**: 2026-07-20
**Auditor**: expert self-review (llm-chat MCP unavailable — graceful degradation)
**Claim**: C2 — Banana signal location in DiT (M1: parameter-space task vectors)
**Linked milestones**: M1

## Overall Verdict: N/A

## Integrity Status: n/a

## Rationale

C2 (M1) uses the **Parameter-Space Task Vectors** method — a read-only weight-space diagnostic that computes Grassmann subspace overlap of LoRA ΔW matrices. It does NOT apply any additive intervention on internal representations during inference. The method is purely analytical (load weights → compute SVD → compare subspaces). No steering coefficient, no activation patch, no hook on the forward pass.

Per the mechanism-audit protocol: "Returns `n/a` when Cx's experiment uses no additive intervention on internal representations." C2's M1 analysis fully satisfies this condition.

The steering-coefficient sweep requirement (Check A) applies to M2 (C3), where the M1-extracted `banana_direction.pt` is used as a Steering Vector for additive interventions on the forward pass.

## Check A: Steering Coefficient Sweep — N/A

C2's M1 analysis performs no steering. No α coefficient. No sweep needed.

## Checks B–F: Reserved — N/A

Not applicable to C2's parameter-space analysis design.

## Summary

C2 uses no mechanism intervention. Mechanism-audit returns n/a (severity=0, equivalent to pass for combined verdict).
