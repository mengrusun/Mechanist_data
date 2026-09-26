# Mechanism Audit — Variant: model-swap-judge-gpt4o (C1)

**Audit date**: 2026-07-10
**Variant**: `C1/model-swap-judge-gpt4o`
**Dimension**: model (eval judge swap)
**Auditor**: auto-verify Phase 9 (variant integrity gate)

## Summary

**Overall verdict**: N/A

C1 is a behavioral phenomenon claim (accuracy drop across seeds). The variant makes no mechanism intervention — no activation steering, no direction extraction, no alpha sweep. Mechanism rigor audit returns n/a, which carries severity=0 and applies no penalty to the Phase 9 combined integrity gate.

This is identical in structure to the baseline mechanism audit for C1 (the main experiment's M0 phase uses no mechanism intervention; M1/M2 CAA is scoped to C3).

## Phase 9 combined verdict

`combined = max_severity(experiment=PASS, mechanism=n/a) = PASS`

Variant `model-swap-judge-gpt4o` is integrity-eligible.
