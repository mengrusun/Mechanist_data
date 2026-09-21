# Mechanism Audit Report — Claim C1

**Date**: 2026-07-14
**Auditor**: executor (early N/A exit — no implemented check triggered; no reviewer call needed)
**Project**: Unified Verification of the Four-Claim Language-Agnostic/Specific Subspace Hypothesis on Qwen-3-4B-Thinking + MGSM
**Claim**: C1 — Hidden representations of Qwen-3-4B-Thinking decompose into a language-specific subspace V_lang (identifiable from a small multilingual probe set via SVD/mean-difference) and an approximately orthogonal language-agnostic residual.
**Linked milestones**: M1

## Overall Verdict: N/A
*N/A means no catalogue check was triggered for C1's scope. C1's M1 milestone uses language-mean-difference SVD + linear classifier probing — no additive residual-stream intervention with a scalar coefficient α is applied. Check A (steering coefficient sweep) trigger did not fire. Downstream combination treats N/A as PASS.*

## Triggered checks (this run): (none)

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: no — M1 scripts (`mlr/m1_locate.py`, `mlr/m1_locate_grid.py`, `mlr/activations.py`, `mlr/aggregate_m1.py`) contain no steering/additive-intervention keywords (`steer`, `CAA`, `alpha * v`, `activations += alpha`, `h = h + alpha*`, etc.). C1's mechanism is a probe-based localization step (SVD + linear classifier + principal-angle test), not a causal additive intervention.

### B–F. Reserved (not_implemented)
Status: not yet implemented. Future checks may cover direction-extraction quality, site / layer selection, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items
- None. C1's M1 experiment does not exercise an additive-intervention mechanism. No mechanism-rigor flags to resolve.
