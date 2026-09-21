# Mechanism Audit — C3: Aligned Reproduces Human Behaviour and Uncertainty

**Claim**: C3 — Aligned model reproduces human choice, uncertainty, and RSA.
**Mechanism family**: Representation and Parameter Analysis / Parameter-Space Task Vectors
**Role**: Recover step — M6 evaluates `apply_to(θ_pretrained, τ_aligned, coef=1.0)` on human behavioral metrics. No additive intervention; forward-only evaluation.

## Check A — Steering Coefficient Sweep
**Finding**: N/A — M6 is a forward-only behavioral comparison of the M3 checkpoint vs M4 (unaligned). No additive direction intervention. The task-vector `τ_aligned` was already applied to produce `θ_M3`; M6 just evaluates it.
**Severity**: N/A
## Checks B–F: not_implemented
## Overall Verdict: **overall_verdict: n/a**
