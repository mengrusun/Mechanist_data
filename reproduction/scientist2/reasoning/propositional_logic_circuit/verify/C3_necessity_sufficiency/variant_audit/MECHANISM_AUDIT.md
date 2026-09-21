# Mechanism Audit Report — Claim C3 Variants

**Date**: 2026-07-15
**Auditor**: executor trigger detection (early N/A exit — no LLM reviewer call needed)
**Project**: Sparse Modular Circuit for Propositional-Logic Reasoning
**Claim**: C3 — necessity + sufficiency
**Variant audited**: model-swap-gemma2-9b (reuses results/M5_gemma9b.json)

## Overall Verdict: N/A

*The C3 variant (M5 replay on Gemma-2-9B) uses the same replacement activation patching as the main experiment — `cross_family_verify.py` replicates M1 (attribution screen) + M2 (necessity path patching) + M3 (sufficiency reinsertion) using the same hook-based replacement mechanism in `prop_circuit_lib.py`. No additive scalar steering coefficient `h += alpha * direction` pattern is present. Check A (steering coefficient sweep) does not apply.*

## Integrity Status: n/a

## Checks

### A. Steering Coefficient Sweep: N/A
Trigger check: same as main experiment — replacement activation patching, no scalar alpha. `cross_family_verify.py` hooks into Gemma-2-9B attention heads and MLP layers via TransformerLens, replaces activations with clean-run values (necessity) or resample-ablated values (sufficiency). No sweep, no alpha.

### B–F. Reserved Checks: not_implemented
