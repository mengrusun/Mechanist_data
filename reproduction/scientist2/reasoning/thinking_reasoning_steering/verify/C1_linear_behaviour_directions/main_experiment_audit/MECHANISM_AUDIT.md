# Mechanism Audit Report — Claim C1

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, GPT-5.4 via DMX, T=0)
**Project**: Linear Steering of Reasoning Behaviours in DeepSeek-R1-Distill
**Claim**: C1 — Each reasoning behaviour maps onto an approximately linear direction in the residual stream.
**Linked milestones**: M1

## Overall Verdict: N/A
*C1's mechanism-rigor verdict. N/A means no catalogue check was triggered for C1's scope — M1 uses probe classification (linear probe + first-PC alignment), not additive steering. No α parameter in M1.*

## Triggered checks (this run): (none)

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: no — M1 (the sole milestone for C1) performs forward-pass activation capture, logistic regression probe training, and mean-difference direction extraction. No additive intervention with a scalar coefficient is performed in M1. The `run_M1_locate.py` script does not contain `alpha`, `steer`, `SteeringHook`, or any activation addition. Check A trigger keywords are absent from C1's milestone scripts.

### B–F. Reserved (not_implemented)
Status: not yet implemented. Future checks may cover direction-extraction quality, site / layer selection, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items
None — C1 uses no mechanism intervention in its primary milestone (M1). The steering mechanism appears first in M3 (C3's milestone).
