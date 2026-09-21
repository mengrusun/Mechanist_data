# Mechanism Audit Report — Claim C2

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: Reproduction of Five SAE-on-ESM-2 Interpretability Claims
**Claim**: C2 — SAE concept alignment gap vs. neurons.
**Linked milestones**: M2

## Overall Verdict: N/A
*C2's experiment uses no mechanism intervention. M2 computes per-residue F1 alignment between unit activation masks and concept annotation masks — a pure measurement, no additive intervention on hidden states.*

## Triggered checks (this run): (none)

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: no (M2 = alignment measurement; no scalar alpha multiplying an additive hidden-state contribution)

### B–F. Reserved (not_implemented)

## Action Items
- None.
