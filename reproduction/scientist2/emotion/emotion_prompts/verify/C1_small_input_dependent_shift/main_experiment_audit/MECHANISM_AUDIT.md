# Mechanism Audit Report — Claim C1

**Date**: 2026-07-14
**Auditor**: executor (Claude Sonnet 4.6)
**Project**: Emotional Framing in Prompts as a Weak, Input-Dependent Signal
**Claim**: C1 — Per-emotion mean|Δaccuracy vs neutral| ≤ format-perturbation noise floor AND per-item sign-consistency ∈ [0.4, 0.6] on Qwen3-14B/GSM8K
**Linked milestones**: M2, M2b

## Overall Verdict: N/A

*C1 is a pure behavioral evaluation claim — it uses no mechanism intervention (no additive activation modification, no steering, no patching). The residual-stream activations cached during M2 are used by M5 (CM claim), not by C1 itself.*

## Triggered checks (this run): (none)

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: no
- Reason: C1's evaluation (M2/M2b) runs prefix evaluation via vLLM generation (run_prefix_eval.py) with no additive intervention on internal representations. No `steer`, `alpha`, `activation_patch`, or `CAA`/`RepE`/`DAS` keywords present in M2/M2b scripts in scope of C1. Check A trigger not matched.

### B–F. Reserved (not_implemented)
Status: not yet implemented. Future checks may cover direction-extraction quality, site/layer selection, n_effective sufficiency, probe-vs-causal disentanglement, intervention scope.

## Action Items
- None (N/A verdict; downstream combination treats N/A as PASS).
