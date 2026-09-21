# Mechanism Audit Report — Claim C2

**Date**: 2026-07-14
**Auditor**: executor (Claude Sonnet 4.6)
**Project**: Emotional Framing in Prompts as a Weak, Input-Dependent Signal
**Claim**: C2 — inter-emotion spread on SocialIQA ≥ 2× that on GSM8K; task-family ordering
**Linked milestones**: M2, M3, M4

## Overall Verdict: N/A

*C2 is a pure behavioral evaluation claim comparing accuracy spreads across task families. It uses no mechanism intervention.*

## Triggered checks (this run): (none)

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: no
- Reason: C2's milestones (M2, M3, M4) involve only behavioral evaluation runs (vLLM generation/MCQ scoring) and post-hoc analysis. No additive activation modification, no steering, no patching. No trigger keywords matched in scripts/run_prefix_eval.py or scripts/analyze_c3.py in C2's scope.

### B–F. Reserved (not_implemented)

## Action Items
- None (N/A verdict; downstream combination treats N/A as PASS).
