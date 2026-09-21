# Variant Mechanism Audit Report — Claim C1

**Date**: 2026-07-18
**Auditor**: executor (Claude) — early N/A exit, no reviewer call needed
**Claim**: C1 — behavioral phenomenon-validation claim (M0 only)
**Variant**: method-swap-rule-based-scorer

## Overall Verdict: N/A

C1's method-swap variant uses a deterministic regex scorer (`score_rule_based.py`) to re-score saved model answers. No additive activation intervention with a scalar coefficient alpha is involved. Check A trigger = false (no steer/CAA/DAS/RepE/patching keywords in variant code).

## Trigger Detection (Check A)

Scanned `score_rule_based.py` for steering keywords: no matches. The variant does no model inference at all — it only reads saved JSONL answer strings and extracts option letters via regex.

**triggered[A] = false** → early N/A exit.

## Checks

### A. Steering Coefficient Sweep: N/A (not triggered)
### B–F. Reserved: not applicable
