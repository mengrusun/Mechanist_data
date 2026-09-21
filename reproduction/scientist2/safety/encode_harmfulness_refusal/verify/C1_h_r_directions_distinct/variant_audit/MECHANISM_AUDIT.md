# Mechanism Audit Report — Claim C1 Variant (model-swap-qwen2-instruct-7b)

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: encode_harmfulness_refusal
**Claim**: C1 — Two distinct, approximately-linear, independently-recoverable directions
**Variant**: model-swap-qwen2-instruct-7b

## Overall Verdict: N/A
*No additive steering intervention in C1's variant; Check A not triggered.*

## Triggered checks (this run): (none)

## Checks

### A. Steering Coefficient Sweep: N/A
- Triggered: no (C1 model-swap variant uses probe AUROC and cosine similarity; no additive scalar steering intervention)

### B–F. Reserved (not_implemented)

## Action Items
None — N/A is not a failure.
