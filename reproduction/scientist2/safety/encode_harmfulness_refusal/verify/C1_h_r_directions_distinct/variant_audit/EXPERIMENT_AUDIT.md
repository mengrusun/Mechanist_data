# Experiment Audit Report — Claim C1 Variant (model-swap-qwen2-instruct-7b)

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: encode_harmfulness_refusal
**Claim**: C1 — Two distinct, approximately-linear, independently-recoverable directions
**Variant**: model-swap-qwen2-instruct-7b
**Linked milestones**: m_prep, m1

## Overall Verdict: WARN
*Variant methodology is sound with a minor caveat on NaN AUROC fields.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
Same as main experiment: AdvBench/Alpaca dataset labels for harmfulness; refusal proxy explicitly documented in variant's directions.json as "harmful-vs-benign proxy". No GT from model outputs without labeling.

### B. Score Normalization: PASS
Unchanged from main: AUROC vs dataset labels; cosine is geometric; no self-normalization.

### C. Result File Existence: WARN
Required files exist: m_prep/directions.json, m1/claim1_verdict.json. Numbers are internally consistent. WARN because claim1_verdict.json reports auroc_r=NaN — this is an expected and documented outcome (Qwen2 also refuses nearly all bare-harmful prompts, so the refusal sub-test val set has near-zero positive class), but NaN is not a "valid numeric" result in the strictest sense.

### D. Dead Code Detection: PASS
Same situation as main: refusal sub-tests are called but return NaN due to no natural jailbreaks in Qwen2's responses on AdvBench. Code is called, output is data-conditional NaN.

### E. Scope Assessment: PASS
Model swap is properly scoped: same 520 pairs, same seed, same evaluation metrics. Reduced layer count (28 vs 32) is documented in DIFF.md. No scope overclaim.

### F. Evaluation Type: proxy_gt
Same as main experiment.

## Action Items
None that would prevent using this variant's verdict. The NaN AUROC_r is the expected finding (same mechanism as main experiment).
