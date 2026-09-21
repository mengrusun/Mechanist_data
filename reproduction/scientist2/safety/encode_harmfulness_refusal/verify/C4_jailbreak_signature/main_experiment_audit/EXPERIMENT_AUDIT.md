# Experiment Audit Report — Claim C4

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: encode_harmfulness_refusal
**Claim**: C4 — A notable class of successful jailbreaks suppresses r while h remains active
**Linked milestones**: M4

## Overall Verdict: WARN
*This is C4's experiment-methodology integrity verdict.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
Attack success adjudication uses Llama Guard 3 8B (external pretrained classifier, not fine-tuned on this project's data) + AdvBench string-match refusal check. Projection labels (successful vs failed) derive from actual model behavior. No model-output-derived GT used as training signal.

### B. Score Normalization: PASS
Delta projections are dot-products onto unit-normed directions (geometric). AUROC and ASR are ratio statistics over discrete outcomes. No self-normalization.

### C. Result File Existence: PASS
results/m4/claim4_verdict.json exists. GCG and PAP metric files exist. ASR=0/500 for both families matches EXPERIMENT_RESULTS.md. Tracker R020-R021 marked done+neg.

### D. Dead Code Detection: PASS
The signature measurement code (delta_r, delta_h, detection AUROC) is implemented and called, but returns NaN for successful subset because ASR=0. The code correctly handles the empty-subset case. Not a dead-code issue — it's data-conditional output, not unused code.

### E. Scope Assessment: PASS
The claim says "notable class" (not "all" or "comprehensive"). M4 tested 2 attack families × 500 attempts each = 1000 total. The result is not-supported because ASR=0 (model too well-aligned), not because the scope was too narrow. The word "notable" is appropriate given the plan's acknowledged risk of low ASR.

### F. Evaluation Type: real_gt
Llama Guard 3 8B is used for success adjudication (external ground truth classifier, published, not tuned on this project). String-match refusal check is an explicit proxy with documented behavior. Classification: real_gt for the success labels.

## Action Items
- The not-supported verdict here is honest and correctly reported. No methodology fix needed.
- For a future retry: use a model with lower baseline refusal or attack families optimized for Llama-3-8B specifically.
