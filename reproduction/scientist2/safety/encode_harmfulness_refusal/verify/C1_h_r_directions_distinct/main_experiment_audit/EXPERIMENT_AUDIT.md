# Experiment Audit Report — Claim C1

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: encode_harmfulness_refusal — Harmfulness and Refusal Directions in Residual Stream
**Claim**: C1 — Two distinct, approximately-linear, independently-recoverable directions in the residual stream
**Linked milestones**: M-prep, M1

## Overall Verdict: WARN
*This is C1's integrity verdict — whether C1's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
Harmfulness labels come from AdvBench dataset (dataset-provided labels, not model output). The refusal-side direction r is extracted from a harmful-vs-benign proxy (is_refusal() classifier on model completions), but this is explicitly documented in directions.json under `contrast_source: "harmful-vs-benign proxy"` and in EXPERIMENT_RESULTS.md. The proxy is transparent and labeled — not silently treated as ground truth. **PASS** on provenance, noting the proxy label.

### B. Score Normalization: PASS
AUROC is computed against dataset-provided harmfulness labels (not model-score denominators). Cosine similarity is a direction-vs-direction geometric measure. No self-normalization observed in m1_claim1_directions.py.

### C. Result File Existence: PASS
results/m1/claim1_verdict.json exists with matching values (auroc_h=0.9998, cos_h_r=0.174, split_half=0.883, auroc_r=NaN). EXPERIMENT_RESULTS.md and EXPERIMENT_TRACKER.md R002 match. All support files (auroc_table.csv, cosine_report.json, shuffled_refusal.json) exist.

### D. Dead Code Detection: WARN
The refusal-side recovery machinery (sub-test i refusal AUROC, sub-test iii shuffled-refusal) is implemented in m1_claim1_directions.py and called, but yields NaN on this data because there were only 7 natural jailbreaks. The code is called but its output branch (refusal sub-tests) cannot be exercised at this data scale. Conservative WARN: the functionality exists but cannot produce results.

### E. Scope Assessment: WARN
The claim states "independently-recoverable directions" but sub-tests i (refusal AUROC) and iii (shuffled-refusal independence check) are unmeasurable at 98.7% baseline refusal rate. The geometry sub-test (ii, cosine) passes cleanly, but the "independently-recoverable" component cannot be verified from this data. The claim text somewhat overstates what was established: only geometric distinctness was demonstrated, not full mutual independence.

### F. Evaluation Type: proxy_gt
Classification: `proxy_gt` — harmfulness side uses real dataset ground truth; refusal side uses model-completion string classifier as proxy (explicitly labeled). Not purely `real_gt` because the r direction validation is proxy-dependent.

## Action Items
- To fully verify "independently-recoverable": run on a model with lower bare-refusal rate (e.g., Qwen2-Instruct-7B), or use a dataset with more natural jailbreaks, so sub-tests i and iii on the refusal side become measurable.
- The existing partial result is honest and documented; no methodology fix required before verify-stage swap can proceed.
