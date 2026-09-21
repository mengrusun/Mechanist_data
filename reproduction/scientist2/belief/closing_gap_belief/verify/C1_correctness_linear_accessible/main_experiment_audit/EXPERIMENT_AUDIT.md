# Experiment Audit Report — Claim C1

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (gpt-5.4, via llm-chat API, cross-model)
**Project**: Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence
**Claim**: C1 — Llama-3.1-8B-Instruct encodes well-calibrated gold correctness on TriviaQA in a linearly accessible direction of the residual-stream hidden state at the canonical hook, beating token-probability and null baselines.
**Linked milestones**: M1 (B1), M1.5 (B5)

## Overall Verdict: WARN

*This is C1's integrity verdict — whether C1's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
Ground truth is explicitly from the real dataset: `y_correct` = TriviaQA alias-based normalized exact-match with code-review fix ("reverse-substring excluded"). This is real GT, not model-output-derived. The fix (whole-word contiguous span or exact match required) is documented in EXPERIMENT_TRACKER.md code-review ledger.

### B. Score Normalization: PASS
No evidence of score normalization by model prediction statistics. Reported metrics are raw AUROC/ECE, with isotonic calibration fitted on dev set only. No metric divided by model's own max/mean.

### C. Result File Existence: WARN
Referenced files exist (`artifacts/probe_metrics.json`, `artifacts/nulls.json`). Core numbers match the reported values (AUROC 0.840, ECE_iso 0.031, shuffled-label null 0.497). However, there is a minor internal inconsistency: the bootstrap CI [0.818, 0.835] does not contain the point estimate 0.840. This occurs because the CI is from bootstrap resamples evaluated on the test set, while 0.840 is the full-data test AUROC — a normal statistical artifact, but flagged as a consistency concern. Additionally, bootstrap n=200 instead of planned 1000 (explicitly disclosed; CI half-widths confirmed < 0.03).

### D. Dead Code Detection: PASS
All metric functions (`_fit_lr_probe`, `_ece`, `_isotonic_calibrate`, AUROC, bootstrap CI) are called in `step4_probes.py` and their outputs appear in `artifacts/probe_metrics.json`. No dead-code concern.

### E. Scope Assessment: PASS
Scope is clearly limited to one model (Llama-3.1-8B-Instruct) / one dataset (TriviaQA rc.nocontext, 10k questions) / one run. Paper scope explicitly qualified as single-model, single-dataset. Bootstrap reduced from 1000 to 200 — flagged explicitly in results with CI widths confirmed tight. No over-broad scope language detected.

### F. Evaluation Type: real_gt
Uses TriviaQA dataset-provided gold answer aliases for correctness scoring.

## Action Items
- The bootstrap CI not containing the point estimate is a minor statistical artifact, not a reporting error. Document in paper that the CI derives from bootstrap resamples and the point estimate from the full training split.
- Consider reporting the relationship between point estimate and bootstrap mean (0.826) to avoid reader confusion.
