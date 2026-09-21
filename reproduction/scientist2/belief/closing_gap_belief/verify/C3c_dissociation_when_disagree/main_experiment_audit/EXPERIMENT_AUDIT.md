# Experiment Audit Report — Claim C3c

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (gpt-5.4, via llm-chat API, cross-model)
**Project**: Orthogonal Linear Subspaces of Gold Calibration and Verbalized Confidence
**Claim**: C3c — Dissociation-when-disagree: for questions where the probe-derived correctness score is LOW, accuracy is HIGHER when the verbalized confidence is LOW than when it is HIGH — McNemar test p < 0.05.
**Linked milestones**: M4 (B4)

## Overall Verdict: WARN

*This is C3c's integrity verdict — whether C3c's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
Uses real correctness GT from TriviaQA for the accuracy comparison (same y_correct from step4's alias-based normalized exact-match scoring). Verbalized confidence c also pre-existing from step3/step4. No model-output-as-GT fraud.

### B. Score Normalization: PASS
No suspicious normalization against model prediction statistics. Accuracy is computed as fraction of correct answers in each cell. No normalization by model output max/mean.

### C. Result File Existence: PASS
`artifacts/dissociation.json` exists and clearly reports: z=1.35, p_one_sided=0.91, (probe_low, verbal_low) n=4, passes=false. The result is an honest null — the file accurately reflects the analysis. The numbers are consistent with the EXPERIMENT_RESULTS.md reporting. No fabrication.

### D. Dead Code Detection: PASS
The dissociation analysis is performed in step6_analyses.py (analysis-only, reusing B1 cached data). Output appears in dissociation.json. No dead-code concern.

### E. Scope Assessment: WARN
The claim tests a specific contingency cell (probe_low, verbal_low) that contains only n=4 samples out of 2000 test cases. This extreme sparsity (base rate 2/1000) renders the test untestable in its planned form. The claim is explicitly reported as NOT SUPPORTED with the tag [suspected under-power: (probe_low, verbal_low) cell n=4]. This is an honest reporting of a design-of-test failure forced by the extreme skew of verbalized c (96% >= 95). The issue is scope-adequacy, not scope over-claim: the experiment ran at the pre-registered scope but the scope was insufficient for the specific contingency.

### F. Evaluation Type: real_gt
Uses TriviaQA gold answers for correctness. Verbalized confidence bins from model output (c), but the accuracy metric is GT-based.

## Action Items
- C3c is honestly reported as not-supported due to under-power. No integrity issue — the failure is a legitimate null finding.
- For future work: use a dataset or prompt design where verbalized confidence has a broader distribution (not 96% clustered at 100).
