# Experiment Audit Report — Claim C5

**Date**: 2026-07-15
**Auditor**: external LLM reviewer (cross-model, via llm-chat MCP; model=gpt-5.4)
**Project**: encode_harmfulness_refusal
**Claim**: C5 — A lightweight harmfulness-direction probe matches or beats Llama Guard 3 8B at flagging jailbreaks, at a fraction of the compute
**Linked milestones**: M5

## Overall Verdict: FAIL
*This is C5's experiment-methodology integrity verdict.*

## Integrity Status: fail

## Checks

### A. Ground Truth Provenance: PASS
Probe training uses AdvBench harmful / Alpaca benign labels (dataset-provided, not model output). Llama Guard 3 8B is used as a comparator baseline, not as a label generator for probe training. Test labels (harmful vs benign/safe) come from dataset category assignments. GT provenance is clean.

### B. Score Normalization: PASS
AUROC is computed against dataset-derived ground truth labels (harmful/benign). No metric is divided by model-predicted quantities. FLOPs comparison uses parameter-count estimates, not model-output statistics.

### C. Result File Existence: PASS
results/m5/claim5_verdict.json exists. Numbers match EXPERIMENT_RESULTS.md (AUROC_probe=1.000, AUROC_LG=0.9992, compute_ratio=1.05e-8). Tracker R022-R023 marked done. results/m5/linear_probe/ and results/m5/shallow_mlp/ directories exist with probe_metrics.json.

### D. Dead Code Detection: PASS
Both probe variants (linear_probe and shallow_mlp) are implemented and run. Llama Guard 3 8B evaluation is called (llamaguard_used=true in verdict). All code paths exercised.

### E. Scope Assessment: FAIL
**CRITICAL SCOPE MISMATCH**: The claim title and abstract say "matches or beats Llama Guard 3 8B at flagging **jailbreaks**" (C5 task.md wording). However, the M5 test set contains **0 successful jailbreak items** (M4 ASR=0; documented in claim5_verdict.json as `successful_jb: 0`). The 400-item test set consists of bare-harmful-refused (100) + benign-compliant (100) + XSTest-safe (200) items. This is a materially easier task than "flagging jailbreaks": jailbreaks are specifically those attacks that bypass refusal while containing harmful content — they would not be flagged by a naive harmful-content classifier because the refusal signal is suppressed. The probe reaching AUROC=1.000 on a simpler harmful-vs-benign task does not establish that it can flag jailbreaks in the intended scenario. The scope language in C5's claim (task.md) directly references "jailbroken + benign-compliant + benign-lookalike" test set, but the actual test set substituted bare-harmful-refused for jailbroken. This is a scope mismatch that overstates what was measured.

### F. Evaluation Type: real_gt
AdvBench harmful + Alpaca benign labels are dataset-provided ground truth. XSTest safe labels are dataset-provided. Classification: real_gt.

## Action Items
- **Required for claim support**: retest on a test set that includes successful jailbreak instances. This requires a model/attack combination where ASR > 0.
- Until then, C5 should be re-scoped to "matches or beats Llama Guard 3 8B at flagging **harmful prompts** (bare-harmful vs benign, including XSTest lookalikes)" — a true but narrower claim.
- The compute-fraction result (1.05e-8 of LG FLOPs) is fully supported and requires no change.
