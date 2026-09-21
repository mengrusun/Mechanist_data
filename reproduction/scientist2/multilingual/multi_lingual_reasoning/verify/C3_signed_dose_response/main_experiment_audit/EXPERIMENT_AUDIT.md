# Experiment Audit Report — Claim C3

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (gpt-5.4 via dmxapi, cross-model)
**Project**: Unified Verification of the Four-Claim Language-Agnostic/Specific Subspace Hypothesis on Qwen-3-4B-Thinking + MGSM
**Claim**: C3 — The steering coefficient α in h ← h + α·Π_lang·h produces MGSM accuracy monotone-decreasing in α over the signed sweep α ∈ [−1.5, +1.5] with A(−1) > A(0) > A(+1) (signed dose-response, negative correlation).
**Linked milestones**: M3

## Overall Verdict: WARN

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
Real MGSM gold answers from dataset parquet files. Same evaluation pipeline as M2. No model-output-derived ground truth.

### B. Score Normalization: PASS
Macro accuracy = mean(per-lang correct/n_problems) with n_problems=50 (fixed dataset size). Numbers match EXPERIMENT_RESULTS.md: A(-1)=0.0509≈0.051 ✓, A(0)=0.7436≈0.744 ✓, A(+1)=0.000 ✓. No normalization fraud.

### C. Result File Existence: PASS
All 18 result files verified (9 V_lang α + 9 random subspace):
- vlang_alpha-1.0_s42_summary.json: macro_acc=0.0509 ✓
- vlang_alpha0.0_s42_summary.json: macro_acc=0.7436 ✓  
- vlang_alpha1.0_s42_summary.json: macro_acc=0.0000 ✓
- random_alpha-1.0_s42_summary.json: macro_acc=0.7400 ✓
- All M3 run directories exist with logs

### D. Dead Code Detection: PASS
Same SteeringHooks and evaluation code as M2 (m2_projection_eval.py). The α-dependent differences in macro_acc across 9 values confirm the intervention is live (dead code would show identical outputs). GlotLID fidelity and accuracy both stored per-run.

### E. Scope Assessment: WARN
- 1 seed only (plan: 3 seeds)
- n=50/lang (plan: n=250/lang)  
- Only winning M2 config (mid, k_top=12, rank_r=2) tested
- BUT: 9-value symmetric α sweep (both signs) with matched random-subspace control at all 9 α — this is the key innovation of M3 and is complete
- The experimental scope is adequate for detecting the sign-asymmetry phenomenon but limited for the strict 3-seed significance tests that EXPERIMENT_PLAN.md's predicate requires

### F. Evaluation Type: real_gt
MGSM gold answers from dataset parquet files.

## Action Items
- [WARN-E] Run 3 seeds at n=250/lang to meet the plan's significance-test predicate; current 1-seed n=50 can support qualitative conclusions but not the plan's bootstrap CI requirements.
