# Experiment Audit Report — Claim C3 (Variant)

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (cross-model, gpt-5.4 via dmxapi, via llm-chat MCP)
**Project**: Unified Verification of the Four-Claim Language-Agnostic/Specific Subspace Hypothesis on Qwen-3-4B-Thinking + MGSM
**Claim**: C3 — The steering coefficient α in h ← h + α·Π_lang·h produces MGSM accuracy monotone-decreasing in α over the signed sweep α ∈ [−1.5, +1.5] with A(−1) > A(0) > A(+1) (signed dose-response, negative correlation).
**Variant**: model_swap_deepseek_r1_llama8b (DeepSeek-R1-Distill-Llama-8B replacing Qwen-3-4B-Thinking)
**Scope**: verify/C3_signed_dose_response/variants/model_swap_deepseek_r1_llama8b/

## Overall Verdict: WARN
*This is C3's variant-level integrity verdict — whether the variant's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
MGSM accuracy ground truth is taken from the dataset, not from model outputs. In the eval loop, gold is read from `df.iloc[i].get('answer_number', None)` with fallback to the 'answer' column (mlr/m2_projection_eval.py:248-253), then compared against the extracted model answer via `numeric_equal()`. This is real dataset GT provenance, not synthetic or self-generated labels.
Evidence: `mlr/m2_projection_eval.py:248-254, 265-271`

### B. Score Normalization: PASS
No score is normalized by any model-output statistic. Per-language accuracy is computed as `n_correct / n` where `n` is the number of dataset problems. Macro accuracy is `np.mean(accs)` over the 11 per-language values (mlr/m2_projection_eval.py:299-303). The alpha=0 summary internal check: mean(per-language accs) ≈ 0.4473 matches reported macro_accuracy. Raw-count normalization only.
Evidence: `mlr/m2_projection_eval.py:294-303`

### C. Result File Existence: WARN
All 9 vlang summary files exist and are non-empty (alpha=-1.5,-1.0,-0.5,-0.25,0.0,+0.25,+0.5,+1.0,+1.5). The alpha=0 full summary JSON is internally consistent (mean of 11 per-language accuracies = 0.4473 = macro_accuracy). However, only the alpha=0 summary was provided in full detail; the other 8 summaries are represented only by top-line macro_accuracy values. Their per-language-to-macro consistency was not independently verified. Additionally, random subspace control results (deploy.sh computes them) are not yet fully available at audit time.
Evidence: `verify/C3_signed_dose_response/variants/model_swap_deepseek_r1_llama8b/results/vlang_alpha*.json` (9 files confirmed present and non-empty)

### D. Dead Code Detection: WARN
`extract_answer()` is called in the evaluation loop (`mlr/m2_projection_eval.py:271`). GlotLID fidelity is computed conditionally and stored. `SteeringHooks` is used. However, `grade()` is imported from `mlr.mgsm_eval` (`mlr/m2_projection_eval.py:33`) but never called in the evaluation loop — only `extract_answer()` and `numeric_equal()` are used directly. This is a minor dead import but not integrity-breaking.
Evidence: `mlr/m2_projection_eval.py:33, 256-293` — grade() imported but unused

### E. Scope Assessment: WARN
Scope evaluated: 11 languages (en,es,fr,de,zh,ja,ru,th,te,bn,sw), 9 alpha values (-1.5 to +1.5), 50 problems/language, 3-shot, 1 seed (seed=42). Single-seed evaluation is the primary concern. The results do not support Claim C3's monotone-decrease: A(-1)=0.3909 < A(0)=0.4473 (C3 requires A(-1) > A(0)), though A(+1)=0.000 < A(0). The claim is violated but the scope is sufficient to judge this.
Evidence: `verify/C3_signed_dose_response/variants/model_swap_deepseek_r1_llama8b/results/vlang_alpha*_summary.json` — 9 alpha values, 1 seed, n=50/lang

### F. Evaluation Type: real_gt
MGSM benchmark with numeric answer comparison against dataset-provided gold answers. GlotLID language identification used as a secondary fidelity metric.

## Action Items
- Provide full per-language breakdowns for all 9 alpha values (not just alpha=0) to enable independent per-file consistency checks
- Run additional seeds (currently single-seed) to increase confidence in the dose-response pattern
- Include random-subspace control results in the audit scope when they complete
- Clean up unused imported function grade() from mlr/m2_projection_eval.py or document its retention
- Note: the variant correctly identifies C3 as NOT supported (non-monotone pattern observed)
