# Experiment Audit Report — Claim C4

**Date**: 2026-07-14
**Auditor**: external LLM reviewer (gpt-5.4 via dmxapi, cross-model)
**Project**: Unified Verification of the Four-Claim Language-Agnostic/Specific Subspace Hypothesis on Qwen-3-4B-Thinking + MGSM
**Claim**: C4 — Training-free null-space projection (the M2 mechanism) achieves ≥ 85% of the accuracy gain of LoRA-SFT fine-tuning on the same MGSM-related task, demonstrating that the V_lang subspace captures most of the representational capacity needed for multilingual generalization.
**Linked milestones**: M4a, M4b

## Overall Verdict: FAIL

## Integrity Status: fail

## Checks

### A. Ground Truth Provenance: PASS
Real MGSM gold answers from dataset parquet files; same pipeline as M2/M3. LoRA-SFT eval uses the identical `mgsm_eval.py` extraction + grading. No model-output-derived GT.

### B. Score Normalization: PASS
Macro accuracy = mean(per-lang correct/n_problems) with n_problems=50 (fixed). Reported baseline 0.762 matches M2/M3 baselines; LoRA macro_acc=0.558 is a raw accuracy, not self-normalized.

### C. Result File Existence: WARN
M4a result file `results/m4a/eval.jsonl` exists with macro=0.558. However M4b (RL fine-tuning comparison component) result files are absent — no `results/m4b/` directory; M4b milestone marked not run. The claim as stated requires a training-free vs. fine-tuning comparison; only the fine-tuning (LoRA-SFT) leg is on disk.

### D. Dead Code Detection: PASS
`mlr/m4_eval.py` is called in the deployment pipeline; `compute_macro_accuracy()` is invoked and its output appears in `results/m4a/eval.jsonl`. No phantom functions detected.

### E. Scope Assessment: FAIL
Critical scope gaps:
1. M4b (the "training-free null-space projection achieves ≥ 85% of SFT gain" comparison) was **never run** — EXPERIMENT_TRACKER.md shows M4b as TODO.
2. LoRA-SFT training used 5,001 examples out of 73,559 available (6.8% of the planned dataset); this is not documented in EXPERIMENT_RESULTS.md.
3. Evaluation: n=50/lang instead of planned n=250/lang (20% of planned scope).
4. The stated ≥ 85% relative-gain predicate cannot be evaluated without M4b's null-space baseline gain figure.
5. EXPERIMENT_RESULTS.md reports "not-supported" for C4, but the scope is insufficient to substantiate that conclusion definitively.

### F. Evaluation Type: real_gt
MGSM gold answers from dataset parquet files.

## Action Items
- [FAIL-E] Run M4b: null-space projection (M2 winning config: mid, k_top=12, rank_r=2) MGSM accuracy measurement as the "training-free" leg of the comparison — without it, the ≥85% predicate is untestable.
- [FAIL-E] Re-run LoRA-SFT with full planned dataset (73,559 examples) or document intentional reduction.
- [FAIL-E] Re-evaluate with n=250/lang to meet plan's scope predicate.
