# Experiment Audit Report — Claim C2

**Date**: 2026-07-14
**Auditor**: executor (Claude Sonnet 4.6) + evidence review
**Project**: Emotional Framing in Prompts as a Weak, Input-Dependent Signal
**Claim**: C2 — inter-emotion spread on SocialIQA ≥ 2× that on GSM8K; ordering spread_social > spread_factual > spread_math in ≥ 2 of 3 comparisons
**Linked milestones**: M2, M3, M4

## Overall Verdict: WARN

*This is C2's integrity verdict — whether C2's experimental process is methodologically sound.*

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
- Evidence:
  - GSM8K: `scripts/run_prefix_eval.py:34-43` — dataset-provided gold via `#### answer` field.
  - SocialIQA: `scripts/run_prefix_eval.py:47-63` — gold loaded from `dev-labels.lst` file (dataset-provided). Letter mapping `{"1":"A","2":"B","3":"C"}[lab.strip()]` at line 61 is correct for SocialIQA's label format.
  - MedQA: `scripts/run_prefix_eval.py:66-79` — gold loaded from jsonl's `answer_idx` field (dataset-provided).
  - No model output used as GT.
- Details: All three tasks use dataset-provided ground truth. Eval type: `real_gt` for all three.

### B. Score Normalization: PASS
- Evidence: `scripts/run_prefix_eval.py:338-348` — MCQ mode computes `correct = int(pred == gold)` (line 343); accuracy is `n_correct / len(results)`. No normalization by model output. The MCQ log-probability selection uses `max(lp_dict, key=lp_dict.get)` (line 203) — argmax over dataset labels, not a model-derived normalization.
- Details: Spread is computed as max(acc) - min(acc) over the 24 emotional conditions. This is a difference of dataset-evaluated accuracies, not a normalized score.

### C. Result File Existence: WARN
- Evidence:
  - SocialIQA: 26 M3 result files expected at `runs/M3/socialiqa_<condition_id>.json`. EXPERIMENT_TRACKER.md row shows `done` with spread=0.030.
  - MedQA: 26 M3 result files expected at `runs/M3/medqa_<condition_id>.json`. EXPERIMENT_TRACKER.md shows `done` with spread=0.028.
  - M4 analysis: `reports/M4_c3_analysis.json` expected — the C2 section of this file should contain spread values. Tracker shows `done`.
  - Note: `analyze_c3.py:155-178` computes C2 spread as `np.max(accs) - np.min(accs)` over conditions **excluding neutral and filler** (line 156-157). This is correct per C2's criterion ("inter-emotion spread over 24 emotional conditions").
  - WARN: C2's claim statement requires comparing spread_social ≥ 2 × spread_math. The realized result (spread_math=0.186 >> spread_social=0.030) disconfirms the claim, which is fine — what matters is methodology integrity. However, a subtle issue: C2's MCQ evaluation uses log-likelihood over letter tokens, while GSM8K uses CoT generation. This evaluation-mode difference may contribute to the much smaller SocialIQA/MedQA spreads (constrained MCQ is less sensitive to prefix wording than free-form CoT generation). This confound is correctly documented in EXPERIMENT_RESULTS.md but the claim does not account for it — the spread comparison is between structurally different evaluation modes.
- Details: WARN because the inter-task spread comparison conflates different evaluation modes (CoT vs MCQ log-likelihood), which is a methodological confounder. Result files appear to exist per tracker status.

### D. Dead Code Detection: PASS
- Evidence: `scripts/analyze_c3.py:155-178` — C2 spread computation at lines 155-179 is called within `main()` and results are written to the output JSON at line 180+. No dead code for C2's metrics.
- Details: The C2 spread computation is called and its results are written to `reports/M4_c3_analysis.json`.

### E. Scope Assessment: WARN
- Evidence: 26 conditions × 500 items × 2 tasks (SocialIQA, MedQA) = 26,000 evaluations. Scope is adequate in quantity.
- WARN: The claim predicts `spread_social > spread_factual > spread_math` but uses structurally non-comparable evaluation modes (CoT for GSM8K, MCQ for SocialIQA/MedQA). EXPERIMENT_RESULTS.md notes this confound explicitly ("GSM8K's larger spread is driven by CoT-based generation variance"). The scope is fine but the evaluation design introduces a mode confound that partially invalidates the cross-task spread comparison.

### F. Evaluation Type
- GSM8K: `real_gt` (CoT generation against dataset gold)
- SocialIQA: `real_gt` (MCQ log-likelihood against dataset gold labels)
- MedQA: `real_gt` (MCQ log-likelihood against dataset gold)
- Mixed evaluation modes across tasks under comparison.

## Action Items
1. In the paper/report, explicitly acknowledge that the CoT vs MCQ log-likelihood evaluation-mode difference is a confound for C2's cross-task spread comparison. C2's result (spread_math >> spread_social) is likely partly attributable to this design choice, not purely to content differences between tasks.
2. Consider running SocialIQA/MedQA in CoT mode (or GSM8K in MCQ mode) as a robustness check for C2 — this is a verify-stage concern.
3. WARN does not block Stage 2 — C2 advances with an integrity warning.
