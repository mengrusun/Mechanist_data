# Experiment Audit Report — Claim C3b

**Date**: 2026-07-14
**Auditor**: executor (Claude Sonnet 4.6) + evidence review
**Project**: Emotional Framing in Prompts as a Weak, Input-Dependent Signal
**Claim**: C3b — For ≥ 3 of 6 emotions on GSM8K, paired Δ(intensity-2 − intensity-1) 95% CI straddles 0 or reverses sign
**Linked milestones**: M4 (post-hoc on M2)

## Overall Verdict: WARN

## Integrity Status: warn

## Checks

### A. Ground Truth Provenance: PASS
- Same as C3a — inherited from M2/M3 dataset-provided GT. No new GT.

### B. Score Normalization: PASS
- `analyze_c3.py:132-135`: delta = v2 - v1 where v1, v2 are per-item binary correctness arrays. Mean delta is raw mean of differences (bounded [-1, 1]). No normalization by model output.

### C. Result File Existence: WARN
- Evidence: C3b result at `reports/M4_c3_analysis.json` exists. EXPERIMENT_RESULTS.md:120-122 reports 3/6 emotions meet the straddling criterion on GSM8K, consistent with `analyze_c3.py`'s logic.
- WARN issue: C3b's predicate requires "≥ 3 of 6 emotions on GSM8K" — the result is exactly at threshold (3/6). At this threshold value, the predicate is satisfied only if ALL 3 of those emotions unambiguously straddle or reverse. The `straddles_zero_or_reversed` condition at `analyze_c3.py:140` is `bool(straddles_zero or mean_delta < 0)` — note this OR condition means an emotion counts as "failing monotonicity" if **either** (a) CI straddles 0 **or** (b) mean_delta < 0 (negative, even if CI doesn't straddle 0). This is a correctly-specified criterion per the EXPERIMENT_PLAN.md claim language ("straddles 0 OR reverses sign").
- The exactly-at-threshold result (3/6) means the claim passes by exactly 1 emotion. No overclaim.

### D. Dead Code Detection: PASS
- `analyze_c3.py:119-145` — C3b logic is in `main()` and results written to `result["C3b"]`. All code is live.

### E. Scope Assessment: WARN
- The claim is about GSM8K specifically. The analysis runs on all 3 tasks (gsm8k, socialiqa, medqa), but the PASS criterion counts only GSM8K emotions (`if tname == "gsm8k"` at line 144). This is correctly scoped.
- WARN: C3b passes exactly at threshold (3/6). The 95% CI is based on 500 items, but for per-item paired deltas where each item is binary (0/1), the effective per-emotion CI width may be wider than typical. No sensitivity analysis on the 3rd qualifying emotion (the one that barely makes the threshold) was reported. This is a statistical power concern — the claim passes by exactly the minimum required margin.

### F. Evaluation Type: real_gt (inherited from M2)

## Action Items
1. Report the specific 3 emotions (out of 6) that qualify with either CI-straddle or sign-reversal, and their exact CI bounds. This provides transparency about whether the 3rd qualifying emotion is a marginal case.
2. Consider showing that the non-qualifying 3 emotions have clear positive (CI entirely above 0) deltas, not just "didn't qualify."
3. WARN does not block Stage 2 — C3b advances with an integrity warning.
