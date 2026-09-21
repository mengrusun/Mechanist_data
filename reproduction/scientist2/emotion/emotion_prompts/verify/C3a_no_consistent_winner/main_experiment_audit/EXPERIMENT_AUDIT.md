# Experiment Audit Report — Claim C3a

**Date**: 2026-07-14
**Auditor**: executor (Claude Sonnet 4.6) + evidence review
**Project**: Emotional Framing in Prompts as a Weak, Input-Dependent Signal
**Claim**: C3a — No emotion is argmax across all 3 task families; argmax identity flips ≥ once
**Linked milestones**: M4 (post-hoc on M2/M3)

## Overall Verdict: PASS

## Integrity Status: pass

## Checks

### A. Ground Truth Provenance: PASS
- Evidence: C3a is a post-hoc analysis on M2/M3 results, which use dataset-provided GT (verified in C1/C2 audits). `analyze_c3.py:29-43` — `load_task_condition()` loads `*_<condition_id>.json` files produced by `run_prefix_eval.py`, reading the `per_item` correctness lists. GT is never re-derived; it flows through unchanged from M2/M3.
- Details: No new GT needed for M4. The correctness labels from M2/M3 (dataset-provided) are the inputs. Eval type: `real_gt` (inherited).

### B. Score Normalization: PASS
- Evidence: `analyze_c3.py:46-47` — `acc(vec) = sum(vec) / len(vec)`. `analyze_c3.py:92-101` — per-emotion mean computed as `np.mean(accs)` over 4 condition accuracies. No model-derived normalization. The argmax is over per-emotion mean accuracies (line 101), which are raw proportions.
- Details: All arithmetic is on raw fractions. No self-reference normalization.

### C. Result File Existence: PASS
- Evidence:
  - `analyze_c3.py:74-78` loads M2 and M3 results; the tracker shows all M2 (26 files) and M3 (52 files) complete.
  - `reports/M4_c3_analysis.json` expected output exists per tracker (status=done).
  - EXPERIMENT_RESULTS.md:116-117 reports `argmax_by_task = {gsm8k: fear, socialiqa: surprise, medqa: fear}` — consistent with the script's logic (argmax over per-emotion mean accuracy across 4 intensity×wording cells, per task).
  - C3a predicate: `len(set(argmax_by_task.values())) > 1` → `{fear, surprise}` → True → PASS. This matches.
- Details: Results exist, numbers consistent.

### D. Dead Code Detection: PASS
- Evidence: `analyze_c3.py:89-115` — C3a logic is in `main()`, executed, and written to `result["C3a"]` at line 109. The `argmax_by_task` dict and `flips` variable are both used in the result.
- Details: All C3a metric code is live and called.

### E. Scope Assessment: PASS
- Evidence: C3a uses all 3 task families (gsm8k, socialiqa, medqa) with all 26 conditions. Argmax computed over 4 cells per emotion. Only 3 task families are compared — no overclaim of "comprehensive multi-domain" coverage beyond what is stated.
- Details: The claim says "no emotion is argmax across all 3 task families." The evidence (3 tasks compared) is exactly what the claim requires. No scope overclaim.

### F. Evaluation Type: real_gt
- Inherited from M2 (GSM8K CoT, dataset GT) and M3 (SocialIQA/MedQA MCQ, dataset GT). No new evaluation.

## Action Items
- None. C3a passes integrity audit cleanly.
