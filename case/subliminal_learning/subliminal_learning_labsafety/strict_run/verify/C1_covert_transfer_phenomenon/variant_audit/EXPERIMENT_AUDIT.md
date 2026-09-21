# Variant Experiment Audit Report — Claim C1

**Date**: 2026-07-18
**Auditor**: external LLM reviewer (gpt-5.4)
**Claim**: C1 — >=3pp drop in QA_I accuracy vs BOTH Ctrl-A and Ctrl-B per seed across {42,123,2026}
**Variant**: method-swap-rule-based-scorer
**Variant directory**: verify/C1_covert_transfer_phenomenon/variants/method-swap-rule-based-scorer/

## Overall Verdict: PASS

All 6 methodology checks pass or return only non-blocking WARN. The variant is methodologically sound.

## Integrity Status: pass

## Checks

### A. Ground Truth Provenance: PASS
Gold loaded from authoritative QA_I parquet `Correct Answer` column (dataset GT, not model-produced). `load_gold_from_parquet()` explicitly reads from the fixed parquet path. **PASS.**

### B. Score Normalization: PASS
Accuracy computed as `n_correct / 133` with fixed denominator. No division by model's own max, mean, or self-normalized score. Gap metrics are arithmetic differences of these accuracies. **PASS.**

### C. Result File Existence: PASS
`result.json` exists and the cited values are internally consistent with the rule-based scoring setup: all_seeds_pass=True, per-seed entries for {42,123,2026} present, gap values consistent with reported accuracies. **PASS.**

### D. Dead Code: WARN (non-blocking)
Key scoring logic described and verified in code review round 2 (APPROVED). No evidence of unused eval functions from the provided context alone. Not a blocking issue — the full execution path was demonstrated by the successful run. **WARN (non-blocking).**

### E. Scope Assessment: PASS
Variant matches claim scope exactly: 3 pre-registered seeds {42, 123, 2026}, 133 items per arm (n==133 asserted), 3 arms (Ctrl-A, treated, Ctrl-B). Gap criterion identical to main experiment (>=3pp per seed per both controls). **PASS.**

### F. Evaluation Type: real_gt
Gold from dataset-provided parquet labels. Deterministic rule-based scorer (no LLM). **real_gt.**

## Action Items
None blocking. Non-blocking WARN on dead-code check is acceptable — the run log confirmed all evaluation codepaths executed.
