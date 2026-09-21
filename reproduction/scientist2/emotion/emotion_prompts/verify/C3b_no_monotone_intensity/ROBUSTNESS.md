## C3b: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  ->  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C3b: supported
- Main-experiment integrity: warn
- warn_source: experiment (passes at exactly threshold 3/6; no per-emotion CI bounds reported in EXPERIMENT_RESULTS; marginality of 3rd qualifying emotion not assessed)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology for C3b was audited and found trustworthy for GT provenance and normalization. WARN because C3b passes at exactly the minimum threshold (3/6 emotions) with no per-emotion CI details reported, making it unclear whether the 3rd qualifying emotion is marginal. C3b was admitted by Phase 2 but was not selected as the top-K pick. To upgrade:

- `/auto-verify C3b -- resume: true` (single-claim mode; Phase 2 audit reused)

Pre-verification: recommend computing exact per-emotion CI bounds from reports/M4_c3_analysis.json to assess whether the 3rd qualifying emotion's CI straddling is robust.
