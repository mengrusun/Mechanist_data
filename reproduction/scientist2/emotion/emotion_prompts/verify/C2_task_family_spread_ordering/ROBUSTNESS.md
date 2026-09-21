## C2: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  ->  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C2: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (cross-task spread comparison conflates CoT vs MCQ log-likelihood evaluation modes; confound acknowledged in EXPERIMENT_RESULTS but not controlled)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology for C2 was audited and found trustworthy for GT provenance and score normalization (real GT for all three tasks, no self-normalization). WARN because the cross-task spread comparison is between structurally different evaluation modes (CoT for GSM8K, MCQ log-likelihood for SocialIQA/MedQA), which is a known confound documented in EXPERIMENT_RESULTS. C2 was admitted by Phase 2 but was not selected as the top-K pick — CM was picked instead. To upgrade:

- `/auto-verify C2 -- resume: true` (single-claim mode; Phase 2 audit reused)
