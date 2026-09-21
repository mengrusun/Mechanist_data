## C1: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  ->  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C1: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (sign-consistency predicate not computed — half of C1's required evidence missing)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology for C1 was audited and found trustworthy for the accuracy-threshold half of the claim (real GT, no self-normalization, result files exist). WARN because the per-item sign-consistency predicate was not computed despite being required by C1's formal statement. No swap stress test was attempted this pass — C1 was admitted by Phase 2 but was not selected as the top-K pick by importance (CM was picked instead). To upgrade to full verification:

- `stage2_skip_reason: max_verify_claims_cap` -> `/auto-verify C1 -- resume: true` (single-claim mode; Phase 2 audit is reused via RESUME)

Note: before verify, consider computing sign-consistency from existing per_item data in runs/M2/*.json to resolve the WARN and strengthen the main-experiment evidence for C1.
