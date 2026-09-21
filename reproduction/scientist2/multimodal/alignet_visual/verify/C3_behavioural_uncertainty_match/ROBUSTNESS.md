## C3: audit-only  (main-experiment integrity = PASS, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C3: supported
- Main-experiment integrity: pass
- warn_source: null
- Variants: none  (Stage 2 skipped — max_verify_claims_cap; C2a selected as top-1)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C3 (behavioural + uncertainty match) audited at Phase 2 — PASS with no caveats. All 3 predicates (choice agreement, uncertainty Spearman, RSA) met in predicted direction; direct two-worker human behavioral data used (corrected from Round 1). C3 was not selected for Stage 2 — it is a consequence of C2a and C2c (behavioral output of applying τ_aligned), making C2a more scientifically central. To upgrade: `/auto-verify C3 -- resume: true`. A model-swap variant would check whether the same behavioural alignment holds for a different student backbone.
