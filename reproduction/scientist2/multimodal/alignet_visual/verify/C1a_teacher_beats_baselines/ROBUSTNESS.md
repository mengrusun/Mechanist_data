## C1a: audit-only  (main-experiment integrity = PASS, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C1a: supported
- Main-experiment integrity: pass
- warn_source: null
- Variants: none  (Stage 2 skipped — max_verify_claims_cap; C2a selected as the top-1 claim by importance)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology (teacher head fitting + THINGS held-out evaluation) was audited and found trustworthy at Phase 2 — PASS with no caveats. C1a was not selected as one of the top-K=1 claims for Stage 2 because C2a has larger effect size, cleaner audit, and greater scientific centrality. No swap stress test was attempted this pass. To upgrade: `/auto-verify C1a -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME).
