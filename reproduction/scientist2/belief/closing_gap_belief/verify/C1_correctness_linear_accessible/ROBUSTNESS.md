## C1: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C1: supported
- Main-experiment integrity: warn
- warn_source: experiment
- warn_detail: Bootstrap CI [0.818, 0.835] does not contain point estimate 0.840 (normal artifact); bootstrap n=200 (planned 1000, disclosed). No FAIL criteria triggered.
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology was audited and found trustworthy at Phase 2 (WARN-level, not FAIL). No swap stress test was attempted this pass — C1 was admitted by Phase 2 but was not picked as the top-K claim by importance (MAX_VERIFY_CLAIMS=1 cap; C3a was picked as the load-bearing claim). C1's main-experiment verdict (supported) stands, with the caveat that it has not been shown robust across method/dataset/model swaps. To upgrade to full verification:
- `stage2_skip_reason: max_verify_claims_cap` → `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME)
