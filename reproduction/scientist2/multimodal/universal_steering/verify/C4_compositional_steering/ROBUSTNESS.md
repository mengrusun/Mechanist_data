## C4: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C4: not-supported
- Main-experiment integrity: warn
- warn_source: experiment+mechanism (experiment: measurement-ceiling at rubric~5.0 makes test uninformative; mechanism: missing matched-random-direction control required by plan)
- Variants: none  (Stage 2 skipped — MAX_VERIFY_CLAIMS=1 cap; C5 selected as top-1 by importance)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C4's not-supported verdict is due to a measurement-ceiling failure (single-vector rubric saturates at ~5.0), not a composition failure. The strict predicate (sum > v2_only on r1 AND sum > v1_only on r2) is unachievable when single vectors already maximize the rubric. The mechanism audit flagged the missing random-direction control (required by plan). The not-supported verdict is honest but uninformative about the compositionality claim. To test the claim properly: use harder prompts that don't saturate single vectors, and add the missing random-direction control. `/auto-verify C4 — resume: true` could test a model swap to examine if a different model shows non-ceiling behavior.
