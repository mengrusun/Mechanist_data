## C3c: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C3c: not-supported
- Main-experiment integrity: warn
- warn_source: experiment
- warn_detail: Load-bearing cell (probe_low, verbal_low) n=4/2000. Test under-powered — design-of-test limitation due to extreme verbalized-c skew (96% >= 95). Honestly reported as [suspected under-power]. Not fabrication.
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology was audited at Phase 2 and found WARN (not FAIL) — C3c was admitted. No swap stress test attempted this pass — C3c was not picked (MAX_VERIFY_CLAIMS=1 cap; C3a picked as load-bearing claim). C3c's main-experiment verdict (not-supported [suspected under-power]) stands. Note: under-power means the null is not informative — the test is design-limited, not a confirmed falsification of the sub-claim. To upgrade:
- `/auto-verify C3c — resume: true` (single-claim mode; Phase 2 audit reused via RESUME)
