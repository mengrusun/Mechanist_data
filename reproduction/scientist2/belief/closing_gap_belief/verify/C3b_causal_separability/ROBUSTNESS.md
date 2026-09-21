## C3b: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C3b: supported
- Main-experiment integrity: warn
- warn_source: experiment+mechanism
- warn_detail: Experiment WARN — primary metric is internal probe readout (pre-registered, weaker than behavioral GT); sparse alpha grid (3 values); single random-direction control (n=1). Mechanism WARN — sparse alpha sweep (< 3 orders of magnitude, < 5 grid points); n_random=1 << 30 recommended.
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology was audited at Phase 2 and found WARN (not FAIL) — C3b was admitted. No swap stress test attempted this pass — C3b was not picked (MAX_VERIFY_CLAIMS=1 cap; C3a picked as load-bearing claim). C3b's main-experiment verdict (supported, internal-readout PRIMARY) stands, with methodology WARNs. To upgrade:
- `/auto-verify C3b — resume: true` (single-claim mode; Phase 2 audit reused via RESUME)
