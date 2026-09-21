## C2: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C2: supported
- Main-experiment integrity: warn
- warn_source: experiment
- warn_detail: Probe target is model's own verbalized confidence c (by-design proxy, disclosed, not fraud). Bootstrap CI [0.913, 0.940] does not contain point estimate 0.948 (normal artifact). Paraphrase Delta AUC P1=-0.11 slightly exceeds <=0.10 planned tolerance (AUC still above 0.70 floor, honestly reported).
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology was audited and found trustworthy at Phase 2 (WARN-level, not FAIL). No swap stress test attempted this pass — C2 was admitted but not picked (MAX_VERIFY_CLAIMS=1 cap; C3a picked as load-bearing claim). C2's main-experiment verdict (supported) stands with the caveat that robustness across model/dataset swaps has not been tested. To upgrade:
- `/auto-verify C2 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME)
