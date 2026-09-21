## c5b: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  ->  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on c5b: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (scope: 3/4 features, 10/25 seqs per batch); mechanism (steering coefficient sweep spans <3 OOM, <5 grid points)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology audited at Phase 2 (WARN on both experiment and mechanism). C5b not selected as top-K claim. Main verdict: not-supported (no-steer yield highest for all 3 features; no dose-response). The mechanism audit flags the coefficient sweep as covering only 8-fold range (<3 OOM). Priority fixes: (1) broader dose range {0.1, 0.3, 1, 3, 10, 30} × sigma_f to confirm null across OOM; (2) extend to 25 seqs/batch. A model swap alone would not resolve the causal mechanism question. To swap-test:
- /auto-verify c5b — resume: true
