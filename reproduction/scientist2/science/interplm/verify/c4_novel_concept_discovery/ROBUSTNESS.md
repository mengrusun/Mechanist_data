## c4: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  ->  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on c4: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (synonym-check criterion non-discriminative; both real and control arms yield 0%; reduced scope 100/500 features)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology audited at Phase 2 (WARN). C4 not selected as top-K claim; admitted but deferred. Main verdict: not-supported (0.0% novel; criterion-design null). A model swap would not resolve the criterion-design issue — the synonym-check gate needs a redesign regardless of model scale. Priority fix: stricter novelty prompt or embedding-distance-based test. To swap-test after fixing criterion:
- /auto-verify c4 — resume: true
