## C3: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  ⚪ INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C3: supported
- Main-experiment integrity: warn
- warn_source: experiment
- Variants: none  (Stage 2 skipped — MAX_VERIFY_CLAIMS=1 selected C2 as the top-1 by importance)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C3's dose-response methodology (7-point sweep, capability metrics, frozen alpha* on validation) passed Phase 2 (experiment WARN because the single-peaked shape is established on the dev sweep while held-out reproduces only the rising sub-grid; mechanism PASS). No swap stress test this pass (cap=1 selected C2). To swap-test C3 later: `/auto-verify C3 — resume: true`.
