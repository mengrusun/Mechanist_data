## C4: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C4: not-supported
- Main-experiment integrity: warn
- warn_source: experiment + mechanism  (experiment: synthetic_proxy GT, min-diagonal=0 degenerate ratio; mechanism: inherits C3's missing L=16 random-direction control; no B2 random-direction w_effects comparison at ell_V*)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology for C4 was audited and found admissible (WARN) at Phase 2. The NOT SUPPORTED verdict is correctly diagnosed as a consequence of layer under-power at ell_V* (sigma_proj=0.008–0.038 → alpha×sigma_proj injection too small), not a selectivity failure per se. The permutation test (p=0.84) and indeterminate c4_ratio (min_diag=0 for V=G) are honest and correctly reported. The C4 matrix at L=16 was not evaluated and is deferred to the iteration loop. No swap stress test was attempted for C4 this pass (MAX_VERIFY_CLAIMS cap assigned the single slot to C3). Main experiment's NOT SUPPORTED verdict stands for the ell_V*-layer evaluation. To upgrade to full verification without re-auditing:
- `/auto-verify C4 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME)
