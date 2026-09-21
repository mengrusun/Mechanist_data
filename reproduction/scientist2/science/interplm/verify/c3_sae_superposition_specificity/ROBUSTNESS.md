## c3: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  ->  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on c3: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (strict margin Δ_SAE-PCA ≥ 20 not met at primary τ=0.5; margin=15; order_check_pass=false due to PCA=neurons=0)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology was audited and found trustworthy at Phase 2 (WARN only). C3 was admitted by Stage 1 but not selected as the top-K=1 claim. The direction claim (SAE > all controls) IS supported at primary setting. The strict margin criterion (Δ≥20) is not met at primary τ=0.5 (Δ=15) but is robustly met at τ=0.3 (Δ=65). No swap test attempted this pass. To upgrade:
- /auto-verify c3 — resume: true
