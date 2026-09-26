## C2: audit-only  (main-experiment integrity = PASS, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C2: not-supported (refuted under LoRA-SVD-derived additive-steering family, both single-block and 9-block-window configurations)
- Main-experiment integrity: pass
- warn_source: null
- Variants: none  (Stage 2 skipped — max_verify_claims_cap: C1 picked as top-1 by importance)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology (evaluation and mechanism intervention rigor) was audited and found trustworthy at Phase 2. No swap stress test was attempted this pass — the MAX_VERIFY_CLAIMS=1 cap selected C1 (the load-bearing positive result) as the higher-priority pick. C2's "not-supported" verdict stands as-is: the LoRA-SVD-derived top-1 singular direction at `attn.to_out.0` does not causally lift P(banana) under additive residual-stream steering in either single-block or 9-block-window configurations. To upgrade to full verification without re-auditing: `/auto-verify C2 — resume: true` (single-claim mode; Phase 2 audit is reused via RESUME).
