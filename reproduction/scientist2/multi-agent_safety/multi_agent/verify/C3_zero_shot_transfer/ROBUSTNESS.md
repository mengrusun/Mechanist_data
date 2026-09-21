## C3: audit-only  (main-experiment integrity = PASS, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C3: supported
- Main-experiment integrity: pass
- warn_source: null
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C3's main-experiment evaluation methodology was audited at Phase 2 and found clean (all checks PASS). No swap stress test was attempted this pass — C3 was admitted by Phase 2 but was not selected as the top-K=1 claim (C1 was picked as the foundational claim). The main-experiment verdict of supported (5/7 transfer families with AUROC >= 0.65, best=0.913) stands as-is. To swap-test C3: `/auto-verify C3 — resume: true` (single-claim mode; Phase 2 audit is reused via RESUME; Stages 2–3 execute for C3 with a model-swap variant testing whether the transfer story holds when the probe host changes to a different model).
