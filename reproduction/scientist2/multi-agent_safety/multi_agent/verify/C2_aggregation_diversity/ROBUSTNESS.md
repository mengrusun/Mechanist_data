## C2: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C2: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (under-power: 49-scenario test split; diversity test on 16/17 domains)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C2's main-experiment evaluation methodology was audited at Phase 2 and found sound with a WARN (under-power on the group-vs-single AUROC delta at N=49 test scenarios; diversity test effective on 16/17 domains). No swap stress test was attempted this pass — C2 was admitted by Phase 2 but was not selected as the top-K=1 claim (C1 was picked as the foundational claim). The main-experiment verdict of not-supported (predicate-a fails: delta=0.025 < 0.05) stands as-is. To swap-test C2: `/auto-verify C2 — resume: true` (single-claim mode; Phase 2 audit is reused via RESUME; Stages 2–3 execute for C2 with a model-swap variant).
