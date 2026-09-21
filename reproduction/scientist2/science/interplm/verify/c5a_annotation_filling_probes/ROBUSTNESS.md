## c5a: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  ->  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on c5a: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (dead code in per_concept_pr_auc; 30/50 concepts; SGDClassifier substitution; 25k/387k test residue subsample)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology audited at Phase 2 (WARN). C5a not selected as top-K claim; admitted but deferred. Main verdict: not-supported (p=0.191, means equal at 0.591). The null result is more likely due to SGDClassifier under-fitting (max_iter=30) than a true null — re-running with LogisticRegression(max_iter=200) and full 387k residue test set is the recommended first fix before a model-swap verify. To swap-test:
- /auto-verify c5a — resume: true
