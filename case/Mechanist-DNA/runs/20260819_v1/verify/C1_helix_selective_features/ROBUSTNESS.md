## C1: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  ⚪ INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C1: supported (qualified)
- Main-experiment integrity: warn
- warn_source: experiment
- Variants: none  (Stage 2 skipped — MAX_VERIFY_CLAIMS=1 selected C2 as the top-1 by importance)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C1's evaluation methodology (real DSSP GT, tie-corrected AUROC, gene-blocked null, matched controls) was audited and found trustworthy at Phase 2, with a WARN only because the pre-registered codon-level AUROC>=0.70 was not met (max 0.633) and the documented specificity-fallback was used — honestly reported and qualified. No swap stress test was attempted this pass (cap=1 selected C2). To swap-test C1 later: `/auto-verify C1 — resume: true`.
