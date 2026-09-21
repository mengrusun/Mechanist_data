## C1b: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C1b: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (numeric citation mismatch between on-disk JSON and EXPERIMENT_RESULTS.md for all 3 levels; fine-level n=0 in JSON vs 0.817 cited in prose; fine-level predicate unverifiable)
- Variants: none  (Stage 2 skipped — max_verify_claims_cap; C2a selected as top-1 by importance)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology audited at Phase 2 with WARN. Two issues: (1) numeric citation mismatch between the on-disk level_separation.json and EXPERIMENT_RESULTS.md; (2) fine-level triplets yield n=0 in the on-disk result, making the monotonicity claim at fine level unverifiable. C1b was not selected for Stage 2 — it has integrity caveats AND the monotonicity finding is already conditional. To upgrade: `/auto-verify C1b -- resume: true` (single-claim mode; Phase 2 audit reused). Recommend also reconciling the numeric discrepancy between on-disk JSON and prose before re-verify.
