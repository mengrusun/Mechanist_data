## C2b: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C2b: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (fine-level Spearman has 0 usable pairs under THINGS categorical stratification; per-level predicate at fine is unverifiable)
- Variants: none  (Stage 2 skipped — max_verify_claims_cap; C2a selected as top-1)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C2b is a per-level decomposition of C2a. The coarse and mid levels both show strong positive Δρ (verified on-disk), but fine level yields 0 Spearman-usable pairs under the THINGS categorical metadata stratification — making the "all three levels" predicate unverifiable. To upgrade: `/auto-verify C2b -- resume: true` (single-claim mode; also consider resolving fine-level bucket construction to yield >30 pairs before re-verify).
