## C4b: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C4b: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (dataset substitution — BREEDS entity13/living17/non-living26/entity30 + ImageNet-A not tested; researcher-constructed BREEDS super13/super26 + in-distribution slices used; 3 of 5 substitute splits not genuine OOD)
- Variants: none  (Stage 2 skipped — max_verify_claims_cap; C2a selected as top-1)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C4b's main experiment carries a dataset-substitution WARN — the specific BREEDS benchmarks and ImageNet-A named in the claim were never tested. The BREEDS super13/super26 results are encouraging (large positive gains), but these are researcher-constructed splits not equivalent to the standard BREEDS benchmarks. The current verdict (conditional) is reasonable but incomplete. To upgrade: `/auto-verify C4b -- resume: true` after downloading the standard BREEDS panel and ImageNet-A, or rerunning on the standard splits.
