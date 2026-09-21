## C1: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  ->  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C1: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (Check E: claim predicate requires matched-control specificity A>D which fails; A=0.738 ≈ D=0.755)
- Variants: none  (Stage 2 skipped — MAX_VERIFY_CLAIMS=1 cap; C2 picked as higher-priority claim)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C1's main experiment methodology (forward-only hidden-state extraction in M1, activation patching in M2) was audited and found sound on GT provenance, normalization, result-file existence, and dead-code checks. The WARN is specific to Check E: the frozen C1 claim requires both A>C (satisfied, +0.28 semantic cosine) AND A>D (matched-control specificity, which fails at A=0.738 ≈ D=0.755). No swap stress test was attempted this pass — C2 was selected as the higher-priority Stage 2 entry under the MAX_VERIFY_CLAIMS=1 cap. The main experiment's own verdict on C1 stands as 'partial' (binary: not-supported), unverified under model/dataset/method swaps. To upgrade to full verification without re-auditing: /auto-verify C1 -- resume: true (single-claim mode; Phase 2 audit is reused via RESUME; only Stages 2-3 execute for C1).
