## C2: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  ->  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C2: not-supported  (Δ_ablate_mean=-0.015pp, fails >=5pp gate; Δ_random indistinguishable; carrier direction delocalized across many DiT blocks)
- Main-experiment integrity: warn
- warn_source: experiment+mechanism  (Exp Check C/E: scope reduction 40->12 runs, dose-response single-point; Mech Check A: sweep cardinality gap — amplify_x3 only, Spearman-rho check not computable)
- Combined Phase 2 verdict: WARN  (max_severity(exp=warn, mech=warn) = warn)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason; C2 also not picked by Phase 3 step 0 importance judgment — ranked below C1, which is the headline claim)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C2's main-experiment methodology is audited and found trustworthy at WARN level. The forward-hook steering on transformer_blocks[50] is correctly implemented (σ_proj-calibrated, matched-random specificity control present, fluency logged). The WARN findings are: (1) scope reduction to 4 seeds × 3 interventions (planned 8×5), reducing the dose-response check to a single amplification point; (2) the resulting Spearman-rho monotonicity test cannot be computed from one point. The negative result (delocalized carrier) is scientifically sound and correctly reported as not-supported [provisional — under-power]. The swap stress test was not attempted: C2 was not the top-K pick under the MAX_VERIFY_CLAIMS=1 cap (C1 is the higher-importance headline claim), and additionally the GPU budget has been exhausted.

To upgrade to full verification without re-auditing:
- stage2_skip_reason: max_verify_claims_cap -> /auto-verify C2 -- resume: true  (single-claim mode; Phase 2 audit reused via RESUME)
- Recommended prerequisite: first complete the planned M2 grid (remaining 28 runs: seeds 201, 203, 204, 206 × interventions ablate, amplify_x2, amplify_x3, amplify_x4, random_ablate) to establish whether the not-supported verdict holds at full power.
