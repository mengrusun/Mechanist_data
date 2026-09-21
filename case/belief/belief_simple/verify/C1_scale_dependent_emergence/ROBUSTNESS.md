## C1: audit-only  (main-experiment integrity = PASS, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C1: supported
- Main-experiment integrity: pass
- warn_source: null
- Variants: none  (Stage 2 skipped — MAX_VERIFY_CLAIMS=1 cap; C2 selected as top-K pick by importance)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology (evaluation honesty) was audited at Phase 2 and found trustworthy (PASS). C1's behavioral dissociation result (personal 0.852 vs attributed 0.457 at pythia-410m, +39pp separation) is supported. No swap stress test was attempted this pass — C1 was admitted by Stage 1 but not selected as the top-K picked claim (C2 was selected as more load-bearing for the paper's mechanistic thesis). To upgrade to full verification without re-auditing: `/auto-verify C1 -- resume: true` (single-claim mode; Phase 2 audit is reused via RESUME).
