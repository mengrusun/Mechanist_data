## C2: audit-only (main-experiment integrity = PASS, swap stress test skipped)  →  INTEGRITY_ONLY

- verdict: INTEGRITY_ONLY
- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- main_experiment_verdict: supported (0 flagged rows on 2611-row scrubbed training set)
- main_experiment_integrity: pass
- warn_source: null
- variants: none (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C2's main experiment methodology (gpt-5.4 rescan oracle, independent cache file, threshold predicate on training set actually used) was audited and found trustworthy at Phase 2. No swap stress test was attempted this pass — C2 was admitted by Phase 2 but not selected as the top-K importance pick (C1 was selected as the single highest-priority claim under MAX_VERIFY_CLAIMS=1 cap). C2's own supported verdict (0 flagged rows on scrubbed set) stands as-is, with the caveat that it has not been shown robust across method / dataset / model swaps.

To upgrade to full verification without re-auditing:
- /auto-verify C2 — resume: true  (single-claim mode; Phase 2 audit reused via RESUME; Stages 2-3 execute)

Potential model-axis variant for C2 (if swap-tested later): swap the rescan judge from gpt-5.4 to an alternative judge (e.g., a different API model or an open-source safety classifier) to test whether the 0-flagged verdict is robust to judge model choice. This would directly stress-test the subliminality claim's precondition.
