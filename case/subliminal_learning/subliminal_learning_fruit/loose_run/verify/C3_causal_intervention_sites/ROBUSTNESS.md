## C3: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C3: not-supported
- Main-experiment integrity: warn
- warn_source: experiment+mechanism (single-seed intervention; incomplete dose grid; σ_l calibrated but no negative amplification; amplify_x3 null)
- Variants: none  (Stage 2 skipped — MAX_VERIFY_CLAIMS=1 cap; C1 was picked as top-importance claim)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology was audited and found methodologically honest but with scope limitations (Phase 2 WARN): single-seed intervention (seed 42 only; plan required ≥3), amplify_x3 null, no negative-dose amplification. The partial verdict (sign correct, magnitude 10× below bar, specificity fails) is consistently documented. No swap stress test was attempted this pass because MAX_VERIFY_CLAIMS=1 and C1 was selected as the top-importance audit target. C3's main-experiment NOT-SUPPORTED (partial) verdict stands as-is, with WARN caveat on single-seed under-power. To upgrade to full verification without re-auditing Phase 2: `/auto-verify C3 -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME, method-swap variant runs a second seed per the task.md suggestion).
