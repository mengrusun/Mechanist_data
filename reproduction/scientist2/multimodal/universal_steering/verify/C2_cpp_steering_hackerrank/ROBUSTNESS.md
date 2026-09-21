## C2: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C2: not-supported [provisional — suspected_under_power]
- Main-experiment integrity: warn
- warn_source: experiment (suspected_under_power: n_held=10 vs plan's 30; stat differences at n=10 with std≈0.49 are meaningless)
- Variants: none  (Stage 2 skipped — MAX_VERIFY_CLAIMS=1 cap; C5 selected as top-1 by importance)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C2's main-experiment methodology is correct (real_gt test cases, no silent empty-output acceptance, no timeout leakage, clean dev/held split). The negative verdict is provisional due to suspected under-power (10 held-out problems × 2 seeds vs plan's 30). Stage 2 swap stress test not performed. To investigate the negative result further: `/auto-verify C2 — resume: true` with a dataset swap to LeetCode (which has a larger problem pool) or a model swap to examine if the language-switch failure is Llama-3.1-8B-Instruct specific.
