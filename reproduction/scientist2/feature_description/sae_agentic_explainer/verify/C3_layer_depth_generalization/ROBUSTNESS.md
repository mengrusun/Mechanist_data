## C3: audit-only (main-experiment integrity = WARN, swap stress test skipped) — INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C3: not-supported (no per-layer CI excludes zero at n=14-15/layer; under-powered)
- Main-experiment integrity: warn
- warn_source: experiment (Bonferroni correction described but not coded in aggregate script; per-layer n too small; L20 gen_acc all-zero differences)
- Variants: none (Stage 2 skipped — max_verify_claims_cap; C4 selected as top-1)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology for C3 has a soft issue (Bonferroni framing not explicitly implemented in code; per-layer n=14-15 is very small). The not-supported verdict is likely correct (no layer shows consistent gains) but the sample size makes it impossible to confirm directional effects at individual depths. A model-swap variant for C3 with the same small n would be uninformative — the priority fix for C3 is more features per layer (n=100), not a model swap. C3 is therefore the lowest-priority claim for swap-variant testing.

To upgrade: `/auto-verify C3 -- resume: true` (single-claim mode; best combined with a larger n_features_per_layer rerun of M1 in iteration)
