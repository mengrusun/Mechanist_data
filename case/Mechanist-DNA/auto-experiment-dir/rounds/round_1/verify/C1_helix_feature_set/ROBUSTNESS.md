## C1: audit-only (main-experiment integrity = WARN, swap stress test skipped)  →  ⚪ INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C1: supported (established, set-level)
- Main-experiment integrity: warn
- warn_source: experiment (scope overclaim — see EXPERIMENT_AUDIT.md: degenerate seed_jaccard=0.0
  undisclosed, post-hoc set-level pass-criterion substitution, missing multi-organism robustness axis)
- Variants: none — Stage 2 skipped (admitted by Phase 2 but not the top-K picked by importance;
  MAX_VERIFY_CLAIMS=1 and C2 was judged the more central claim to swap-test this pass)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C1's main-experiment methodology was audited and found trustworthy on GT provenance,
score normalization, result-file existence, and evaluation type (all PASS), but scope-level overclaim
issues were found — see `verify/C1_helix_feature_set/main_experiment_audit/EXPERIMENT_AUDIT.md`. No
swap stress test was attempted this pass; the main experiment's own verdict on C1 stands as-is, with
the caveat that it has not been shown robust across method/dataset/model swaps. To upgrade to full
verification without re-auditing: `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit
reused via RESUME; Stages 2–3 execute for this one claim).
