## C2: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C2: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (shortlist scope reduction — M4/M4.stab ran on top-40 of 158 components; claim predicate references "the shortlisted components" i.e. all 158; scope reduction is disclosed in EXPERIMENT_RESULTS.md)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology was audited and found trustworthy at Phase 2 (WARN). The WARN arises from the shortlist scope reduction for M4/M4.stab (top-40 of 158 components, for budget reasons). The C2 modularity verdict (not-supported) is conditioned on the top-40 sublist. No swap stress test was attempted this pass — C3 was selected as the top-K=1 pick by importance (C3's necessity/sufficiency asymmetry is the most central and surprising finding). The main experiment's own verdict on C2 (not-supported) stands as-is, with the caveat that it has not been shown robust across method/dataset/model swaps.

To upgrade to full verification without re-auditing:
- `/auto-verify C2 — resume: true` (single-claim mode; Phase 2 audit is reused via RESUME; Stages 2–3 execute for this one claim)
