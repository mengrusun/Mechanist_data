## C2: audit-only (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C2: not-supported (split-half cos 0.588 < 0.70 floor at n=25 for uncertainty; other 3 behaviours untestable at plan scale)
- Main-experiment integrity: WARN (exp=WARN — ratio_to_large_pool normalized by model-derived reference delta; mech=WARN — borrows M3's under-validated α_op with no random-direction control)
- warn_source: experiment + mechanism
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology was audited. The evaluation WARN reflects that (a) ratio_to_large_pool normalizes by a model-output-derived denominator and (b) the mechanism WARN reflects that C2's steering-effect ratio measurement uses an α_op borrowed from M3 where no random-direction control was run and the plateau is not cleanly established. The main-experiment verdict on C2 (not-supported — split-half stability below floor at the only testable behaviour, others untestable) stands with these caveats. No swap stress test this pass. To upgrade:

`/auto-verify C2 — resume: true` (single-claim mode; Phase 2 audit reused; Stages 2-3 execute)

Note: A model-swap for C2 would test whether direction stability (split-half cosine) holds in a different R1-Distill-Qwen model — this requires re-annotating an auxiliary corpus for the new model, which is a significant effort. Recommend fixing the corpus size issue first (larger contrastive corpus for all 4 behaviours) before stress-testing C2 cross-model.
