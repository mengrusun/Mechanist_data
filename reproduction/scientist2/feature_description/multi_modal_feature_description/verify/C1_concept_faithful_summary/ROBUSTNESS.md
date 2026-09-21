## C1: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C1: supported
- Main-experiment integrity: warn
- warn_source: experiment (scope overstatement — "every component c in a trained vision model" but only ResNet-50 tested; P3 cross-model SKIPPED; P1b docstring mismatch re: "matched-control" vs. top-1/top-2 gap)
- Variants: none  (Stage 2 skipped — C1 not selected as top-K by Phase 3 step 0 importance pick; MAX_VERIFY_CLAIMS=1 cap, C2 was the more central claim selected)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology for C1 (evaluation honesty and, N/A on mechanism intervention) was audited at Phase 2 and found trustworthy with a scope-language caveat. C1's main-experiment verdict of "supported" stands on ResNet-50 evidence (P1a top-1 purity=0.898 at k=16; P1b layer4 Δ_sep=0.020 at p≈0; P1c plateau at k∈{1,4,16}). No swap stress test was attempted this pass because C2 was selected as the higher-importance Stage-2 pick under the MAX_VERIFY_CLAIMS=1 cap. The main experiment's own verdict on C1 stands as-is, with the caveat that it has not been shown robust across method / dataset / model swaps. P3 cross-model universality (M12) remains unverified by user directive. To upgrade to full verification without re-auditing:
- stage2_skip_reason: max_verify_claims_cap → `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit is reused via RESUME)
