## C1: audit-only (main-experiment integrity = WARN, swap stress test skipped) — INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C1: partial-support (supported vs. Neuronpedia; not-supported vs. matched-backbone GPT-5-1shot)
- Main-experiment integrity: warn
- warn_source: experiment (statistical resolution: Wilcoxon n_eff=4 / 90.9% zero-differences in gen_acc; 5 probes/feature coarse metric)
- Variants: none (Stage 2 skipped — max_verify_claims_cap; C4 was selected as top-1 by importance)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology was audited and found sound with a statistical-resolution caveat (gen_acc's coarse 5-probe scale produces 90.9% zero-differences, making the Wilcoxon test's effective n=4; the bootstrap CI and significance are real but fragile). No swap stress test was attempted this pass — the main experiment's partial-support verdict on C1 stands as-is, with the caveat that it has not been shown robust across model swaps.

C1 was admitted by Phase 2 but not selected as the top-K pick by importance (C4 was chosen instead because its cross-pair generalization claim is more scientifically central and its deferred pair — GPT-OSS-20B + resid-post-aa — has never been tested).

To upgrade to full verification without re-auditing:
- `stage2_skip_reason: max_verify_claims_cap` -> `/auto-verify C1 -- resume: true` (single-claim mode; Phase 2 audit is reused via RESUME)

Suggested model-swap variant for C1 when budget allows: Qwen3-4B + transcoder-hp (already partially covered in M2 but only for predictive-accuracy; full generative-accuracy test on Qwen3-4B would be new).
