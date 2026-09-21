## C2: audit-only (main-experiment integrity = WARN, swap stress test skipped) — INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C2: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (scope: position dissociation untestable due to ceiling effects; refusal crossover unmeasurable at 98.7% baseline refusal)
- Variants: none (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C2's main-experiment audit passed with a WARN (scope: both positions at AUROC ceiling, making the crossover untestable; refusal crossover NaN). C2 was admitted to Stage 2 but was not the top-K pick (MAX_VERIFY_CLAIMS=1 cap; C1 was picked as more scientifically central). C2's not-supported verdict stands as-is. A model-swap to Qwen2-Instruct-7B (which may have lower baseline refusal) could test whether a position crossover emerges when the refusal signal is measurable. To upgrade: `/auto-verify C2 — resume: true` (Phase 2 audit reused; Stage 2 runs for C2 only).
