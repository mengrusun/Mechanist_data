## C1: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C1: not-supported
- Main-experiment integrity: warn
- warn_source: experiment (undisclosed minimality sampling — 20/158 components sampled in the minimality sweep; approximation not surfaced in per-claim verdict text)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology (evaluation) was audited and found trustworthy at Phase 2 (WARN). The WARN arises from the undisclosed minimality sampling approximation (20 of 158 components). No swap stress test was attempted this pass — C3 was selected as the top-K=1 pick by importance (C1's sparsity result is a boundary case at the 15% cap, but less scientifically central than C3's necessity/sufficiency asymmetry). The main experiment's own verdict on C1 (not-supported) stands as-is, with the caveat that it has not been shown robust across method/dataset/model swaps.

To upgrade to full verification without re-auditing:
- `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit is reused via RESUME; Stages 2–3 execute for this one claim)
