## C1: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C1: supported (political-stance component; overall 'partial' per EXPERIMENT_RESULTS.md)
- Main-experiment integrity: warn
- warn_source: experiment+mechanism (experiment: alpha_star labelling bug cosmetic + scope '3 scenarios' overclaim; mechanism: degenerate block selection for trivially-separable concepts + alpha selection on same 50-prompt held-out set)
- Variants: none  (Stage 2 skipped — MAX_VERIFY_CLAIMS=1 cap; C5 selected as top-1 by importance)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C1's main-experiment methodology was audited. The political-stance finding is the strongest sub-component (signed monotone α=-3→3.08, α=+3→4.02; random control only 0.30/0.12). The main-experiment integrity received WARN (not FAIL) — the numbers are trustworthy for the political sub-claim. Stage 2 swap stress test was not performed this pass due to the MAX_VERIFY_CLAIMS=1 cap (C5 was selected as the more publication-critical claim). To swap-test C1: `/auto-verify C1 — resume: true` (single-claim mode; Phase 2 audit reused). Suggested model swap: DeepSeek-R1-Distill-Llama-8B (same 8B scale, different training regime).
