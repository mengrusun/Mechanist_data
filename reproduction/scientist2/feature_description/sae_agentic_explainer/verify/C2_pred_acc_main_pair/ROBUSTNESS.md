## C2: audit-only (main-experiment integrity = PASS, swap stress test skipped) — INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C2: not-supported (Δpearson=-0.018, CI[-0.094,+0.058], p=0.43; provisional under-power)
- Main-experiment integrity: pass
- warn_source: null
- Variants: none (Stage 2 skipped — max_verify_claims_cap; C4 selected as top-1)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology for C2 is sound (PASS). The null result (Δpearson=-0.018, not significant) is real but provisional at n=44/300. No swap stress test was attempted. A model-swap variant would test whether the null result is robust across different LLM+SAE pairs; if GPT-OSS-20B + resid-post-aa also shows a null or negative Δpearson, that is a robustly negative result (PASS for FAIL direction). The under-power caveat limits interpretation.

To upgrade: `/auto-verify C2 -- resume: true`
