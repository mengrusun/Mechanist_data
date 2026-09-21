## C3a: audit-only  (main-experiment integrity = PASS, swap stress test skipped)  ->  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C3a: supported
- Main-experiment integrity: pass
- warn_source: null
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology for C3a was audited and found clean (PASS). GT provenance, score normalization, result file existence, dead code, and scope all pass. C3a was admitted by Phase 2 but was not selected as the top-K pick — CM was picked instead as the more scientifically central claim. The argmax-flip finding (fear->surprise between GSM8K/MedQA and SocialIQA) is methodologically sound but has not been stress-tested under model swaps. To upgrade:

- `/auto-verify C3a -- resume: true` (single-claim mode; Phase 2 audit reused)
