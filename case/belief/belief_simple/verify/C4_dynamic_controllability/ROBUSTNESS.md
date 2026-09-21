## C4: audit-only  (main-experiment integrity = PASS, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C4: supported
- Main-experiment integrity: pass
- warn_source: null
- Variants: none  (Stage 2 skipped — MAX_VERIFY_CLAIMS=1 cap; C2 selected as top-K pick by importance)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology (evaluation honesty and mechanism integrity per HARD CONSTRAINTS: probing layers before H*, alpha on matched-frame heads only, OOD split disjoint) was audited at Phase 2 and found trustworthy (PASS). C4's controller achieves net_impr=+151 (pythia-1b) and +27 (pythia-2.8b) OOD, beating the prompt-hint baseline (net_impr=-1 and -87 respectively). No swap stress test was attempted this pass — C4 was admitted but not selected as top-K (C2 selected instead). To upgrade: `/auto-verify C4 -- resume: true`.
