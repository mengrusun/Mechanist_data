## C2: audit-only  (main-experiment integrity = PASS, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C2: supported
- Main-experiment integrity: pass
- warn_source: null
- Variants: none  (Stage 2 skipped — MAX_VERIFY_CLAIMS=1 cap; C1 was picked as the top-importance claim)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology (evaluation and mechanism) was audited and found trustworthy at Phase 2 (PASS). C2's evidence — b*=block 47/60, top-block ratio 11.30×, top-1 SVD variance 0.300 — is well-verified with all numbers matching on disk. No swap stress test was attempted this pass because MAX_VERIFY_CLAIMS=1 and C1 was selected as the highest-importance audit target (phenomenon-validation gate with residue override caveat). C2's main-experiment SUPPORTED verdict stands as-is, with the caveat that it has not been shown robust across method/dataset/model swaps. To upgrade to full verification without re-auditing Phase 2: `/auto-verify C2 -- resume: true` (single-claim mode; Phase 2 audit reused via RESUME, only Stages 2–3 execute).
