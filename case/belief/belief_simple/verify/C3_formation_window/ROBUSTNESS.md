## C3: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C3: supported
- Main-experiment integrity: warn
- warn_source: experiment  (scope: 24 of 154 planned pythia-1b checkpoints available on disk — environmental gap, documented honestly)
- Variants: none  (Stage 2 skipped — MAX_VERIFY_CLAIMS=1 cap; C2 selected as top-K pick by importance)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology was audited at Phase 2 and found WARN (scope: 24/154 checkpoints is a genuine limitation but properly documented as an environmental constraint). The formation windows [13000,13000] (personal) vs [0,33000] (attributed) are clearly distinct at the available log-spaced resolution. C3 was admitted but not selected as the top-K picked claim (C2 selected instead). To upgrade to full verification: `/auto-verify C3 -- resume: true` (single-claim mode; Phase 2 audit is reused via RESUME). Iteration note: the WARN is informational — the main experiment's distinctness claim is well-supported at coarse resolution; finer resolution would require additional checkpoint downloads.
