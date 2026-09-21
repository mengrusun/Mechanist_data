## C2: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  ⚪ INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C2: supported (non-inferior)
- Main-experiment integrity: warn
- warn_source: experiment
- Variants: none  (Stage 2 skipped — C2 admitted by Phase 2 but not the top-1-by-importance claim picked at Phase 3 step 0 under MAX_VERIFY_CLAIMS=1)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: C2's validity non-inferiority methodology was audited at Phase 2 and found sound (pre-registered margin −0.05, one-sided 95% LB −0.035 > margin; full-set validity rate; honest frontier showing coef≥2 fails NI). It carries a WARN on experiment-audit check E: the claim wording "off-target properties not degraded" is optimistic against the documented CAA-specific off-target shifts at the winning setting (protein length −36.6 aa ≈ −37%; GC −0.139), which are transparently reported and pre-registered as "reported, not disqualifying." No swap stress test was attempted this pass (MAX_VERIFY_CLAIMS=1 selected the more central C1). To swap-test C2 later without re-auditing:
- `/auto-verify C2 — resume: true`  (single-claim mode; Phase 2 audit reused via RESUME; Stages 2–3 execute for C2)
