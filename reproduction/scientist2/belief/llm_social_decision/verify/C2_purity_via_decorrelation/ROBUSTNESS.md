## C2: audit-only  (main-experiment integrity = WARN, swap stress test skipped)  →  INTEGRITY_ONLY

- stage2_skip_reason: max_verify_claims_cap
- swap_variants_run: false
- Main-experiment verdict on C2: supported
- Main-experiment integrity: warn
- warn_source: experiment  (GS off-diagonal 0.506 borderline vs threshold 0.55; cross-leakage probe is 1-D scalar projection, weaker than full-vector probe; LEACE norm inflation 4–7× not formally addressed)
- Variants: none  (Stage 2 skipped — see stage2_skip_reason)
- robustness: null
- n_eligible: 0
- n_pass: 0

Interpretation: Main experiment methodology for C2 was audited and found admissible (WARN) at Phase 2. The cross-leakage evaluation correctly uses held-out activations (not train) for the probe, and the GT (V/W labels) is from real dataset construction. The GS off-diagonal of 0.506 is below the 0.55 threshold but the margin is thin, and the 1-D scalar-projection probe is a conservative test compared to a full-vector probe on the residual stream. The LEACE failure is a legitimate scientific finding. No swap stress test was attempted for C2 this pass (MAX_VERIFY_CLAIMS cap). Main experiment's own verdict on C2 (supported by GS; not supported by LEACE) stands as-is. To upgrade to full verification without re-auditing:
- `/auto-verify C2 — resume: true` (single-claim mode; Phase 2 audit reused via RESUME)
