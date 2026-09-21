# Mechanism Audit — C2c: Gain is Alignment-Specific

**Claim**: C2c — Spearman gain is specifically attributable to human alignment, not generic KD.
**Mechanism family**: Representation and Parameter Analysis / Parameter-Space Task Vectors
**Role**: Specificity check — M3 (τ_aligned) vs M5 (τ_control); both at coef=1.0; C2c tests that τ_aligned ≠ τ_control in the human-similarity direction.

## Check A — Steering Coefficient Sweep
**Finding**: N/A — both M3 and M5 apply their respective task vectors at coef=1.0. The specificity check is a two-point comparison (τ_aligned vs τ_control), not a scaling sweep. This is appropriate for the claim (does alignment-specific information in τ matter?) rather than a dose-response characterization. N/A.
**Severity**: N/A
## Checks B–F: not_implemented
## Overall Verdict: **overall_verdict: n/a**
