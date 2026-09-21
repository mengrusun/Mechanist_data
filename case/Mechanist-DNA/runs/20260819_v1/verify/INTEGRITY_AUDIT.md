# Integrity Audit

**Overall**: WARN    <- max severity across main experiment + variants (variants: Phase 9)
**Main-experiment integrity (Phase 2)**: WARN    <- max severity across main-experiment combined verdicts
**Variant integrity (Phase 9)**: [filled by Phase 9]

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | WARN | N/A | WARN | continue-with-warn (experiment) | verify/C1_helix_selective_features/main_experiment_audit/ |
| C2 | PASS | PASS | PASS | continue | verify/C2_amplification_raises_helix/main_experiment_audit/ |
| C3 | WARN | PASS | WARN | continue-with-warn (experiment) | verify/C3_doseresponse_optimum_alpha/main_experiment_audit/ |

> Column glossary: Exp.=/experiment-audit overall; Mech.=/mechanism-audit overall (N/A when no intervention); Combined=max_severity(exp,mech) with n/a=pass; Gate=Phase 2 action.

## Variant integrity (Phase 9)
[filled by Phase 9 — only for claims admitted by Phase 2]
