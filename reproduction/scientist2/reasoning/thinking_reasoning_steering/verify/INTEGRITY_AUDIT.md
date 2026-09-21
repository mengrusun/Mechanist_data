# Integrity Audit

**Overall**: WARN
**Main-experiment integrity (Phase 2)**: WARN (C1 WARN, C2 WARN, C3 FAIL → INCONCLUSIVE, C4 WARN)
**Variant integrity (Phase 9)**: WARN

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | WARN | N/A | WARN | continue-with-warn (experiment — LLM-judge proxy annotation) | verify/C1_linear_behaviour_directions/main_experiment_audit/ |
| C2 | WARN | WARN | WARN | continue-with-warn (experiment + mechanism) | verify/C2_small_pool_extractability/main_experiment_audit/ |
| C3 | WARN | FAIL | FAIL | INCONCLUSIVE (mechanism broken — α range < 3 OOM, no random-direction control, no clean plateau) | verify/C3_dose_response_causal_control/main_experiment_audit/ |
| C4 | WARN | WARN | WARN | continue-with-warn (experiment + mechanism) | verify/C4_finer_than_prompt_engineering/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. `N/A` when no mechanism intervention used (C1/M1 is probe-only).
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a`.
> - **Gate decision** — what Phase 2 does with this claim.

## Variant integrity (Phase 9)

**Variants audited**: 1  (0 pass, 1 warn, 0 fail)

### Per-claim audit verdicts

| Claim | Exp. audit (variants) | Mech. audit (variants) | Combined |
|-------|-----------------------|------------------------|----------|
| C1 | — (INTEGRITY_ONLY — max_verify_claims_cap) | — | — |
| C2 | — (INTEGRITY_ONLY — max_verify_claims_cap) | — | — |
| C3 | — (INCONCLUSIVE — skipped) | — | — |
| C4 (picked) | WARN | WARN | WARN |

### Findings (per variant)
- C4 / variant model-swap-qwen-14b: [INTEGRITY: WARN — experiment + mechanism] — Evaluation methodology uses same LLM-judge proxy pipeline as main experiment (by design); mechanism inherits M3's under-validated α_op. Neither is a FAIL (design intent disclosed, coherence logged). Variant admitted to robustness computation.
