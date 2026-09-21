# Integrity Audit

**Overall**: FAIL
**Main-experiment integrity (Phase 2)**: FAIL    ← C1 combined = FAIL (exp=WARN, mech=FAIL)
**Variant integrity (Phase 9)**: [skipped — all main-experiment audits FAIL]

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | WARN | FAIL | FAIL | INCONCLUSIVE (mechanism broken) | verify/C1_verbal_confidence_cache/main_experiment_audit/ |

> **Column glossary:**
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim (WARN: P2 ratio aggregation ambiguous, P5 cross-seed inconsistency, answer_acc_preserved trivially 1.0 by design, seed2024 still running).
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. FAIL: M5 steering shows no independent capability metric, target effect within noise floor (span ~1.5 units vs std ~41), no locked α, no random-direction control.
> - **Combined** — `max_severity(exp=warn, mech=fail) = fail`.
> - **Gate decision** — C1 is INCONCLUSIVE: main-experiment mechanism rigor broken (Phase 2). Phases 3–10 skipped for C1.

## Variant integrity (Phase 9)

[skipped — all main-experiment audits FAIL]
