# Integrity Audit

**Overall**: WARN
**Main-experiment integrity (Phase 2)**: WARN
**Variant integrity (Phase 9)**: skipped (budget-blocked — full variant run not launched)

---

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | WARN | N/A | WARN | continue-with-warn (experiment) | verify/C1_probe_existence_signal/main_experiment_audit/ |
| C2 | WARN | N/A | WARN | continue-with-warn (experiment) | verify/C2_aggregation_diversity/main_experiment_audit/ |
| C3 | PASS | N/A | PASS | continue | verify/C3_zero_shot_transfer/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. N/A when the claim uses no mechanism intervention (all three claims use passive probing with no additive intervention).
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a`.
> - **Gate decision** — what Phase 2 does with this claim.

### Phase 2 log

```
[main-experiment-audit] claim=C1 exp_verdict=warn mech_verdict=n/a combined=warn
[main-experiment-audit] claim=C1 exp_source=verify/C1_probe_existence_signal/main_experiment_audit/EXPERIMENT_AUDIT.json
[main-experiment-audit] claim=C1 mech_source=verify/C1_probe_existence_signal/main_experiment_audit/MECHANISM_AUDIT.json
[main-experiment-audit] claim=C1 action=continue-with-warn

[main-experiment-audit] claim=C2 exp_verdict=warn mech_verdict=n/a combined=warn
[main-experiment-audit] claim=C2 exp_source=verify/C2_aggregation_diversity/main_experiment_audit/EXPERIMENT_AUDIT.json
[main-experiment-audit] claim=C2 mech_source=verify/C2_aggregation_diversity/main_experiment_audit/MECHANISM_AUDIT.json
[main-experiment-audit] claim=C2 action=continue-with-warn

[main-experiment-audit] claim=C3 exp_verdict=pass mech_verdict=n/a combined=pass
[main-experiment-audit] claim=C3 exp_source=verify/C3_zero_shot_transfer/main_experiment_audit/EXPERIMENT_AUDIT.json
[main-experiment-audit] claim=C3 mech_source=verify/C3_zero_shot_transfer/main_experiment_audit/MECHANISM_AUDIT.json
[main-experiment-audit] claim=C3 action=continue

[verify] Stage 1 verdicts → admitted: [C1 (WARN), C2 (WARN), C3 (PASS)]; rejected → INCONCLUSIVE: []
```

---

## Variant integrity (Phase 9)

[skipped — variant run not completed due to budget constraint]

**C1 / model-swap-qwen3-32b-bf16**: Sanity run started (2/6 scenarios extracted cleanly) but the session died before completion. Full run (282 scenarios, ~10.45 GPU-h) exceeds remaining budget (~2.59 GPU-h). No activations.pt, no probe outputs, no result.json were written. Phase 9 integrity audit cannot execute on a variant that produced no artifacts. N_eligible = 0.

Round-End Decision required — see `verify/C1_probe_existence_signal/ROBUSTNESS.md` for options.
