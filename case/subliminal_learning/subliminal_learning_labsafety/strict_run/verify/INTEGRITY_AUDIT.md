# Integrity Audit

**Overall**: FAIL
**Main-experiment integrity (Phase 2)**: FAIL (C2 mechanism-audit FAIL drives this)
**Variant integrity (Phase 9)**: [to be filled by Phase 9]

## Main-experiment integrity (Phase 2, per-claim)

| Claim | Exp. audit | Mech. audit | Combined | Gate decision | Detail |
|-------|------------|-------------|----------|---------------|--------|
| C1 | PASS | N/A | PASS | continue | verify/C1_covert_transfer_phenomenon/main_experiment_audit/ |
| C2 | WARN | FAIL | FAIL | INCONCLUSIVE (mechanism broken) | verify/C2_bounded_null_mechanism/main_experiment_audit/ |

> Column glossary:
> - **Exp. audit** — `overall_verdict` of `/experiment-audit` on this claim.
> - **Mech. audit** — `overall_verdict` of `/mechanism-audit` on this claim. `N/A` when the claim uses no mechanism intervention.
> - **Combined** — `max_severity(exp, mech)` with `fail > warn > pass > n/a`.
> - **Gate decision** — what Phase 2 does with this claim.

**C1 Phase 2 summary**: PASS — behavioral phenomenon claim (M0 only), pure evaluation with real GT, all 6 experiment-audit checks pass, mechanism audit N/A (no steering/CAA intervention in M0). Admitted to Stages 2–3.

**C2 Phase 2 summary**: FAIL — mechanism claim (M1+M2 milestones), experiment-audit WARN (external reviewer noted incomplete artifact traceability; executor post-hoc verification resolves C/D to PASS, but WARN preserved per reviewer-independence protocol), mechanism-audit FAIL (Check A: no independent capability metric logged at any alpha sweep point; sign pattern broken — median Spearman rho=+0.371, 2/3 seeds show positive rho contrary to expected negative direction). Combined = FAIL → INCONCLUSIVE.

## Variant integrity (Phase 9)

> Only for claims in `picked_claims` (Phase 3 step 0). C2 was INCONCLUSIVE (Phase 2 FAIL) — skipped. C1 was picked.

| Claim | Variant tag | Exp. audit | Mech. audit | Combined | Eligible? | Verdict |
|-------|-------------|------------|-------------|----------|-----------|---------|
| C1 | method-swap-rule-based-scorer | PASS | N/A | PASS | yes | pass (consistent_with_main_experiment=pass) |

**Summary**: 1 variant audited — 1 pass, 0 warn, 0 fail. N_eligible = 1, N_pass = 1.

**C1 variant audit summary**: The rule-based scorer re-uses the same saved model-output JSONL files as the main experiment, loading gold from the authoritative QA_I parquet (not JSONL), with a fixed denominator of 133 (asserted). No steering/CAA in variant code — mechanism audit N/A. All 6 experiment-audit checks pass or warn (non-blocking). Combined = PASS. Variant is integrity-clean and counted in the robustness numerator.

Variant integrity artifacts:
- `verify/C1_covert_transfer_phenomenon/variant_audit/EXPERIMENT_AUDIT.{md,json}`
- `verify/C1_covert_transfer_phenomenon/variant_audit/MECHANISM_AUDIT.{md,json}`
